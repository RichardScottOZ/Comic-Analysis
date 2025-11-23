#!/usr/bin/env python
"""Lithops-compatible OCR processing for comic pages.

This module provides serverless, distributed OCR processing using Lithops
for massive parallelization across cloud functions (AWS Lambda, Azure Functions, etc.).

Similar to the embedding precomputation pipeline, this enables processing
thousands of comic pages in parallel using serverless infrastructure.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add ocr module to path
sys.path.insert(0, str(Path(__file__).parent))


def initialize_ocr_processor(ocr_method: str, ocr_config: Dict[str, Any]):
    """Initialize OCR processor once per worker (called by Lithops).
    
    Args:
        ocr_method: OCR method name ('tesseract', 'easyocr', 'paddleocr', etc.)
        ocr_config: Configuration dictionary for the OCR processor
        
    Returns:
        Initialized OCR processor
    """
    from ocr import create_ocr_processor
    
    print(f"Initializing OCR processor: {ocr_method}")
    ocr = create_ocr_processor(ocr_method, ocr_config)
    
    if not ocr.is_available():
        raise RuntimeError(f"OCR method {ocr_method} not available in this worker")
    
    return ocr


def process_image_lithops(task_data: Dict[str, Any], storage=None) -> Dict[str, Any]:
    """Process a single image with OCR in Lithops worker.
    
    This function is designed to be executed by Lithops workers in parallel.
    Each worker processes one comic page image.
    
    Args:
        task_data: Dict containing:
            - canonical_id: Unique identifier for the page
            - image_path: Path to the image file (can be local or S3/Azure URL)
            - ocr_method: OCR method to use
            - ocr_config: Configuration for the OCR processor
            - output_bucket: Cloud storage bucket for results (optional)
            - output_key_prefix: Prefix for output keys (optional)
        storage: Lithops storage backend instance (optional)
        
    Returns:
        Dict with processing status and metadata
    """
    canonical_id = task_data['canonical_id']
    image_path = task_data['image_path']
    ocr_method = task_data['ocr_method']
    ocr_config = task_data.get('ocr_config', {})
    
    try:
        # Initialize OCR processor (cached per worker)
        ocr = initialize_ocr_processor(ocr_method, ocr_config)
        
        # Handle cloud storage paths
        if storage and (image_path.startswith('s3://') or 
                       image_path.startswith('http://') or 
                       image_path.startswith('https://')):
            # Download image from cloud storage to /tmp
            import tempfile
            import requests
            
            # Parse cloud path
            if image_path.startswith('s3://'):
                # Extract bucket and key from s3://bucket/key
                parts = image_path[5:].split('/', 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ''
                
                # Download from S3
                image_data = storage.get_object(bucket=bucket, key=key)
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=Path(image_path).suffix, 
                    delete=False
                )
                temp_file.write(image_data)
                temp_file.close()
                local_image_path = temp_file.name
            else:
                # HTTP/HTTPS URL
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.jpg', 
                    delete=False
                )
                temp_file.write(response.content)
                temp_file.close()
                local_image_path = temp_file.name
        else:
            # Local path
            local_image_path = image_path
        
        # Check if image exists
        if not os.path.exists(local_image_path):
            return {
                'canonical_id': canonical_id,
                'status': 'error',
                'error': 'Image file not found',
                'ocr_method': ocr_method
            }
        
        # Process image with OCR
        results = ocr.process_image(local_image_path)
        
        # Prepare output data
        ocr_data = {
            'canonical_id': canonical_id,
            'source_image_path': image_path,
            'ocr_method': ocr_method,
            'timestamp': datetime.now().isoformat(),
            'text_regions': []
        }
        
        # Add OCR results
        for result in results:
            region = {
                'text': result.text,
                'confidence': result.confidence
            }
            
            if result.bbox:
                region['bbox'] = result.bbox
            
            if result.polygon:
                region['polygon'] = result.polygon
            
            if result.metadata:
                region['metadata'] = result.metadata
            
            ocr_data['text_regions'].append(region)
        
        # Add summary
        ocr_data['full_text'] = ocr.get_full_text(results)
        ocr_data['num_regions'] = len(results)
        
        # Save to cloud storage if storage backend is provided
        if storage and 'output_bucket' in task_data:
            output_bucket = task_data['output_bucket']
            output_key_prefix = task_data.get('output_key_prefix', 'ocr_results')
            output_key = f"{output_key_prefix}/{canonical_id}_ocr.json"
            
            # Serialize to JSON
            json_data = json.dumps(ocr_data, indent=2, ensure_ascii=False)
            
            # Upload to cloud storage
            storage.put_object(
                bucket=output_bucket,
                key=output_key,
                body=json_data.encode('utf-8')
            )
            
            result_status = {
                'canonical_id': canonical_id,
                'status': 'success',
                'ocr_method': ocr_method,
                'num_regions': len(results),
                'output_key': output_key,
                'output_bucket': output_bucket
            }
        else:
            # Return data directly (for local processing or custom storage)
            result_status = {
                'canonical_id': canonical_id,
                'status': 'success',
                'ocr_method': ocr_method,
                'num_regions': len(results),
                'ocr_data': ocr_data
            }
        
        # Cleanup temp file if created
        if local_image_path != image_path and os.path.exists(local_image_path):
            try:
                os.unlink(local_image_path)
            except (OSError, FileNotFoundError):
                pass
        
        return result_status
        
    except Exception as e:
        return {
            'canonical_id': canonical_id,
            'status': 'error',
            'error': str(e),
            'ocr_method': ocr_method
        }


def run_lithops_ocr(
    manifest_file: str,
    ocr_method: str,
    output_bucket: str,
    output_key_prefix: str = 'ocr_results',
    backend: str = 'aws_lambda',
    workers: int = 100,
    ocr_config: Dict[str, Any] = None,
    image_path_prefix: str = None
) -> List[Dict[str, Any]]:
    """Run OCR processing using Lithops for parallel processing.
    
    Args:
        manifest_file: Path to manifest CSV file (canonical_id, absolute_image_path)
        ocr_method: OCR method to use ('tesseract', 'easyocr', 'paddleocr', 'qwen', etc.)
        output_bucket: Cloud storage bucket name for output
        output_key_prefix: Prefix for output keys in cloud storage
        backend: Lithops backend ('aws_lambda', 'aws_batch', 'azure_functions', etc.)
        workers: Maximum number of parallel workers
        ocr_config: Configuration dict for OCR processor (e.g., {'lang': 'en', 'gpu': False})
        image_path_prefix: Prefix to prepend to image paths (e.g., 's3://bucket/' for S3 paths)
        
    Returns:
        List of processing results
        
    Example:
        >>> # Process with Tesseract on AWS Lambda
        >>> run_lithops_ocr(
        ...     manifest_file='manifest.csv',
        ...     ocr_method='tesseract',
        ...     output_bucket='comic-ocr-results',
        ...     backend='aws_lambda',
        ...     workers=500,
        ...     ocr_config={'lang': 'eng'}
        ... )
        
        >>> # Process with PaddleOCR on AWS Batch (GPU)
        >>> run_lithops_ocr(
        ...     manifest_file='manifest.csv',
        ...     ocr_method='paddleocr',
        ...     output_bucket='comic-ocr-results',
        ...     backend='aws_batch',
        ...     workers=50,
        ...     ocr_config={'lang': 'en', 'use_gpu': True}
        ... )
        
        >>> # Process with Qwen VLM
        >>> run_lithops_ocr(
        ...     manifest_file='manifest.csv',
        ...     ocr_method='qwen',
        ...     output_bucket='comic-ocr-results',
        ...     backend='aws_lambda',
        ...     workers=100,
        ...     ocr_config={'api_key': os.environ['OPENROUTER_API_KEY']}
        ... )
    """
    import csv
    import lithops
    
    if ocr_config is None:
        ocr_config = {}
    
    # Load manifest
    print(f"Loading manifest from: {manifest_file}")
    with open(manifest_file, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    
    print(f"Loaded {len(records)} images from manifest")
    
    # Prepare task data for workers
    tasks = []
    for record in records:
        canonical_id = record['canonical_id']
        image_path = record['absolute_image_path']
        
        # Prepend path prefix if provided (e.g., for S3 paths)
        if image_path_prefix and not image_path.startswith(('s3://', 'http://', 'https://', '/')):
            image_path = f"{image_path_prefix}/{image_path}"
        
        tasks.append({
            'canonical_id': canonical_id,
            'image_path': image_path,
            'ocr_method': ocr_method,
            'ocr_config': ocr_config,
            'output_bucket': output_bucket,
            'output_key_prefix': output_key_prefix
        })
    
    print(f"\nStarting Lithops OCR processing:")
    print(f"  Backend: {backend}")
    print(f"  OCR Method: {ocr_method}")
    print(f"  Workers: {workers}")
    print(f"  Images: {len(tasks)}")
    print(f"  Output Bucket: {output_bucket}")
    
    # Initialize Lithops executor
    executor = lithops.FunctionExecutor(backend=backend)
    
    # Map process_image_lithops across all tasks
    futures = executor.map(process_image_lithops, tasks)
    
    # Wait for all tasks to complete
    print("\nProcessing images in parallel...")
    results = executor.get_result(futures)
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    total_regions = sum(r.get('num_regions', 0) for r in results if r['status'] == 'success')
    
    print(f"\n{'='*60}")
    print(f"OCR Processing Complete")
    print(f"{'='*60}")
    print(f"  Successful: {successful}/{len(tasks)} images")
    print(f"  Failed: {failed}/{len(tasks)} images")
    print(f"  Total text regions: {total_regions}")
    print(f"  Output bucket: {output_bucket}/{output_key_prefix}")
    print(f"{'='*60}\n")
    
    # Print failed images if any
    if failed > 0:
        print("Failed images:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['canonical_id']}: {r.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lithops-based distributed OCR processing for comic pages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with Tesseract on AWS Lambda (500 workers)
  python batch_ocr_processing_lithops.py \\
    --manifest manifest.csv \\
    --method tesseract \\
    --output-bucket comic-ocr-results \\
    --backend aws_lambda \\
    --workers 500 \\
    --lang eng

  # Process with PaddleOCR on AWS Batch with GPU (50 workers)
  python batch_ocr_processing_lithops.py \\
    --manifest manifest.csv \\
    --method paddleocr \\
    --output-bucket comic-ocr-results \\
    --backend aws_batch \\
    --workers 50 \\
    --gpu

  # Process with Qwen VLM on AWS Lambda
  python batch_ocr_processing_lithops.py \\
    --manifest manifest.csv \\
    --method qwen \\
    --output-bucket comic-ocr-results \\
    --backend aws_lambda \\
    --workers 100 \\
    --api-key $OPENROUTER_API_KEY

  # Process with EasyOCR and S3 image paths
  python batch_ocr_processing_lithops.py \\
    --manifest manifest.csv \\
    --method easyocr \\
    --output-bucket comic-ocr-results \\
    --image-path-prefix s3://my-images \\
    --workers 200
        """
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest CSV file'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['tesseract', 'easyocr', 'paddleocr', 'qwen', 'gemma', 'deepseek'],
        help='OCR method to use'
    )
    parser.add_argument(
        '--output-bucket',
        type=str,
        required=True,
        help='Cloud storage bucket name for output'
    )
    parser.add_argument(
        '--output-key-prefix',
        type=str,
        default='ocr_results',
        help='Prefix for output keys in cloud storage (default: ocr_results)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='aws_lambda',
        choices=['aws_lambda', 'aws_batch', 'azure_functions', 'gcp_functions', 
                'ibm_cf', 'code_engine', 'localhost'],
        help='Lithops backend to use (default: aws_lambda)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=100,
        help='Maximum number of parallel workers (default: 100)'
    )
    parser.add_argument(
        '--image-path-prefix',
        type=str,
        help='Prefix to prepend to image paths (e.g., s3://bucket/)'
    )
    
    # OCR configuration options
    parser.add_argument(
        '--lang',
        type=str,
        default='en',
        help='Language code for OCR (default: en)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for OCR (EasyOCR/PaddleOCR on aws_batch)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help='API key for VLM OCR methods (or set OPENROUTER_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to use for VLM methods'
    )
    
    args = parser.parse_args()
    
    # Prepare OCR configuration
    ocr_config = {}
    
    if args.method == 'tesseract':
        ocr_config['lang'] = args.lang
    elif args.method == 'easyocr':
        ocr_config['languages'] = [args.lang]
        ocr_config['gpu'] = args.gpu
    elif args.method == 'paddleocr':
        ocr_config['lang'] = args.lang
        ocr_config['use_gpu'] = args.gpu
    elif args.method in ['qwen', 'gemma', 'deepseek']:
        if not args.api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set and --api-key not provided.")
            sys.exit(1)
        
        ocr_config['api_key'] = args.api_key
        if args.model:
            ocr_config['model'] = args.model
    
    # Run Lithops OCR processing
    results = run_lithops_ocr(
        manifest_file=args.manifest,
        ocr_method=args.method,
        output_bucket=args.output_bucket,
        output_key_prefix=args.output_key_prefix,
        backend=args.backend,
        workers=args.workers,
        ocr_config=ocr_config,
        image_path_prefix=args.image_path_prefix
    )
    
    # Save results summary
    summary_file = f"lithops_ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'manifest_file': args.manifest,
            'ocr_method': args.method,
            'backend': args.backend,
            'workers': args.workers,
            'output_bucket': args.output_bucket,
            'total_images': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'results': results
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
