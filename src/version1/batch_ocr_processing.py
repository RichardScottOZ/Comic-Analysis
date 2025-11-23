#!/usr/bin/env python
"""Batch OCR processing for comic pages.

This script provides OCR capabilities complementary to VLM analysis,
specifically for extracting literal text from text pages, advertisements,
and other text-heavy content where VLM dialogue extraction may not be suitable.
"""

import argparse
import multiprocessing as mp
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import OCR factory
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ocr.factory import create_ocr_processor, list_available_methods


def process_single_image(args):
    """Process a single image with OCR.
    
    Args:
        args: Tuple of (record, output_dir, ocr_method, ocr_config)
        
    Returns:
        Dictionary with processing status
    """
    record, output_dir, ocr_method, ocr_config = args
    
    canonical_id = record['canonical_id']
    absolute_image_path = record['absolute_image_path']
    
    # Construct output path
    output_path = Path(output_dir) / f"{canonical_id}_ocr.json"
    
    # Skip if already processed
    if output_path.exists():
        return {'status': 'skipped', 'canonical_id': canonical_id}
    
    # Check if image exists
    if not os.path.exists(absolute_image_path):
        return {
            'status': 'error',
            'canonical_id': canonical_id,
            'error': 'Image file not found'
        }
    
    try:
        # Create OCR processor
        ocr = create_ocr_processor(ocr_method, ocr_config)
        
        # Check if OCR method is available
        if not ocr.is_available():
            return {
                'status': 'error',
                'canonical_id': canonical_id,
                'error': f'OCR method {ocr_method} not available'
            }
        
        # Process image
        results = ocr.process_image(absolute_image_path)
        
        # Prepare output data
        ocr_data = {
            'canonical_id': canonical_id,
            'source_image_path': absolute_image_path,
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
        
        # Add full text
        ocr_data['full_text'] = ocr.get_full_text(results)
        ocr_data['num_regions'] = len(results)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_data, f, indent=2, ensure_ascii=False)
        
        return {
            'status': 'success',
            'canonical_id': canonical_id,
            'num_regions': len(results)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'canonical_id': canonical_id,
            'error': str(e)
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Batch OCR processing for comic pages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OCR Methods:
  tesseract  - Tesseract OCR (CPU, requires pytesseract)
  easyocr    - EasyOCR (CPU/GPU, requires easyocr)
  paddleocr  - PaddleOCR (CPU/GPU, requires paddleocr)
  qwen       - Qwen VL via OpenRouter (requires OPENROUTER_API_KEY)
  gemma      - Gemma via OpenRouter (requires OPENROUTER_API_KEY)
  deepseek   - Deepseek via OpenRouter (requires OPENROUTER_API_KEY)

Examples:
  # Process with Tesseract
  python batch_ocr_processing.py --manifest manifest.csv --output_dir ocr_results --method tesseract

  # Process with EasyOCR on GPU
  python batch_ocr_processing.py --manifest manifest.csv --output_dir ocr_results --method easyocr --gpu

  # Process with Qwen VL
  python batch_ocr_processing.py --manifest manifest.csv --output_dir ocr_results --method qwen --api_key YOUR_KEY

  # Check available methods
  python batch_ocr_processing.py --list-methods
        """
    )
    
    parser.add_argument(
        '--manifest_file',
        type=str,
        help='Path to the master_manifest.csv file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for OCR JSON files'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tesseract',
        choices=['tesseract', 'easyocr', 'paddleocr', 'qwen', 'gemma', 'deepseek'],
        help='OCR method to use (default: tesseract)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Maximum number of worker processes (default: 4)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout for VLM API requests in seconds (default: 120)'
    )
    
    # CPU OCR options
    parser.add_argument(
        '--lang',
        type=str,
        default='en',
        help='Language code for OCR (default: en)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for OCR (EasyOCR/PaddleOCR)'
    )
    
    # VLM OCR options
    parser.add_argument(
        '--api_key',
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help='API key for VLM OCR methods (or set OPENROUTER_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to use (overrides default for method)'
    )
    
    # Utility options
    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List available OCR methods and exit'
    )
    
    args = parser.parse_args()
    
    # List available methods if requested
    if args.list_methods:
        print("Checking available OCR methods...\n")
        methods = list_available_methods()
        
        print("Available OCR Methods:")
        print("-" * 50)
        for method, available in methods.items():
            status = "✓ Available" if available else "✗ Not available"
            print(f"  {method:12} - {status}")
        
        print("\nNotes:")
        print("  - CPU methods require respective packages to be installed")
        print("  - VLM methods require OPENROUTER_API_KEY to be set")
        print("  - Install packages: pip install pytesseract easyocr paddleocr")
        return
    
    # Validate required arguments
    if not args.manifest_file or not args.output_dir:
        parser.error("--manifest_file and --output_dir are required (unless using --list-methods)")
    
    # Prepare OCR configuration
    ocr_config = {}
    
    if args.method in ['tesseract', 'easyocr', 'paddleocr']:
        ocr_config['lang'] = args.lang
        if args.method in ['easyocr', 'paddleocr']:
            ocr_config['gpu'] = args.gpu
            if args.method == 'easyocr':
                ocr_config['languages'] = [args.lang]
            else:  # paddleocr
                ocr_config['use_gpu'] = args.gpu
    
    elif args.method in ['qwen', 'gemma', 'deepseek']:
        if not args.api_key:
            print("Error: OPENROUTER_API_KEY environment variable not set and --api_key not provided.")
            return
        
        ocr_config['api_key'] = args.api_key
        ocr_config['timeout'] = args.timeout
        
        if args.model:
            ocr_config['model'] = args.model
    
    # Load manifest
    print(f"Loading manifest from: {args.manifest_file}")
    try:
        with open(args.manifest_file, 'r', encoding='utf-8') as f:
            records = list(csv.DictReader(f))
    except Exception as e:
        print(f"Fatal: Could not load manifest file: {e}")
        return
    
    print(f"Loaded {len(records)} records from manifest.")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (record, args.output_dir, args.method, ocr_config)
        for record in records
    ]
    
    # Process in parallel
    print(f"Processing {len(process_args)} images with {args.max_workers} workers...")
    print(f"OCR Method: {args.method}")
    print(f"Output Directory: {args.output_dir}")
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Adjust worker count for VLM methods to avoid rate limiting
    max_workers = args.max_workers
    if args.method in ['qwen', 'gemma', 'deepseek']:
        max_workers = min(max_workers, 2)
        print(f"Note: Limited to {max_workers} workers for VLM API to avoid rate limits")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, p_args) for p_args in process_args]
        
        with tqdm(total=len(futures), desc="Processing Images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['status'] == 'success':
                    successful += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"\nFailed to process {result['canonical_id']}: {result.get('error', 'Unknown error')}")
                
                pbar.set_description(
                    f"Success: {successful}, Skipped: {skipped}, Failed: {failed}"
                )
                pbar.update(1)
    
    print("\n--- OCR Processing Complete ---")
    print(f"✅ Successful: {successful}")
    print(f"⏭️  Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()
