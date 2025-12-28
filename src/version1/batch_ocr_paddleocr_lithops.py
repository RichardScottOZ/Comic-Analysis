#!/usr/bin/env python
"""
Lithops OCR processing using PaddleOCR (good for rotated/skewed text).
"""

import os
import sys
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def process_image_paddleocr(canonical_id, image_path, output_bucket, output_key_prefix) -> Dict[str, Any]:
    """Process a single image with PaddleOCR (pre-installed in runtime)."""

    import os

    os.environ['HOME'] = '/tmp'
    os.environ['XDG_CACHE_HOME'] = '/tmp/.cache'
    os.makedirs('/tmp/.cache', exist_ok=True)

    import tempfile
    import boto3
    from PIL import Image
    from paddleocr import PaddleOCR
    import os

    
    try:
        # Initialize PaddleOCR (models should be pre-downloaded, but allow download if needed)
        #ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)


        # Pass explicit directories for models and cache
        from paddleocr import PaddleOCR
        ocr_model_dir = '/tmp/paddleocr_models'
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            #model_storage_directory=ocr_model_dir,
            model_storage_directory='/tmp'
        )

        # Download image from S3
        s3_client = boto3.client('s3')
        
        # Parse S3 path
        if image_path.startswith('s3://'):
            parts = image_path[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        else:
            parts = image_path.split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            tmp_path = tmp.name
        
        # Run PaddleOCR
        try:
            results = ocr.ocr(tmp_path, cls=True)
            
            # Format results
            text_regions = []
            full_text_parts = []
            
            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence)
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    if text.strip():
                        # Convert bbox to [x, y, x2, y2] format
                        x_coords = [int(p[0]) for p in bbox]
                        y_coords = [int(p[1]) for p in bbox]
                        
                        text_regions.append({
                            'text': text,
                            'confidence': float(confidence),
                            'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                            'polygon': [[int(p[0]), int(p[1])] for p in bbox]
                        })
                        full_text_parts.append(text)
            
            result = {
                'OCRResult': {
                    'full_text': '\n'.join(full_text_parts),
                    'text_regions': text_regions
                },
                'metadata': {
                    'canonical_id': canonical_id,
                    'ocr_method': 'paddleocr',
                    'num_regions': len(text_regions)
                }
            }
        finally:
            os.unlink(tmp_path)
        
        # Save result to S3
        output_key = f"{output_key_prefix}/{canonical_id}_ocr.json"
        s3_client.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )
        
        return {
            'canonical_id': canonical_id,
            'status': 'success',
            'num_regions': len(text_regions),
            'output_key': output_key
        }
        
    except Exception as e:
        return {
            'canonical_id': canonical_id,
            'status': 'error',
            'error': str(e)
        }


def run_lithops_ocr_paddleocr(
    manifest_file: str,
    output_bucket: str,
    output_key_prefix: str = 'ocr_results_paddleocr',
    workers: int = 50
):
    """Run PaddleOCR using Lithops."""
    import lithops
    
    print(f"Loading manifest from: {manifest_file}")
    with open(manifest_file, 'r', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    
    print(f"Loaded {len(records)} images")
    
    # Prepare tasks
    tasks = []
    for record in records:
        tasks.append({
            'canonical_id': record['canonical_id'],
            'image_path': record['absolute_image_path'],
            'output_bucket': output_bucket,
            'output_key_prefix': output_key_prefix
        })
    
    print(f"\nStarting Lithops PaddleOCR processing:")
    print(f"  Workers: {workers}")
    print(f"  Images: {len(tasks)}")
    print(f"  Output: s3://{output_bucket}/{output_key_prefix}/")
    print(f"  Note: PaddleOCR handles rotated/skewed text well")
    print()
    
    # Execute with Lithops
    executor = lithops.FunctionExecutor()
    futures = executor.map(process_image_paddleocr, tasks)
    
    print("Processing images...")
    results = executor.get_result(futures)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n{'='*60}")
    print(f"PaddleOCR Processing Complete")
    print(f"{'='*60}")
    print(f"  Successful: {successful}/{len(tasks)}")
    print(f"  Failed: {failed}/{len(tasks)}")
    print(f"  Output: s3://{output_bucket}/{output_key_prefix}/")
    print(f"{'='*60}\n")
    
    if failed > 0:
        print("Failed images:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['canonical_id']}: {r.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lithops OCR with PaddleOCR (handles rotation/skew)')
    parser.add_argument('--manifest', required=True, help='Manifest CSV file')
    parser.add_argument('--output-bucket', required=True, help='S3 output bucket')
    parser.add_argument('--output-prefix', default='ocr_results_paddleocr', help='S3 output prefix')
    parser.add_argument('--workers', type=int, default=50, help='Number of workers')
    
    args = parser.parse_args()
    
    run_lithops_ocr_paddleocr(
        manifest_file=args.manifest,
        output_bucket=args.output_bucket,
        output_key_prefix=args.output_prefix,
        workers=args.workers
    )
