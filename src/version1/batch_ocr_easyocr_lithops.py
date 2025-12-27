#!/usr/bin/env python
"""
Lithops OCR processing using EasyOCR (better for comics).
"""

import os
import sys
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


def process_image_easyocr(canonical_id, image_path, output_bucket, output_key_prefix) -> Dict[str, Any]:
    """Process a single image with EasyOCR (pre-installed in runtime)."""
    import tempfile
    import boto3
    from PIL import Image
    import easyocr
    import os
    
    # Set all paths to /tmp (Lambda /home is read-only)
    os.environ['HOME'] = '/tmp'
    os.environ['EASYOCR_MODULE_PATH'] = '/tmp/easyocr'
    os.environ['XDG_CACHE_HOME'] = '/tmp/.cache'
    
    try:
        # Initialize reader (EasyOCR and models are pre-installed in the runtime)
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp/easyocr', user_network_directory='/tmp/easyocr', download_enabled=False)
        
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
        
        # Run EasyOCR
        try:
            results = reader.readtext(tmp_path)
            
            # Format results
            text_regions = []
            full_text_parts = []
            
            for (bbox, text, confidence) in results:
                if text.strip():
                    # bbox is list of 4 points, convert to [x, y, x2, y2]
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
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
                    'ocr_method': 'easyocr',
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


def run_lithops_ocr_easyocr(
    manifest_file: str,
    output_bucket: str,
    output_key_prefix: str = 'ocr_results_easyocr',
    workers: int = 50  # EasyOCR is slower, use fewer workers
):
    """Run EasyOCR using Lithops."""
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
    
    print(f"\nStarting Lithops EasyOCR processing:")
    print(f"  Workers: {workers}")
    print(f"  Images: {len(tasks)}")
    print(f"  Output: s3://{output_bucket}/{output_key_prefix}/")
    print(f"  Note: EasyOCR is slower but more accurate for comics")
    print()
    
    # Execute with Lithops
    executor = lithops.FunctionExecutor()
    futures = executor.map(process_image_easyocr, tasks)
    
    print("Processing images...")
    results = executor.get_result(futures)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n{'='*60}")
    print(f"EasyOCR Processing Complete")
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
    
    parser = argparse.ArgumentParser(description='Lithops OCR with EasyOCR (better for comics)')
    parser.add_argument('--manifest', required=True, help='Manifest CSV file')
    parser.add_argument('--output-bucket', required=True, help='S3 output bucket')
    parser.add_argument('--output-prefix', default='ocr_results_easyocr', help='S3 output prefix')
    parser.add_argument('--workers', type=int, default=50, help='Number of workers (EasyOCR is slower)')
    
    args = parser.parse_args()
    
    run_lithops_ocr_easyocr(
        manifest_file=args.manifest,
        output_bucket=args.output_bucket,
        output_key_prefix=args.output_prefix,
        workers=args.workers
    )
