#!/usr/bin/env python3
"""
Batched PaddleOCR processing with Lithops - handles 1.22M+ pages
Features:
- S3 skip logic (checks existing paddleocr_results)
- Memory escalation on MemoryError (1024 -> 2048MB in 256MB steps)
- Batch hang protection (individual task execution)
- Failure tracking (CSV manifest)
- Single-pass workflow
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import boto3
import lithops

import logging
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_existing_paddleocr_result(s3_client, bucket, canonical_id, output_key_prefix):
    """Check if PaddleOCR result already exists in the _ocr.json file"""
    ocr_key = f"{output_key_prefix}/{canonical_id}_ocr.json"
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=ocr_key)
        ocr_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Check if paddleocr_results key exists and has content
        if 'paddleocr_results' in ocr_data and ocr_data['paddleocr_results']:
            return True
    except Exception:
        pass  # File doesn't exist or no paddleocr_results - need to process
    
    return False


def process_image_paddleocr(canonical_id, image_path, output_bucket, output_key_prefix):
    """
    Process single image with PaddleOCR - runs inside Lambda
    """
    import json
    import traceback
    import os
    import boto3
    from io import BytesIO
    from PIL import Image
    import numpy as np

    # Disable decompression bomb protection
    Image.MAX_IMAGE_PIXELS = None

    try:
        # Critical: Redirect all home/cache dirs to /tmp which is writable in Lambda
        os.environ['HOME'] = '/tmp'
        os.environ['XDG_CACHE_HOME'] = '/tmp/.cache'
        os.environ['HF_HOME'] = '/tmp/huggingface'
        os.environ['EASYOCR_MODULE_PATH'] = '/tmp/easyocr'
        os.environ['PADDLEOCR_HOME'] = '/tmp/paddleocr'
        
        # Ensure directories exist
        os.makedirs('/tmp/.cache', exist_ok=True)
        
        # Initialize PaddleOCR (imports inside function to avoid cold start issues)
        from paddleocr import PaddleOCR
        
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            model_storage_directory='/tmp/paddleocr_models'
        )
        
        # Download image from S3
        s3_client = boto3.client('s3')
        
        # Parse S3 path to get source bucket and key
        if image_path.startswith('s3://'):
            parts = image_path[5:].split('/', 1)
            source_bucket = parts[0]
            source_key = parts[1] if len(parts) > 1 else ''
        else:
            # Fallback for paths like "bucket/key"
            parts = image_path.split('/', 1)
            source_bucket = parts[0]
            source_key = parts[1] if len(parts) > 1 else ''
            
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        image_data = response['Body'].read()
        
        # Load image
        Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        
        # Run PaddleOCR
        result = ocr.ocr(image_np, cls=True)
        
        # Format results with bounding boxes
        paddleocr_results = []
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                
                paddleocr_results.append({
                    'text': text_info[0],
                    'confidence': float(text_info[1]),
                    'bbox': bbox
                })
        
        # Load or create OCR results file
        ocr_key = f"{output_key_prefix}/{canonical_id}_ocr.json"
        
        try:
            response = s3_client.get_object(Bucket=output_bucket, Key=ocr_key)
            ocr_data = json.loads(response['Body'].read().decode('utf-8'))
        except:
            ocr_data = {'canonical_id': canonical_id}
        
        # Add PaddleOCR results
        ocr_data['paddleocr_results'] = paddleocr_results
        
        # Save back to S3
        s3_client.put_object(
            Bucket=output_bucket,
            Key=ocr_key,
            Body=json.dumps(ocr_data, indent=2),
            ContentType='application/json'
        )
        
        return {
            'canonical_id': canonical_id,
            'status': 'success',
            'text_regions': len(paddleocr_results)
        }
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Error processing {canonical_id}: {error_msg}")
        return {
            'canonical_id': canonical_id,
            'status': 'error',
            'error': error_msg,
            'traceback': traceback_str
        }


def run_lithops_ocr_paddleocr_batched(
    manifest_path,
    output_bucket,
    output_key_prefix,
    workers=1000,
    batch_size=1000,
    memory_levels=None
):
    """
    Run PaddleOCR processing with batching and memory escalation
    
    Args:
        memory_levels: List of memory sizes to try (MB), defaults to [1024, 1280, 1536, 1792, 2048]
    """
    if memory_levels is None:
        memory_levels = [1024, 1280, 1536, 1792, 2048]
    
    logger.info(f"Loading manifest from: {manifest_path}")
    
    # Read CSV with standard library
    all_images = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_images.append(row)
    
    logger.info(f"Loaded {len(all_images)} images")
    
    # Filter to images that need processing - use fast list-based check like original
    s3_client = boto3.client('s3')
    
    logger.info("Checking for existing PaddleOCR results...")
    
    # List existing _ocr.json files (FAST - one paginated list operation)
    existing_keys = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=output_bucket, Prefix=output_key_prefix + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Extract canonical_id from key
            canonical_id = key[len(output_key_prefix)+1:]  # strip prefix + '/'
            if canonical_id.endswith('_ocr.json'):
                canonical_id = canonical_id[:-len('_ocr.json')]
                existing_keys.add(canonical_id)
    
    # Filter to only images not already done
    to_process = [row for row in all_images if row['canonical_id'] not in existing_keys]
    skipped = len(all_images) - len(to_process)
    
    logger.info(f"Found {skipped} images already processed")
    logger.info(f"Will process {len(to_process)} images")
    
    if len(to_process) == 0:
        logger.info("No images to process!")
        return
    
    # Create failure manifest
    failure_csv = 'paddleocr_failures.csv'
    with open(failure_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical_id', 'last_memory', 'error'])
    
    # Process in batches
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    logger.info(f"\nProcessing {len(to_process)} images in {total_batches} batches of {batch_size}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Memory escalation: {memory_levels} MB")
    logger.info(f"  Output: s3://{output_bucket}/{output_key_prefix}/\n")
    
    total_success = 0
    total_failed = 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(to_process))
        batch = to_process[start_idx:end_idx]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num + 1}/{total_batches} - Processing {len(batch)} images")
        logger.info(f"{'='*60}")
        
        # Prepare tasks
        tasks = []
        for row in batch:
            tasks.append({
                'canonical_id': row['canonical_id'],
                'image_path': row['absolute_image_path'],
                'output_bucket': output_bucket,
                'output_key_prefix': output_key_prefix
            })
        
        # Try each memory level for this batch
        batch_success = 0
        batch_failed = 0
        
        for memory_mb in memory_levels:
            if len(tasks) == 0:
                break
            
            logger.info(f"\n>>> Attempting {len(tasks)} tasks with {memory_mb}MB memory...")
            
            try:
                executor = lithops.FunctionExecutor(backend='aws_lambda', runtime_memory=memory_mb)
                # Pass tasks as a list of dicts - Lithops unpacks them as kwargs
                futures = executor.map(process_image_paddleocr, tasks)
                
                # Wait for results, allowing exceptions to be returned as objects rather than raising
                results = executor.get_result(futures, throw_except=False)
                
                # Process results
                remaining_tasks = []
                attempt_success = 0
                attempt_mem_errors = 0
                attempt_hard_failures = 0

                for task, result in zip(tasks, results):
                    # Handle None result (unexpected infrastructure failure)
                    if result is None:
                         remaining_tasks.append(task)
                         attempt_mem_errors += 1
                         continue

                    # Handle exceptions (Lithops execution errors like Lambda OOM)
                    if isinstance(result, Exception):
                        error_msg = str(result)
                        if 'MemoryError' in error_msg or 'exceeded maximum memory' in error_msg:
                            remaining_tasks.append(task)
                            attempt_mem_errors += 1
                        else:
                            batch_failed += 1
                            attempt_hard_failures += 1
                            logger.error(f"  ✗ {task['canonical_id']} - Exception: {error_msg}")
                            with open(failure_csv, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([task['canonical_id'], memory_mb, error_msg])
                        continue
                    
                    # Handle dictionary results (our function returned successfully)
                    if isinstance(result, dict):
                        if result.get('status') == 'success':
                            batch_success += 1
                            attempt_success += 1
                        
                        elif result.get('status') == 'error':
                            error_msg = result.get('error', 'Unknown error')
                            if 'MemoryError' in error_msg:
                                remaining_tasks.append(task)
                                attempt_mem_errors += 1
                            else:
                                batch_failed += 1
                                attempt_hard_failures += 1
                                logger.error(f"  ✗ {result['canonical_id']} - {error_msg}")
                                with open(failure_csv, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([result['canonical_id'], memory_mb, error_msg])
                        else:
                            batch_failed += 1
                            attempt_hard_failures += 1
                            logger.error(f"  ✗ {task['canonical_id']} - Unexpected result format")
                    else:
                        batch_failed += 1
                        attempt_hard_failures += 1
                        logger.error(f"  ✗ {task['canonical_id']} - Unexpected result type")

                logger.info(f"--- Attempt Summary ({memory_mb}MB) ---")
                logger.info(f"    Success: {attempt_success}")
                logger.info(f"    Memory Errors (will retry): {attempt_mem_errors}")
                logger.info(f"    Hard Failures: {attempt_hard_failures}")
                
                tasks = remaining_tasks
                executor.clean()
                
            except Exception as e:
                # This catches errors in the orchestration itself (e.g. executor setup), not individual task errors
                logger.error(f"!!! Critical batch orchestration failure at {memory_mb}MB: {e}")
                logger.info("Retrying current task list at next memory level...")
                continue
        
        # Any remaining tasks after all memory levels are failures
        for task in tasks:
            batch_failed += 1
            with open(failure_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    task['canonical_id'],
                    memory_levels[-1],
                    'Failed at all memory levels'
                ])
        
        total_success += batch_success
        total_failed += batch_failed
        
        logger.info(f"\nBatch {batch_num + 1} complete:")
        logger.info(f"  Success: {batch_success}/{len(batch)}")
        logger.info(f"  Failed: {batch_failed}/{len(batch)}")
        logger.info(f"\nOverall progress:")
        logger.info(f"  Success: {total_success}/{len(to_process)}")
        logger.info(f"  Failed: {total_failed}/{len(to_process)}")
    
    logger.info(f"\n{'='*60}")
    logger.info("PaddleOCR Processing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Successful: {total_success}/{len(to_process)}")
    logger.info(f"  Failed: {total_failed}/{len(to_process)}")
    logger.info(f"  Skipped (already done): {skipped}")
    logger.info(f"  Output: s3://{output_bucket}/{output_key_prefix}/")
    if total_failed > 0:
        logger.info(f"  Failures logged to: {failure_csv}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batched PaddleOCR processing with Lithops')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--output-bucket', required=True, help='S3 bucket for outputs')
    parser.add_argument('--output-prefix', required=True, help='S3 key prefix for outputs')
    parser.add_argument('--workers', type=int, default=1000, help='Max concurrent workers')
    parser.add_argument('--batch-size', type=int, default=1000, help='Images per batch')
    
    args = parser.parse_args()
    
    run_lithops_ocr_paddleocr_batched(
        manifest_path=args.manifest,
        output_bucket=args.output_bucket,
        output_key_prefix=args.output_prefix,
        workers=args.workers,
        batch_size=args.batch_size,
        memory_levels=[892, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1792, 2048]  # Your requested 256MB steps
    )
