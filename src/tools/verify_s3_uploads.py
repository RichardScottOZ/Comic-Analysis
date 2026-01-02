#!/usr/bin/env python3
"""
Verify S3 Uploads for VLM Analysis
Checks if a random sample of canonical IDs from the manifest
actually exist in the S3 output bucket.
"""

import csv
import argparse
import random
import boto3
import json
from tqdm import tqdm

def verify_uploads(manifest_path, bucket, prefix, sample_size=50):
    print(f"Loading manifest: {manifest_path}")
    all_ids = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ids.append(row['canonical_id'])
            
    print(f"Total IDs in manifest: {len(all_ids)}")
    
    # Pick random sample
    sample = random.sample(all_ids, min(sample_size, len(all_ids)))
    print(f"Checking {len(sample)} random IDs in s3://{bucket}/{prefix}/...")
    
    s3 = boto3.client('s3')
    found = 0
    missing = 0
    valid_json = 0
    invalid_json = 0
    
    for cid in tqdm(sample):
        key = f"{prefix}/{cid}.json"
        try:
            # Check existence and fetch content
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read().decode('utf-8')
            found += 1
            
            # Validate JSON
            try:
                data = json.loads(content)
                # Check for key fields
                if 'overall_summary' in data or 'panels' in data:
                    valid_json += 1
                else:
                    # Maybe it's a raw error log?
                    if 'error' in data:
                        print(f"\n[WARN] {cid} exists but contains error: {data['error']}")
                    else:
                        print(f"\n[WARN] {cid} exists but has unexpected structure.")
                        invalid_json += 1
            except json.JSONDecodeError:
                print(f"\n[FAIL] {cid} exists but is invalid JSON.")
                invalid_json += 1
                
        except s3.exceptions.NoSuchKey:
            missing += 1
            if missing <= 5:
                print(f"[MISSING] {key}")
        except Exception as e:
            print(f"\n[ERR] Error checking {cid}: {e}")
            missing += 1

    print(f"\n--- Verification Results ---")
    print(f"Sample Size: {len(sample)}")
    print(f"Found on S3: {found}")
    print(f"Missing:     {missing}")
    print(f"Valid JSON:  {valid_json}")
    print(f"Invalid JSON:{invalid_json}")
    
    success_rate = (found / len(sample)) * 100
    print(f"Coverage Estimate: {success_rate:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify S3 Uploads')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--bucket', default='calibrecomics-extracted', help='S3 bucket')
    parser.add_argument('--prefix', default='vlm_analysis', help='S3 prefix')
    parser.add_argument('--sample', type=int, default=50, help='Number of items to check')
    
    args = parser.parse_args()
    
    verify_uploads(args.manifest, args.bucket, args.prefix, args.sample)
