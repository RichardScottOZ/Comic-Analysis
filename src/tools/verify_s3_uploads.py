#!/usr/bin/env python3
"""
Verify S3 Uploads (Strict Manifest Mode)
Simply checks if the expected JSON exists for a random sample of manifest rows.
"""

import csv
import argparse
import random
import boto3
from tqdm import tqdm

def verify_manifest_on_s3(manifest_path, prefix='vlm_analysis', sample_size=200):
    print(f"Loading manifest: {manifest_path}")
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    print(f"Total rows in manifest: {len(records)}")
    
    # Pick random sample
    sample = random.sample(records, min(sample_size, len(records)))
    print(f"Checking {len(sample)} random samples...")
    
    s3 = boto3.client('s3')
    found = 0
    missing = 0
    
    for row in tqdm(sample):
        cid = row['canonical_id']
        # The expected S3 key for the analysis is prefix + cid + .json
        expected_key = f"{prefix}/{cid}.json"
        
        # Get bucket from absolute_image_path (e.g. s3://calibrecomics-extracted/...)
        s3_url = row['absolute_image_path']
        bucket = s3_url.split('/')[2]
        
        try:
            s3.head_object(Bucket=bucket, Key=expected_key)
            found += 1
        except s3.exceptions.ClientError:
            missing += 1
            if missing <= 10:
                print(f"\n[MISSING] {expected_key} (ID: {cid})")

    print(f"\n--- Verification Results ---")
    print(f"Sample Size: {len(sample)}")
    print(f"Found on S3: {found}")
    print(f"Missing:     {missing}")
    
    success_rate = (found / len(sample)) * 100
    print(f"Verified Coverage: {success_rate:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--prefix', default='vlm_analysis')
    parser.add_argument('--sample', type=int, default=200)
    args = parser.parse_args()
    
    verify_manifest_on_s3(args.manifest, args.prefix, args.sample)