"""
Create manifest CSV from comic pages stored in S3.

This script scans S3 buckets/prefixes for comic page images and creates a manifest CSV
compatible with OCR and PSS processing pipelines.

Usage:
    python create_s3_manifest.py \
        --bucket calibrecomics-extracted \
        --prefixes NeonIchiban/ CalibreComics_extracted_20251107/ \
        --output_csv s3_manifest_test.csv \
        --region us-east-1 \
        --sample 4000

    # Or scan all prefixes
    python create_s3_manifest.py \
        --bucket calibrecomics-extracted \
        --scan-all \
        --output_csv s3_manifest_all.csv \
        --region us-east-1
"""

import argparse
import csv
import boto3
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import sys


def list_s3_images(bucket: str, prefix: str, region: str, 
                   sample: Optional[int] = None,
                   image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')) -> List[dict]:
    """
    List image files in an S3 bucket/prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (directory path)
        region: AWS region
        sample: Optional limit on number of images to return
        image_extensions: Tuple of valid image extensions
        
    Returns:
        List of dicts with 'key' and 'size' for each image
    """
    s3_client = boto3.client('s3', region_name=region)
    
    images = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    print(f"  Scanning s3://{bucket}/{prefix}...")
    
    # Use paginator to handle large directories
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in page_iterator:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            # Check if it's an image file
            if any(key.lower().endswith(ext) for ext in image_extensions):
                images.append({
                    'key': key,
                    'size': obj['Size']
                })
                
                # Stop if we've reached sample limit
                if sample and len(images) >= sample:
                    print(f"  ✓ Reached sample limit of {sample} images")
                    return images
    
    print(f"  ✓ Found {len(images)} images")
    return images


def list_s3_prefixes(bucket: str, region: str, delimiter: str = '/') -> List[str]:
    """
    List top-level prefixes (directories) in an S3 bucket.
    
    Args:
        bucket: S3 bucket name
        region: AWS region
        delimiter: Delimiter for prefix separation (usually '/')
        
    Returns:
        List of prefix strings
    """
    s3_client = boto3.client('s3', region_name=region)
    
    print(f"Scanning for prefixes in s3://{bucket}/...")
    
    result = s3_client.list_objects_v2(
        Bucket=bucket,
        Delimiter=delimiter
    )
    
    prefixes = []
    if 'CommonPrefixes' in result:
        for prefix_obj in result['CommonPrefixes']:
            prefixes.append(prefix_obj['Prefix'])
    
    print(f"✓ Found {len(prefixes)} top-level prefixes")
    return prefixes


def create_s3_manifest(bucket: str, 
                       prefixes: List[str],
                       output_csv: str,
                       region: str = 'us-east-1',
                       sample: Optional[int] = None,
                       use_s3_paths: bool = False):
    """
    Create manifest CSV from S3 comic page images.
    
    Args:
        bucket: S3 bucket name
        prefixes: List of S3 prefixes to scan
        output_csv: Output CSV file path
        region: AWS region
        sample: Optional total sample limit across all prefixes
        use_s3_paths: If True, use s3:// URIs; if False, use boto3 URI format
    """
    print("--- Starting S3 Manifest Creation ---")
    print(f"Bucket: s3://{bucket}")
    print(f"Region: {region}")
    print(f"Prefixes: {len(prefixes)}")
    if sample:
        print(f"Sample limit: {sample} total images")
    print()
    
    all_images = []
    remaining_sample = sample
    
    # Scan each prefix
    for prefix in prefixes:
        if remaining_sample is not None and remaining_sample <= 0:
            print(f"Reached total sample limit of {sample} images")
            break
            
        prefix_sample = remaining_sample if remaining_sample else None
        images = list_s3_images(bucket, prefix, region, sample=prefix_sample)
        all_images.extend(images)
        
        if remaining_sample is not None:
            remaining_sample -= len(images)
    
    if not all_images:
        print("❌ No images found!")
        return
    
    print(f"\n📊 Total images found: {len(all_images)}")
    print(f"\n--- Writing manifest CSV ---")
    
    # Write manifest
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["canonical_id", "absolute_image_path"])
        
        for img in tqdm(all_images, desc="Writing manifest"):
            key = img['key']
            
            # Create canonical ID from key
            # Remove prefix directory structure and file extension
            canonical_id = Path(key).with_suffix('').as_posix()
            
            # Create absolute path
            if use_s3_paths:
                # Standard s3:// URI
                absolute_path = f"s3://{bucket}/{key}"
            else:
                # Boto3-compatible format (no protocol prefix)
                absolute_path = f"{bucket}/{key}"
            
            writer.writerow([canonical_id, absolute_path])
    
    print(f"\n✅ Manifest creation complete!")
    print(f"   Output: {output_csv}")
    print(f"   Total entries: {len(all_images):,}")
    
    # Calculate total size
    total_size_gb = sum(img['size'] for img in all_images) / (1024**3)
    print(f"   Total size: {total_size_gb:.2f} GB")
    
    # Show sample entries
    print(f"\n📄 Sample entries:")
    with open(output_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:6]:  # Header + 5 samples
            print(f"   {line.strip()}")
        if len(lines) > 6:
            print(f"   ... and {len(lines) - 6:,} more")


def main():
    parser = argparse.ArgumentParser(
        description="Create manifest CSV from comic pages in S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with NeonIchiban (small dataset)
  python create_s3_manifest.py \\
    --bucket calibrecomics-extracted \\
    --prefixes NeonIchiban/ \\
    --output_csv manifests/neon_test.csv \\
    --region us-east-1

  # Sample 4000 pages from new comics
  python create_s3_manifest.py \\
    --bucket calibrecomics-extracted \\
    --prefixes CalibreComics_extracted_20251107/ \\
    --output_csv manifests/test_4k.csv \\
    --sample 4000 \\
    --region us-east-1

  # Scan all top-level prefixes in bucket
  python create_s3_manifest.py \\
    --bucket calibrecomics-extracted \\
    --scan-all \\
    --output_csv manifests/all_comics.csv \\
    --region us-east-1

  # Multiple specific prefixes
  python create_s3_manifest.py \\
    --bucket calibrecomics-extracted \\
    --prefixes NeonIchiban/ CalibreComics_extracted/ CalibreComics_extracted_20251107/ \\
    --output_csv manifests/combined.csv \\
    --region us-east-1
        """
    )
    
    parser.add_argument('--bucket', type=str, required=True,
                       help='S3 bucket name (e.g., calibrecomics-extracted)')
    
    prefix_group = parser.add_mutually_exclusive_group(required=True)
    prefix_group.add_argument('--prefixes', type=str, nargs='+',
                             help='One or more S3 prefixes to scan (e.g., NeonIchiban/ CalibreComics_extracted/)')
    prefix_group.add_argument('--scan-all', action='store_true',
                             help='Scan all top-level prefixes in the bucket')
    
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output manifest CSV file path')
    
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region (default: us-east-1)')
    
    parser.add_argument('--sample', type=int, default=None,
                       help='Limit total number of images in manifest (for testing)')
    
    parser.add_argument('--use-s3-paths', action='store_true',
                       help='Use s3:// URI format instead of boto3 format (bucket/key)')
    
    args = parser.parse_args()
    
    # Get prefixes
    if args.scan_all:
        prefixes = list_s3_prefixes(args.bucket, args.region)
        if not prefixes:
            print("❌ No prefixes found in bucket")
            sys.exit(1)
    else:
        prefixes = args.prefixes
    
    # Create manifest
    create_s3_manifest(
        bucket=args.bucket,
        prefixes=prefixes,
        output_csv=args.output_csv,
        region=args.region,
        sample=args.sample,
        use_s3_paths=args.use_s3_paths
    )


if __name__ == "__main__":
    main()
