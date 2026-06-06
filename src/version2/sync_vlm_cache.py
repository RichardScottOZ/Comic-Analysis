#!/usr/bin/env python3
"""
Sync VLM JSON results from S3 to a local cache directory.

Downloads output from batch_vlm_analysis_lithops_v2.py (stored in S3 as
{prefix}/{canonical_id}.json) to a local mirror directory for Stage 3 training.

Features:
- Manifest-driven: only downloads pages listed in the master manifest
- Optional PSS label filter (--narrative-only for training datasets)
- Resume-safe: skips files that already exist locally
- Parallel downloads via ThreadPoolExecutor
"""

import os
import csv
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import boto3
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NARRATIVE_TYPES = {'narrative', 'story'}


def download_one(s3_client, bucket, s3_key, local_path, lock, counters, pbar):
    """Download a single VLM JSON from S3 to local disk. Thread-safe."""
    if local_path.exists():
        with lock:
            counters['skipped'] += 1
            pbar.update(1)
            pbar.set_postfix(ok=counters['success'], skip=counters['skipped'],
                             miss=counters['missing'], err=counters['error'], refresh=False)
        return 'skipped'

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        local_path.write_bytes(response['Body'].read())
        with lock:
            counters['success'] += 1
            pbar.update(1)
            pbar.set_postfix(ok=counters['success'], skip=counters['skipped'],
                             miss=counters['missing'], err=counters['error'], refresh=False)
        return 'success'
    except s3_client.exceptions.NoSuchKey:
        with lock:
            counters['missing'] += 1
            pbar.update(1)
            pbar.set_postfix(ok=counters['success'], skip=counters['skipped'],
                             miss=counters['missing'], err=counters['error'], refresh=False)
        return 'missing'
    except Exception as e:
        with lock:
            counters['error'] += 1
            if len(counters['errors']) < 20:
                counters['errors'].append(f"{s3_key}: {e}")
            pbar.update(1)
            pbar.set_postfix(ok=counters['success'], skip=counters['skipped'],
                             miss=counters['missing'], err=counters['error'], refresh=False)
        return 'error'


def main():
    parser = argparse.ArgumentParser(description='Sync VLM JSONs from S3 to local cache for Stage 3 training')
    parser.add_argument('--manifest', required=True,
                        help='Master manifest CSV (must have canonical_id column)')
    parser.add_argument('--cache-dir', required=True,
                        help='Local root directory to store downloaded JSONs')
    parser.add_argument('--bucket', default='calibrecomics-extracted',
                        help='S3 bucket name (default: calibrecomics-extracted)')
    parser.add_argument('--prefix', default='vlm_analysis_production',
                        help='S3 key prefix (default: vlm_analysis_production)')
    parser.add_argument('--pss-labels',
                        help='PSS labels JSON (required when using --narrative-only)')
    parser.add_argument('--narrative-only', action='store_true',
                        help='Only download narrative/story pages (requires --pss-labels)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Parallel download threads (default: 32)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of pages to download (for testing)')
    args = parser.parse_args()

    if args.narrative_only and not args.pss_labels:
        parser.error('--narrative-only requires --pss-labels')

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.rstrip('/')

    # Load PSS labels if filtering
    pss_labels = None
    if args.pss_labels:
        logger.info(f"Loading PSS labels: {args.pss_labels}")
        with open(args.pss_labels, 'r') as f:
            pss_labels = json.load(f)
        logger.info(f"  {len(pss_labels):,} entries loaded")

    # Build list of canonical_ids to sync
    logger.info(f"Reading manifest: {args.manifest}")
    canonical_ids = []
    skipped_type = 0
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cid = row['canonical_id']
            if args.narrative_only and pss_labels is not None:
                if pss_labels.get(cid, '') not in NARRATIVE_TYPES:
                    skipped_type += 1
                    continue
            canonical_ids.append(cid)

    if args.limit:
        canonical_ids = canonical_ids[:args.limit]

    scope = "narrative pages only" if args.narrative_only else "all pages"
    logger.info(f"Pages to sync : {len(canonical_ids):,} ({scope})")
    if skipped_type:
        logger.info(f"Skipped (non-narrative): {skipped_type:,}")

    if not canonical_ids:
        logger.warning("Nothing to download.")
        return

    s3 = boto3.client('s3')
    counters = {'success': 0, 'skipped': 0, 'missing': 0, 'error': 0, 'errors': []}
    lock = Lock()

    # Stream-submit so the progress bar appears immediately instead of blocking
    # while Python builds a 1.2M-entry futures dict
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with tqdm(total=len(canonical_ids), desc="Downloading", unit="file") as pbar:
            futures = {}
            for cid in canonical_ids:
                fut = executor.submit(
                    download_one,
                    s3,
                    args.bucket,
                    f"{prefix}/{cid}.json",
                    cache_dir / f"{cid}.json",
                    lock,
                    counters,
                    pbar
                )
                futures[fut] = cid

            for fut in as_completed(futures):
                pass  # progress already updated inside download_one

    logger.info(
        f"\n=== Sync Complete ===\n"
        f"  Downloaded : {counters['success']:,}\n"
        f"  Skipped    : {counters['skipped']:,} (already existed)\n"
        f"  Missing    : {counters['missing']:,} (not in S3 — VLM failed or not run)\n"
        f"  Errors     : {counters['error']:,}"
    )
    if counters['errors']:
        logger.warning("Sample errors (first 20):")
        for e in counters['errors']:
            logger.warning(f"  {e}")


if __name__ == '__main__':
    main()
