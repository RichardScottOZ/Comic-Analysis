#!/usr/bin/env python3
"""
Check Metadata Lengths & Quality
Parses the full manifest using the embedding script's logic and reports max lengths and sample data.
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

MANIFEST = "manifests/calibrecomics-extracted_manifest.csv"

def clean_series_name(name: str) -> str:
    name_no_volume = re.sub(r'\s+(v\d+|\(\d{4}-\d{4}\)|\d{4})', '', name)
    cleaned = re.sub(r'[^\w\s]', '', name_no_volume)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned.lower()

def extract_metadata(path: str, cid: str):
    parts = Path(path).parts
    filename = parts[-1] if parts else ""
    meta = {'series': 'unknown', 'volume': 'unknown', 'issue': '000', 'page': 'p000', 'source': 'unknown'}
    
    if 'amazon' in path.lower(): meta['source'] = 'amazon'
    elif 'calibre' in path.lower(): meta['source'] = 'calibre'
    
    page_match = re.search(r'p(\d{3,4})', filename)
    if page_match: meta['page'] = f"p{page_match.group(1)}"
    
    issue_match = re.search(r'(\d{3})', filename)
    if issue_match: meta['issue'] = issue_match.group(1)
    
    if len(parts) >= 2:
        parent = parts[-2]
        meta['series'] = clean_series_name(parent)
        vol_match = re.search(r'(v\d+|\(\d{4}-\d{4}\)|\d{4})', parent)
        if vol_match: meta['volume'] = vol_match.group(1)
        
    return meta

def check_manifest():
    print(f"Reading {MANIFEST}...")
    df = pd.read_csv(MANIFEST)
    
    max_lens = {'series': 0, 'volume': 0, 'issue': 0, 'page': 0, 'source': 0}
    samples = []
    
    calibre_audit_data = []
    seen_calibre_keys = set()

    print("Parsing metadata...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        path = row['absolute_image_path']
        cid = row['canonical_id']
        meta = extract_metadata(path, cid)
        
        # Collect Calibre Audit Data
        if meta.get('source') == 'calibre':
            parts = Path(path).parts
            raw_parent = parts[-2] if len(parts) >= 2 else ""
            raw_grandparent = parts[-3] if len(parts) >= 3 else ""
            
            # Key by folder structure to avoid 300k rows
            audit_key = (raw_parent, raw_grandparent)
            
            if audit_key not in seen_calibre_keys:
                seen_calibre_keys.add(audit_key)
                calibre_audit_data.append({
                    'canonical_id': cid,
                    'extracted_series': meta['series'],
                    'extracted_volume': meta['volume'],
                    'raw_parent': raw_parent,
                    'raw_grandparent': raw_grandparent,
                    'full_path': path
                })

        for k, v in meta.items():
            if len(v) > max_lens[k]:
                max_lens[k] = len(v)
        
        if idx % 100000 == 0:
            samples.append(meta)

    print("\n--- Max Lengths ---")
    for k, v in max_lens.items():
        print(f"{k}: {v}")

    # Save Calibre Audit
    if calibre_audit_data:
        print(f"\nSaving {len(calibre_audit_data)} unique Calibre folder patterns to calibre_metadata_audit.csv...")
        pd.DataFrame(calibre_audit_data).to_csv("calibre_metadata_audit.csv", index=False)

    print("\n--- Samples ---")
    for s in samples:
        print(s)

if __name__ == "__main__":
    check_manifest()
