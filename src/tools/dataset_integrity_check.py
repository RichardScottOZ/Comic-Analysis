#!/usr/bin/env python3
"""
Manifest-Driven Dataset Integrity Checker.
Verifies existence of VLM and Detection files for every entry in the Calibre Manifest.
Uses Suffix Matching (Proven Strategy) for bridging to Detections.
"""

import os
import csv
import argparse
from tqdm import tqdm
from pathlib import Path

def build_suffix_map(master_manifest):
    print("Indexing Master Manifest (Suffix Strategy)...")
    suffix_map = {} 
    with open(master_manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            master_id = row['canonical_id']
            # Map filename -> Master ID
            # Map stem -> Master ID
            # Map ID -> Master ID
            
            # Key 1: Filename
            local_path = row['absolute_image_path']
            filename = Path(local_path).name
            suffix_map[filename] = master_id
            
            # Key 2: Master ID itself
            suffix_map[master_id] = master_id
            
            # Key 3: Stem
            stem = Path(filename).stem
            suffix_map[stem] = master_id
            
    return suffix_map

def main():
    calibre_manifest = "manifests/calibrecomics-extracted_manifest.csv"
    master_manifest = "manifests/master_manifest_20251229.csv"
    
    vlm_root = "E:/vlm_recycling_staging"
    det_root = "E:/Comic_Analysis_Results_v2/detections"
    
    # 1. Build Bridge
    master_map = build_suffix_map(master_manifest)
    
    missing_vlm = []
    missing_det = []
    total = 0
    
    print("Checking Calibre Manifest entries...")
    with open(calibre_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            cid = row['canonical_id']
            if "__MACOSX" in cid: continue
            total += 1
            
            # 1. Check VLM (Direct Path)
            cid_path = cid.replace('/', os.sep)
            vlm_path = os.path.join(vlm_root, f"{cid_path}.json")
            
            # Try alternate VLM location (Strip 'CalibreComics_extracted/')
            if not os.path.exists(vlm_path):
                stripped = cid.replace("CalibreComics_extracted/", "")
                vlm_path = os.path.join(vlm_root, f"{stripped.replace('/', os.sep)}.json")
            
            if not os.path.exists(vlm_path):
                missing_vlm.append(cid)

            # 2. Check Detections (Suffix Bridge)
            # Try to match Calibre ID to Master ID using suffixes
            parts = cid.split('/')
            master_id = None
            
            # Filename match
            filename = parts[-1]
            master_id = master_map.get(filename)
            
            # Suffix match (e.g. #Guardian...)
            if not master_id:
                for i in range(len(parts)):
                    suffix = "/".join(parts[i:])
                    if suffix in master_map:
                        master_id = suffix
                        break
            
            det_path = None
            if master_id:
                det_path = os.path.join(det_root, f"{master_id}.json")
            
            if not det_path or not os.path.exists(det_path):
                missing_det.append(cid)

    # Write Report
    report_path = "integrity_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"--- Dataset Integrity Report ---\n")
        f.write(f"Total Checked: {total}\n")
        f.write(f"Missing VLM: {len(missing_vlm)} ({len(missing_vlm)/total*100:.2f}%)\n")
        f.write(f"Missing Det: {len(missing_det)} ({len(missing_det)/total*100:.2f}%)\n\n")
        
        if missing_vlm:
            f.write("-" * 20 + "\n[MISSING VLM]\n" + "-" * 20 + "\n")
            for c in missing_vlm:
                f.write(f"{c}\n")
                
        if missing_det:
            f.write("\n" + "-" * 20 + "\n[MISSING DETECTIONS]\n" + "-" * 20 + "\n")
            for c in missing_det:
                f.write(f"{c}\n")
                
    print(f"\n--- Summary ---")
    print(f"Total Checked: {total}")
    print(f"Missing VLM: {len(missing_vlm)} ({len(missing_vlm)/total*100:.2f}%)")
    print(f"Missing Det: {len(missing_det)} ({len(missing_det)/total*100:.2f}%)")
    print(f"Full report saved to: {report_path}")

if __name__ == "__main__":
    main()
