#!/usr/bin/env python3
"""
Audit MacOS Junk in Manifests
Compares Master Manifest (Local) vs Calibre Manifest (S3) to quantify junk files.
"""

import csv
import argparse

def analyze_manifest(path, name):
    print(f"\n--- Analyzing {name}: {path} ---")
    total = 0
    macos_junk = 0
    dot_underscore = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cid = row['canonical_id']
            path = row['absolute_image_path']
            
            if "__MACOSX" in cid or "__MACOSX" in path:
                macos_junk += 1
            elif "/._" in cid or "\\._" in cid: # Dot underscore files often inside MACOSX folder but check separately
                dot_underscore += 1
                
    clean_count = total - macos_junk
    print(f"Total Rows: {total}")
    print(f"MacOS Junk (__MACOSX): {macos_junk}")
    print(f"Dot Underscore (._): {dot_underscore}")
    print(f"Clean Count (Est): {clean_count}")
    return clean_count

def main():
    master = "manifests/master_manifest_20251229.csv"
    calibre = "manifests/calibrecomics-extracted_manifest.csv"
    
    clean_master = analyze_manifest(master, "Master Manifest")
    clean_calibre = analyze_manifest(calibre, "Calibre Manifest")
    
    print(f"\n--- Comparison ---")
    print(f"Master Clean:  {clean_master}")
    print(f"Calibre Clean: {clean_calibre}")
    print(f"Difference:    {abs(clean_master - clean_calibre)}")
    
    if clean_calibre > clean_master:
        print("\nNote: Calibre has MORE files. Are there other junk types? (Thumbs.db, .DS_Store?)")
    elif clean_master > clean_calibre:
        print("\nNote: Master has MORE files. Did some valid files get lost in S3 upload?")
    else:
        print("\nMATCH! The difference was exactly the junk files.")

if __name__ == "__main__":
    main()
