import csv
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def get_image_resolution(path_str):
    try:
        with Image.open(path_str) as img:
            w, h = img.size
            return w, h, w * h
    except Exception:
        return 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description="Find highest resolution images in manifest")
    parser.add_argument('--manifest', default='manifests/master_manifest_20251229.csv')
    parser.add_argument('--output', default='manifests/top_high_res_manifest.csv')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest {args.manifest} not found.")
        return

    print(f"Loading {args.manifest}...")
    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            records.append(row)
            
    print(f"Analyzing resolutions for {len(records)} records...")
    # Use tqdm for progress tracking
    for rec in tqdm(records, desc="Checking resolutions"):
        w, h, pixels = get_image_resolution(rec['absolute_image_path'])
        rec['_w'] = w
        rec['_h'] = h
        rec['_pixels'] = pixels
        
    print("Sorting by total pixels descending...")
    records.sort(key=lambda x: x['_pixels'], reverse=True)
    
    top_records = records[:args.top_k]
    
    print(f"
Top {args.top_k} Highest Resolution Files (Resolution Stress Test):")
    for rec in top_records:
        mp = rec['_pixels'] / 1_000_000
        print(f"  {rec['_w']} x {rec['_h']} ({mp:.2f} MP) | {rec['canonical_id']}")

    print(f"
Writing to {args.output}...")
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in top_records:
            # Clean up temporary keys before writing
            out_rec = {k: v for k, v in rec.items() if not k.startswith('_')}
            writer.writerow(out_rec)

    print("Done!")

if __name__ == "__main__":
    main()
