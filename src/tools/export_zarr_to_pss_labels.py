#!/usr/bin/env python3
"""
Export PSS Labels from Zarr to JSON for Stage 3 Training.
"""

import argparse
import json
import xarray as xr
from tqdm import tqdm

# Class Mapping (Must match PSSDataset / classify_pages_zarr.py)
CLASS_NAMES = ["advertisement", "cover", "story", "textstory", "first-page", "credits", "art", "text", "back_cover"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', required=True, help='Path to Zarr store')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    args = parser.parse_args()
    
    print(f"Opening Zarr: {args.zarr}")
    ds = xr.open_zarr(args.zarr)
    
    ids = ds['ids'].values
    preds = ds['prediction'].values
    
    label_map = {}
    
    print("Exporting labels...")
    count = 0
    story_count = 0
    
    for i in tqdm(range(len(ids))):
        cid = ids[i]
        pred_idx = preds[i]
        
        # Skip unprocessed (-1)
        if pred_idx == -1:
            continue
            
        label_str = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "unknown"
        
        # Stage 3 expects "narrative" for story pages usually, 
        # but your dataset loader explicitly checks for 'story' or 'narrative'.
        # We will use the exact class name "story".
        
        label_map[cid] = label_str
        
        if label_str == 'story':
            story_count += 1
        count += 1
        
    print(f"Exported {count} labels.")
    print(f"Story pages: {story_count}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
