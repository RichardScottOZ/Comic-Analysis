#!/usr/bin/env python3
"""
Collect FULL Model Comparisons for P002 and P003
Scans E:/openroutertests* directories and aggregates the FULL JSON content
for specific pages so Gemini can analyze them in depth.
"""

import os
import json
import glob
from pathlib import Path

# The specific pages we want to compare
TARGET_PAGES = [
    "#Guardian 001_#Guardian 001 - p002.jpg",
    "#Guardian 001_#Guardian 001 - p003.jpg"
]

# Root directories to scan
ROOT_DIRS = [
    "E:/openroutertests",
    "E:/openroutertests_premium"
]

def collect_comparisons(output_file="model_comparisons_p002_p003_FULL.json"):
    aggregated_data = {}

    for root in ROOT_DIRS:
        if not os.path.exists(root):
            print(f"Skipping missing root: {root}")
            continue
            
        print(f"Scanning {root}...")
        # Iterate over model subdirectories
        for model_dir in glob.glob(os.path.join(root, "*")):
            if not os.path.isdir(model_dir):
                continue
                
            model_name = os.path.basename(model_dir)
            
            # Initialize model entry if not exists
            if model_name not in aggregated_data:
                aggregated_data[model_name] = {}
            
            for page_id in TARGET_PAGES:
                # Try exact match first
                json_path = os.path.join(model_dir, f"{page_id}.json")
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            # Load full JSON content
                            data = json.load(f)
                            aggregated_data[model_name][page_id] = data
                    except Exception as e:
                        aggregated_data[model_name][page_id] = {"error": f"Read Failure: {str(e)}"}
                else:
                    aggregated_data[model_name][page_id] = {"status": "MISSING_FILE"}

    print(f"Collected data from {len(aggregated_data)} models.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"Saved FULL aggregation to: {output_file}")

if __name__ == "__main__":
    collect_comparisons()