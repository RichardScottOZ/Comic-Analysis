#!/usr/bin/env python3
"""
Compare Panel Counts: Guided vs Unguided vs R-CNN
"""

import os
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def get_panel_count(json_path, source_type):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if source_type == 'rcnn':
            # R-CNN format: detections list with label='panel'
            # Filter by score > 0.5 to be fair
            panels = [d for d in data.get('detections', []) if d['label'] == 'panel' and d.get('score', 0) > 0.5]
            return len(panels)
        else:
            # VLM format: panels list
            panels = data.get('panels', [])
            return len(panels)
    except Exception:
        return -1

def compare_counts(guided_dir, unguided_dir, rcnn_dir, output_csv):
    print("Scanning directories...")
    
    # Map RelativePath -> {guided: n, unguided: n, rcnn: n}
    results = {}
    
    guided_path = Path(guided_dir)
    unguided_path = Path(unguided_dir)
    rcnn_path = Path(rcnn_dir)
    
    # 1. Walk Guided Directory (The Master List)
    print(f"Walking Guided VLM: {guided_dir}")
    files = list(guided_path.rglob('*.json'))
    print(f"Found {len(files)} guided results.")
    
    for f in files:
        try:
            rel_path = f.relative_to(guided_path)
            
            # Lookup counterparts
            u_file = unguided_path / rel_path
            r_file = rcnn_path / rel_path
            
            # Retrieve counts
            g_count = get_panel_count(f, 'vlm')
            u_count = get_panel_count(u_file, 'vlm') if u_file.exists() else -1
            r_count = get_panel_count(r_file, 'rcnn') if r_file.exists() else -1
            
            results[str(rel_path)] = {
                'guided': g_count,
                'unguided': u_count,
                'rcnn': r_count
            }
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # 4. Analyze
    print("Analyzing results...")
    
    guided_matches_rcnn = 0
    unguided_matches_rcnn = 0
    total_comparable = 0
    guided_better = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Relative_Path', 'RCNN_Count', 'Guided_Count', 'Unguided_Count', 'Guided_Diff', 'Unguided_Diff'])
        
        for rel_path, counts in results.items():
            r = counts['rcnn']
            g = counts['guided']
            u = counts['unguided']
            
            # Only compare if we have all three (or at least Guided vs RCNN)
            # Let's focus on the triple match for the "better" metric
            if r == -1:
                continue
                
            total_comparable += 1
            
            g_diff = g - r
            
            # Handle unguided missing
            if u != -1:
                u_diff = u - r
                if u_diff == 0: unguided_matches_rcnn += 1
                
                # Metric: Guided fixed a mistake
                if g_diff == 0 and u_diff != 0:
                    guided_better += 1
            else:
                u_diff = "N/A"

            if g_diff == 0: guided_matches_rcnn += 1
            
            writer.writerow([rel_path, r, g, u, g_diff, u_diff])

    print(f"\n--- Comparison Summary ({total_comparable} pages with R-CNN data) ---")
    print(f"Guided VLM matched R-CNN:   {guided_matches_rcnn} ({guided_matches_rcnn/total_comparable*100:.1f}%)")
    
    # Calculate unguided stats only for valid unguided files
    valid_unguided_comparisons = len([x for x in results.values() if x['unguided'] != -1 and x['rcnn'] != -1])
    if valid_unguided_comparisons > 0:
        print(f"Unguided VLM matched R-CNN: {unguided_matches_rcnn} ({unguided_matches_rcnn/valid_unguided_comparisons*100:.1f}%)")
        print(f"Cases where Guidance FIXED the count (Guided=RCNN, Unguided!=RCNN): {guided_better}")
    
    print(f"Detailed report saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--guided', default="E:/Comic_Analysis_Results_v2/vlm_lite_guided")
    parser.add_argument('--unguided', default="E:/Comic_Analysis_Results_v2/vlm")
    parser.add_argument('--rcnn', default="E:/Comic_Analysis_Results_v2/detections")
    parser.add_argument('--output', default="panel_count_comparison.csv")
    args = parser.parse_args()
    
    compare_counts(args.guided, args.unguided, args.rcnn, args.output)
