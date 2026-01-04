#!/usr/bin/env python3
"""
Unified JSON Generator for Stage 3 (Phase 1: Structure Only)
Combines CNN Panel Detections and VLM Text Content.
Simple, robust path matching.
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def filter_boxes(data):
    """Clean VLM data by removing coordinate keys."""
    if isinstance(data, dict):
        junk_keys = {'box_2d', 'box', 'polygon', 'detections', 'coordinates', 'bbox'}
        return {k: filter_boxes(v) for k, v in data.items() if k.lower() not in junk_keys}
    elif isinstance(data, list):
        return [filter_boxes(i) for i in data]
    else:
        return data

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def process_page(cnn_path, vlm_path, output_path):
    cnn_data = load_json(cnn_path)
    if not cnn_data:
        return "Missing CNN"
    
    img_w, img_h = cnn_data.get('image_size_wh', [0, 0])
    
    panels = []
    dets = cnn_data.get('detections', []) or cnn_data.get('panels', [])
    for det in dets:
        if det.get('label') == 'panel' and det.get('score', 1.0) > 0.4:
            box = det.get('box_xyxy') or det.get('bbox')
            x1, y1, x2, y2 = box
            panels.append({
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'confidence': float(det.get('score', 1.0)),
                'text': "" 
            })
    
    panels.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))

    vlm_text = ""
    vlm_data = load_json(vlm_path)
    if vlm_data:
        if 'OCRResult' in vlm_data: 
            vlm_data = vlm_data['OCRResult']
        cleaned = filter_boxes(vlm_data)
        vlm_text = json.dumps(cleaned, separators=(',', ':'))

    output_data = {
        "image_width": img_w,
        "image_height": img_h,
        "panels": panels,
        "full_page_text": vlm_text
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    return "Success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--detections-root', required=True)
    parser.add_argument('--vlm-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    if args.limit:
        print(f"Limiting to first {args.limit} pages.")
        df = df.head(args.limit)
    
    success = 0
    missing_cnn = 0
    
    print("Aligning Panels and Text...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cid = row['canonical_id']
        # Handle Windows paths if necessary
        cid_path = cid.replace('/', os.sep)
        
        # VLM Resolution
        vlm_path = None
        vlm_candidates = [
            os.path.join(args.vlm_root, f"{cid_path}.json"),
            os.path.join(args.vlm_root, "CalibreComics_extracted", f"{cid_path}.json")
        ]
        for p in vlm_candidates:
            if os.path.exists(p):
                vlm_path = p
                break
        
        # CNN Resolution (Direct Path Logic)
        cnn_path = os.path.join(args.detections_root, f"{cid_path}.json")
        
        # Fallback: Check for stripped prefix if direct path fails
        if not os.path.exists(cnn_path):
            parts = cid.split('/')
            if len(parts) > 1:
                stripped_cid = "/".join(parts[1:])
                stripped_path = stripped_cid.replace('/', os.sep)
                cnn_path_2 = os.path.join(args.detections_root, f"{stripped_path}.json")
                if os.path.exists(cnn_path_2):
                    cnn_path = cnn_path_2

        if args.debug and idx < 5:
            print(f"\n[DEBUG] CID: {cid}")
            print(f"  VLM: {vlm_path if vlm_path else 'NOT FOUND'}")
            print(f"  CNN: {cnn_path if cnn_path else 'NOT FOUND'}")

        if not cnn_path or not os.path.exists(cnn_path):
            missing_cnn += 1
            continue
            
        output_path = os.path.join(args.output_dir, f"{cid_path}.json")
        if process_page(cnn_path, vlm_path, output_path) == "Success":
            success += 1
            
    print(f"\n--- Alignment Complete ---")
    print(f"Processed: {len(df)}")
    print(f"Success: {success}")
    print(f"Missing CNN: {missing_cnn}")

if __name__ == "__main__":
    main()