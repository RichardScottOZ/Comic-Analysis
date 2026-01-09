#!/usr/bin/env python3
"""
Unified JSON Generator for Stage 3 (Phase 2: With Text Alignment)
Combines CNN Panel Detections, VLM Text, and PaddleOCR Spatial Text.
Matches PaddleOCR text boxes to CNN panel boxes for panel-level text.
Fast path matching (No indexing).
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
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

def compute_iota(box1, box2):
    """Intersection over Text Area (box2 is text)."""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / area2 if area2 > 0 else 0.0

def process_page(cnn_path, vlm_path, ocr_path, output_path):
    # 1. Load CNN Detections
    cnn_data = load_json(cnn_path)
    if not cnn_data:
        return "Missing CNN"
    
    img_w, img_h = cnn_data.get('image_size_wh', [0, 0])
    
    # Extract panel boxes
    panels = []
    dets = cnn_data.get('detections', []) or cnn_data.get('panels', [])
    for det in dets:
        if det.get('label') == 'panel' and det.get('score', 1.0) > 0.4:
            box = det.get('box_xyxy') or det.get('bbox')
            x1, y1, x2, y2 = box
            panels.append({
                'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(det.get('score', 1.0)),
                'text_regions': []
            })
    
    # 2. Load PaddleOCR for spatial text matching
    ocr_data = load_json(ocr_path)
    if ocr_data and 'OCRResult' in ocr_data:
        regions = ocr_data['OCRResult'].get('text_regions', [])
        for reg in regions:
            t_box = reg.get('bbox') # [x1, y1, x2, y2]
            t_text = reg.get('text', '')
            if t_box and t_text:
                # Assign to best panel
                best_iota = 0.0
                best_panel = None
                for p in panels:
                    iota = compute_iota(p['bbox_xyxy'], t_box)
                    if iota > 0.5 and iota > best_iota:
                        best_iota = iota
                        best_panel = p
                if best_panel:
                    best_panel['text_regions'].append({'bbox': t_box, 'text': t_text})

    # 3. Load VLM Text (Page-level blob)
    vlm_text = ""
    vlm_data = load_json(vlm_path)
    if vlm_data:
        if 'OCRResult' in vlm_data: 
            vlm_data = vlm_data['OCRResult']
        cleaned = filter_boxes(vlm_data)
        vlm_text = json.dumps(cleaned, separators=(',', ':'))

    # 4. Final Formatting
    final_panels = []
    # Sort panels by Y, then X (Reading order)
    panels.sort(key=lambda p: (p['bbox_xyxy'][1], p['bbox_xyxy'][0]))
    
    for p in panels:
        # Sort text within panel by Y then X
        p['text_regions'].sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
        panel_text = " ".join([t['text'] for t in p['text_regions']])
        
        x1, y1, x2, y2 = p['bbox_xyxy']
        final_panels.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1], # Convert to [x, y, w, h]
            'text': panel_text,
            'confidence': p['confidence']
        })

    output_data = {
        "image_width": img_w,
        "image_height": img_h,
        "panels": final_panels,
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
    parser.add_argument('--ocr-root', required=True, help='Root of PaddleOCR JSONs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--zarr-path', help='Optional: Path to Zarr for filtering (Story only)')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    
    # Optional Zarr Filtering
    if args.zarr_path:
        import xarray as xr
        print(f"Filtering by Zarr predictions: {args.zarr_path}")
        ds = xr.open_zarr(args.zarr_path)
        # Prediction 2 is 'story' in our list
        story_mask = ds['prediction'].values == 2
        story_ids = set(ds['ids'].values[story_mask])
        
        initial_len = len(df)
        df = df[df['canonical_id'].isin(story_ids)]
        print(f"Filtered to story pages: {len(df)} / {initial_len}")

    if args.limit:
        print(f"Limiting to first {args.limit} pages.")
        df = df.head(args.limit)
    
    success = 0
    missing_cnn = 0
    missing_ocr = 0
    
    print("Aligning Panels and Text...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cid = row['canonical_id']
        cid_path = cid.replace('/', os.sep)
        
        # VLM resolution
        vlm_path = None
        vlm_candidates = [
            os.path.join(args.vlm_root, f"{cid_path}.json"),
            os.path.join(args.vlm_root, "CalibreComics_extracted", f"{cid_path}.json")
        ]
        for p in vlm_candidates:
            if os.path.exists(p):
                vlm_path = p
                break
        
        # OCR resolution (PaddleOCR)
        ocr_path = None
        ocr_candidates = [
            os.path.join(args.ocr_root, f"{cid_path}_ocr.json"),
            os.path.join(args.ocr_root, "CalibreComics_extracted", f"{cid_path}_ocr.json")
        ]
        for p in ocr_candidates:
            if os.path.exists(p):
                ocr_path = p
                break
        
        # CNN Resolution: Check paths directly (No scanning)
        cnn_path = None
        # 1. Try exact path (detections/Calibre/Book/Page.json)
        p1 = os.path.join(args.detections_root, f"{cid_path}.json")
        
        # 2. Try removing top folder (detections/Book/Page.json)
        parts = cid_path.split(os.sep)
        p2 = None
        if len(parts) > 1:
            p2 = os.path.join(args.detections_root, os.sep.join(parts[1:]) + ".json")
            
        if os.path.exists(p1):
            cnn_path = p1
        elif p2 and os.path.exists(p2):
            cnn_path = p2
        
        if args.debug and idx < 5:
            print(f"\n[DEBUG] CID: {cid}")
            print(f"  Check P1: {p1} -> {os.path.exists(p1)}")
            print(f"  Check P2: {p2} -> {os.path.exists(p2) if p2 else 'Skip'}")
            print(f"  OCR Found: {bool(ocr_path)}")

        if not cnn_path:
            missing_cnn += 1
            continue
            
        if not ocr_path:
            missing_ocr += 1
            
        output_path = os.path.join(args.output_dir, f"{cid_path}.json")
        if process_page(cnn_path, vlm_path, ocr_path, output_path) == "Success":
            success += 1
            
    print(f"\nProcessed: {len(df)} | Success: {success}")
    print(f"Missing CNN: {missing_cnn}")
    print(f"Missing OCR: {missing_ocr}")

if __name__ == "__main__":
    main()