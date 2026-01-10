#!/usr/bin/env python3
"""
Unified JSON Generator for Stage 3 (Manifest-Driven)
Combines CNN Panel Detections (Master ID style) with VLM/OCR Text (Calibre ID style).
Bridges IDs using Suffix Matching for 100% deterministic matching across disparate manifests.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import csv

def filter_boxes(data):
    if isinstance(data, dict):
        junk_keys = {'box_2d', 'box', 'polygon', 'detections', 'coordinates', 'bbox'}
        return {k: filter_boxes(v) for k, v in data.items() if k.lower() not in junk_keys}
    elif isinstance(data, list):
        return [filter_boxes(i) for i in data]
    else: return data

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

def compute_iota(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / area2 if area2 > 0 else 0.0

def process_page(cnn_path, vlm_path, ocr_path, output_path):
    cnn_data = load_json(cnn_path)
    if not cnn_data: return "Missing CNN"
    img_w, img_h = cnn_data.get('image_size_wh', [0, 0])
    
    panels = []
    dets = cnn_data.get('detections', []) or cnn_data.get('panels', [])
    for det in dets:
        if det.get('label') == 'panel' and det.get('score', 1.0) > 0.4:
            box = det.get('box_xyxy') or det.get('bbox')
            panels.append({'bbox_xyxy': [float(x) for x in box], 'score': float(det.get('score', 1.0)), 'text_regions': []})
    
    ocr_data = load_json(ocr_path)
    if ocr_data and 'OCRResult' in ocr_data:
        for reg in ocr_data['OCRResult'].get('text_regions', []):
            t_box, t_text = reg.get('bbox'), reg.get('text', '')
            if t_box and t_text:
                best_iota, best_panel = 0.0, None
                for p in panels:
                    iota = compute_iota(p['bbox_xyxy'], t_box)
                    if iota > 0.5 and iota > best_iota: best_iota, best_panel = iota, p
                if best_panel: best_panel['text_regions'].append({'bbox': t_box, 'text': t_text})

    vlm_text = ""
    vlm_data = load_json(vlm_path)
    if vlm_data:
        if 'OCRResult' in vlm_data: vlm_data = vlm_data['OCRResult']
        vlm_text = json.dumps(filter_boxes(vlm_data), separators=(',', ':'))

    final_panels = []
    panels.sort(key=lambda p: (p['bbox_xyxy'][1], p['bbox_xyxy'][0]))
    for p in panels:
        p['text_regions'].sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
        final_panels.append({
            'bbox': [p['bbox_xyxy'][0], p['bbox_xyxy'][1], p['bbox_xyxy'][2]-p['bbox_xyxy'][0], p['bbox_xyxy'][3]-p['bbox_xyxy'][1]],
            'text': " ".join([t['text'] for t in p['text_regions']]) ,
            'confidence': p['score']
        })

    out_data = {"image_width": img_w, "image_height": img_h, "panels": final_panels, "full_page_text": vlm_text}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f: json.dump(out_data, f, indent=2)
    return "Success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Calibre Manifest (IDs)')
    parser.add_argument('--master-manifest', required=True, help='Master Manifest (Local Paths/IDs)')
    parser.add_argument('--detections-root', required=True)
    parser.add_argument('--vlm-root', required=True)
    parser.add_argument('--ocr-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    # 1. Index Master Manifest (Direct ID Index)
    print("Indexing Master Manifest...")
    master_id_set = set()
    with open(args.master_manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            master_id_set.add(row['canonical_id'])
    print(f"Indexed {len(master_id_set)} Master IDs.")

    # 2. Process Calibre Manifest
    df = pd.read_csv(args.manifest)
    if args.limit: df = df.head(args.limit)
    
    success, missing_cnn, skipped = 0, 0, 0
    print("Aligning...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cid = row['canonical_id'] # e.g. Calibre/.../amazon/#Guardian...
        cid_path = cid.replace('/', os.sep)
        output_path = os.path.join(args.output_dir, f"{cid_path}.json")
        
        if os.path.exists(output_path):
            skipped += 1
            continue
            
        # Bridge Detection Path via Suffix Search
        # Try every suffix of the Calibre CID to see if it matches a Master CID
        parts = cid.split('/')
        master_id = None
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            if suffix in master_id_set:
                master_id = suffix
                break
        
        cnn_path = os.path.join(args.detections_root, f"{master_id}.json") if master_id else None
        
        if not cnn_path or not os.path.exists(cnn_path):
            missing_cnn += 1
            continue
            
        vlm_path = os.path.join(args.vlm_root, f"{cid_path}.json")
        ocr_path = os.path.join(args.ocr_root, f"{cid_path}_ocr.json")
        
        if process_page(cnn_path, vlm_path, ocr_path, output_path) == "Success":
            success += 1
            
    print(f"\nProcessed: {len(df)} | Success: {success} | Skipped: {skipped} | Missing CNN: {missing_cnn}")

if __name__ == "__main__":
    main()