#!/usr/bin/env python3
"""
Align Panels and Text for Stage 3
Matches CNN Panel Detections with VLM/OCR Text to create Stage 3 Training Data.

Workflow:
1. Load CNN Panel Boxes (from batch_detections_local.py output)
2. Load VLM/OCR Text Boxes (from VLM staging)
3. Assign text to panels based on spatial overlap
4. Output unified JSONs compatible with Stage 3 Dataset
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if area1 == 0 or area2 == 0: return 0.0
    
    # We use Intersection over Text Area (IoTA) because text is small and fits inside panels
    return intersection / area2 

def align_page(detection_path, vlm_path, output_path):
    # 1. Load Data
    cnn_data = load_json(detection_path)
    vlm_data = load_json(vlm_path)
    
    if not cnn_data:
        return "Missing CNN Data"
    
    # 2. Extract Panels
    panels = []
    if 'detections' in cnn_data:
        for det in cnn_data['detections']:
            if det['label'] == 'panel' and det['score'] > 0.5:
                # box_xyxy is [x1, y1, x2, y2]
                panels.append({
                    'bbox': det['box_xyxy'],
                    'score': det['score'],
                    'text_content': []
                })
    
    # Sort panels by position (Top-Left to Bottom-Right roughly)
    # Sort by Y, then X
    panels.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))
    
    # 3. Extract Text
    text_boxes = []
    if vlm_data:
        # Check structure: might be 'OCRResult' or direct list
        ocr_list = vlm_data.get('OCRResult', []) if isinstance(vlm_data, dict) else []
        
        # Flatten if list of lists (sometimes happens in OCR output)
        if ocr_list and isinstance(ocr_list[0], list):
            ocr_list = [item for sublist in ocr_list for item in sublist]
            
        for item in ocr_list:
            # item format usually: {'box': [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], 'text': "foo", 'score': 0.9}
            # OR {'box_2d': [x1, y1, x2, y2], 'text': ...}
            
            bbox = None
            text = ""
            
            if 'box_2d' in item:
                # [x1, y1, x2, y2] format
                # Ensure it's 1000-scale or pixel scale? 
                # VLM usually 1000-scale. Paddle is pixel scale.
                # We need to normalize if sources differ. 
                # Assuming pixel scale for now based on file names.
                bbox = item['box_2d']
                text = item.get('text_content', '') or item.get('text', '')
            elif 'box' in item:
                # [[x1,y1], [x2,y1]...] polygon format
                poly = np.array(item['box'])
                x1, y1 = poly.min(axis=0)
                x2, y2 = poly.max(axis=0)
                bbox = [x1, y1, x2, y2]
                text = item.get('text', '')
            
            if bbox and text:
                text_boxes.append({'bbox': bbox, 'text': text})

    # 4. Assign Text to Panels
    for t_box in text_boxes:
        best_iou = 0.0
        best_panel = None
        
        for panel in panels:
            iou = compute_iou(panel['bbox'], t_box['bbox'])
            # Intersection over Text Area > 0.5 means mostly inside
            if iou > 0.5 and iou > best_iou:
                best_iou = iou
                best_panel = panel
        
        if best_panel:
            best_panel['text_content'].append(t_box)
    
    # 5. Format Output
    # Convert panel xyxy to xywh for Stage 3 Dataset
    final_panels = []
    for p in panels:
        # Sort text within panel reading order
        p['text_content'].sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
        full_text = " ".join([t['text'] for t in p['text_content']])
        
        x1, y1, x2, y2 = p['bbox']
        w = x2 - x1
        h = y2 - y1
        
        final_panels.append({
            'bbox': [x1, y1, w, h], # xywh format
            'text': full_text,
            'confidence': p['score']
        })
        
    output_data = {
        "image_width": cnn_data.get('image_size_wh', [0, 0])[0],
        "image_height": cnn_data.get('image_size_wh', [0, 0])[1],
        "panels": final_panels
    }
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    return "Success"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--detections-root', required=True, help='Root of CNN JSONs')
    parser.add_argument('--vlm-root', required=True, help='Root of VLM/OCR JSONs')
    parser.add_argument('--output-dir', required=True, help='Where to save Stage 3 JSONs')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--debug', action='store_true', help='Print detailed path debug info')
    args = parser.parse_args()
    
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    
    if args.limit:
        print(f"Limiting to first {args.limit} pages.")
        df = df.head(args.limit)
    
    success_count = 0
    missing_vlm = 0
    missing_cnn = 0
    
    print("Aligning Panels and Text...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cid = row['canonical_id'] # e.g. Book/Page or Book/Sub/Page
        
        # 1. Resolve CNN Path
        # Try full path first
        cid_path = cid.replace('/', os.sep)
        cnn_path = os.path.join(args.detections_root, f"{cid_path}.json")
        
        # Fallback: Strip first folder (e.g. CalibreComics_extracted)
        if not os.path.exists(cnn_path):
            parts = cid.split('/')
            if len(parts) > 1:
                stripped_cid = "/".join(parts[1:])
                stripped_path = stripped_cid.replace('/', os.sep)
                cnn_path_2 = os.path.join(args.detections_root, f"{stripped_path}.json")
                if os.path.exists(cnn_path_2):
                    cnn_path = cnn_path_2
        
        # 2. Resolve VLM Path (Try standard locations)
        vlm_candidates = [
            os.path.join(args.vlm_root, f"{cid_path}.json"),
            os.path.join(args.vlm_root, "CalibreComics_extracted", f"{cid_path}.json"),
            # Some amazon IDs might be under 'amazon' or root
            os.path.join(args.vlm_root, "amazon", f"{cid_path}.json")
        ]
        
        vlm_path = None
        for p in vlm_candidates:
            if os.path.exists(p):
                vlm_path = p
                break
        
        # Output Path
        output_path = os.path.join(args.output_dir, f"{cid_path}.json")
        
        if not os.path.exists(cnn_path):
            if args.debug and missing_cnn < 5:
                print(f"\n[DEBUG] Missing CNN for ID: {cid}")
                print(f"  Constructed Path: {cnn_path}")
                parent = os.path.dirname(cnn_path)
                if os.path.exists(parent):
                    print(f"  Parent Exists: {parent}")
                    print(f"  Parent Contents: {os.listdir(parent)[:5]}")
                else:
                    print(f"  Parent MISSING: {parent}")
            missing_cnn += 1
            continue
            
        res = align_page(cnn_path, vlm_path, output_path)
        if res == "Success":
            success_count += 1
        elif res == "Missing CNN Data": # Should be caught above, but safety check
            missing_cnn += 1
        else:
            if not vlm_path: missing_vlm += 1 # Track missing VLM explicitly if needed
            
    print(f"\n--- Alignment Complete ---")
    print(f"Processed: {len(df)}")
    print(f"Success: {success_count}")
    print(f"Missing CNN: {missing_cnn}")

if __name__ == "__main__":
    main()
