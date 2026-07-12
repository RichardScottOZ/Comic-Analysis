#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align VLM panel text and grounding (box_2d) with precise RCNN bounding boxes (bbox)
across the dataset using spatial overlap (IoU) and greedy/Hungarian matching.

Saves the aligned panel structure to a final stage3 aligned metadata JSON,
retaining both geometries and setting proper modality masks.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV = "documentation/plots/vlm_vs_rcnn_comparison.csv"
VLM_CACHE = "E:/vlm_cache"
RCNN_ROOT = "E:/Comic_Analysis_Results_v2/stage3_json"
DEFAULT_OUT = "stage3_aligned_metadata.json"

# ── Geometry & Matching Helpers ───────────────────────────────────────────────

def box_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes in [x, y, w, h] format.
    """
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    # Intersection
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    
    iw = max(0, ix_max - ix_min)
    ih = max(0, iy_max - iy_min)
    inter = iw * ih
    
    # Union
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union > 0 else 0.0


def align_panels_for_page(row_dict):
    """
    Core function to load, convert coordinates, and align panels for a single page.
    Returns a dict representing the aligned page record.
    """
    vlm_cid = row_dict['vlm_canonical_id']
    rcnn_cid = row_dict['rcnn_canonical_id']
    cluster_id = int(row_dict['cluster_id'])
    
    vlm_path = os.path.join(VLM_CACHE, vlm_cid.replace('/', os.sep) + '.json')
    rcnn_path = os.path.join(RCNN_ROOT, rcnn_cid.replace('/', os.sep) + '.json') if rcnn_cid != 'NOT_FOUND' else None
    
    # 1. Load VLM data
    vlm_panels_raw = []
    overall_summary = ""
    width, height = 100, 100
    
    if os.path.exists(vlm_path):
        try:
            with open(vlm_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            vlm_panels_raw = data.get('panels') or []
            overall_summary = data.get('overall_summary') or data.get('summary', {}).get('plot', '')
            width = data.get('image_width', 100)
            height = data.get('image_height', 100)
        except Exception:
            pass
            
    # 2. Load RCNN data
    rcnn_panels_raw = []
    if rcnn_path and os.path.exists(rcnn_path):
        try:
            with open(rcnn_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rcnn_panels_raw = data.get('panels') or []
        except Exception:
            pass
            
    # 3. Extract and scale VLM coordinates
    vlm_panels = []
    for idx, p in enumerate(vlm_panels_raw):
        if not isinstance(p, dict):
            continue
        box2d = p.get('box_2d')
        bbox_px = None
        if isinstance(box2d, (list, tuple)) and len(box2d) == 4:
            y1, x1, y2, x2 = [float(v) for v in box2d]
            # Convert normalized 0-1000 to pixel [x, y, w, h]
            px = x1 / 1000.0 * width
            py = y1 / 1000.0 * height
            pw = (x2 - x1) / 1000.0 * width
            ph = (y2 - y1) / 1000.0 * height
            bbox_px = [px, py, pw, ph]
            
        # Parse dialogue
        text_parts = []
        desc = p.get('description', '').strip()
        if desc:
            text_parts.append(desc)
        for tc in p.get('text_content', []):
            if isinstance(tc, dict) and tc.get('text'):
                t = str(tc['text']).strip()
                if t:
                    text_parts.append(t)
        text = " ".join(text_parts)
        
        vlm_panels.append({
            'index': idx,
            'vlm_box_2d': box2d,
            'vlm_box_px': bbox_px,
            'text': text
        })
        
    # 4. Extract RCNN coordinates
    rcnn_panels = []
    for idx, p in enumerate(rcnn_panels_raw):
        if not isinstance(p, dict):
            continue
        bbox = p.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            rcnn_panels.append({
                'index': idx,
                'bbox': [float(v) for v in bbox]
            })
            
    # 5. Spatial alignment (Hungarian-like greedy matching)
    aligned_panels = []
    matched_rcnn_indices = set()
    matched_vlm_indices = set()
    
    # Compute similarity cost matrix (IoU)
    matches = []
    for v_p in vlm_panels:
        if v_p['vlm_box_px'] is None:
            continue
        for r_p in rcnn_panels:
            iou = box_iou(v_p['vlm_box_px'], r_p['bbox'])
            if iou > 0.1:  # threshold to be considered a spatial candidate
                matches.append((iou, v_p['index'], r_p['index']))
                
    # Sort matches by highest overlap first
    matches.sort(key=lambda x: -x[0])
    
    for iou, v_idx, r_idx in matches:
        if v_idx not in matched_vlm_indices and r_idx not in matched_rcnn_indices:
            v_p = next(x for x in vlm_panels if x['index'] == v_idx)
            r_p = next(x for x in rcnn_panels if x['index'] == r_idx)
            
            aligned_panels.append({
                'rcnn_bbox': r_p['bbox'],
                'vlm_box_2d': v_p['vlm_box_2d'],
                'vlm_box_px': v_p['vlm_box_px'],
                'text': v_p['text'],
                'modality_mask': [1.0, 1.0, 1.0],  # Fully fused
                'match_iou': iou
            })
            matched_vlm_indices.add(v_idx)
            matched_rcnn_indices.add(r_idx)
            
    # Add unmatched RCNN boxes (vision only)
    for r_p in rcnn_panels:
        if r_p['index'] not in matched_rcnn_indices:
            aligned_panels.append({
                'rcnn_bbox': r_p['bbox'],
                'vlm_box_2d': None,
                'vlm_box_px': None,
                'text': "",
                'modality_mask': [1.0, 0.0, 1.0],  # Vision + Layout, no Text
                'match_iou': 0.0
            })
            
    # Add unmatched VLM panels (text only)
    for v_p in vlm_panels:
        if v_p['index'] not in matched_vlm_indices:
            aligned_panels.append({
                'rcnn_bbox': None,
                'vlm_box_2d': v_p['vlm_box_2d'],
                'vlm_box_px': v_p['vlm_box_px'],
                'text': v_p['text'],
                'modality_mask': [0.0, 1.0, 0.0],  # Text only, no Vision/Layout
                'match_iou': 0.0
            })
            
    # If the page had panels but completely failed to match or extract anything,
    # supply a dummy single-panel fallback to maintain dataloader sanity
    if not aligned_panels:
        aligned_panels.append({
            'rcnn_bbox': [0.0, 0.0, float(width), float(height)],
            'vlm_box_2d': [0.0, 0.0, 1000.0, 1000.0],
            'vlm_box_px': [0.0, 0.0, float(width), float(height)],
            'text': overall_summary or "",
            'modality_mask': [1.0, 1.0, 1.0] if overall_summary else [1.0, 0.0, 1.0],
            'match_iou': 1.0
        })
        
    return {
        'canonical_id': vlm_cid,
        'cluster_id': cluster_id,
        'overall_summary': overall_summary,
        'image_width': width,
        'image_height': height,
        'panels': aligned_panels
    }


def worker_task(batch_rows):
    """Worker process helper to handle a batch of rows."""
    results = []
    for r in batch_rows:
        try:
            results.append(align_panels_for_page(r))
        except Exception:
            # Safely catch any unhandled page failures
            pass
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconcile and align VLM and RCNN panel geometries")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of pages to process (for dry-runs)")
    parser.add_argument('--workers', type=int, default=8, help="Number of process workers")
    parser.add_argument('--output', type=str, default=DEFAULT_OUT, help="Path to save aligned JSON metadata")
    args = parser.parse_args()
    
    print(f"Loading page list from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    if args.limit:
        print(f"Applying dry-run limit of first {args.limit} pages.")
        df = df.head(args.limit)
        
    records = df.to_dict('records')
    total = len(records)
    print(f"Total pages to align: {total:,}")
    
    # Chunk dataset into batches for process pooling
    chunk_size = max(1, min(100, total // (args.workers * 4)))
    batches = [records[i:i + chunk_size] for i in range(0, total, chunk_size)]
    print(f"Chunk size: {chunk_size} (Total batches: {len(batches)})")
    
    aligned_dataset = []
    
    print(f"\nStarting parallel alignment using {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker_task, b) for b in batches]
        
        with tqdm(total=total, desc="Aligning page geometries") as pbar:
            for fut in as_completed(futures):
                res_list = fut.result()
                aligned_dataset.extend(res_list)
                pbar.update(len(res_list))
                
    print(f"\nAlignment complete! Successfully aligned {len(aligned_dataset):,} / {total:,} records.")
    
    print(f"Saving aligned metadata to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(aligned_dataset, f, indent=2)
        
    print("Dataset generation complete!")

if __name__ == '__main__':
    main()
