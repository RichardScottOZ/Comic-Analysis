#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align VLM panel text and grounding (box_2d) with precise RCNN bounding boxes (bbox)
across the dataset using spatial overlap (IoU) and greedy matching.

Supports incremental saving and resume capability to survive interruptions.
Uses ThreadPoolExecutor for concurrent disk I/O.
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV = "documentation/plots/vlm_vs_rcnn_comparison.csv"
VLM_CACHE = "E:/vlm_cache"
RCNN_ROOT = "E:/Comic_Analysis_Results_v2/stage3_json"
DEFAULT_OUT = "stage3_aligned_metadata.json"

# ── Geometry Helpers ──────────────────────────────────────────────────────────

def box_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    
    iw = max(0, ix_max - ix_min)
    ih = max(0, iy_max - iy_min)
    inter = iw * ih
    
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union > 0 else 0.0


def align_panels_for_page(row_dict):
    vlm_cid = row_dict['vlm_canonical_id']
    rcnn_cid = row_dict['rcnn_canonical_id']
    cluster_id = int(row_dict['cluster_id'])
    seq_idx = int(row_dict['page_index'])
    
    vlm_path = os.path.join(VLM_CACHE, vlm_cid.replace('/', os.sep) + '.json')
    rcnn_path = os.path.join(RCNN_ROOT, rcnn_cid.replace('/', os.sep) + '.json') if rcnn_cid != 'NOT_FOUND' else None
    
    # 1. Load RCNN data & image size (RCNN has accurate dimensions)
    rcnn_panels_raw = []
    width, height = 100, 100
    if rcnn_path and os.path.exists(rcnn_path):
        try:
            with open(rcnn_path, 'r', encoding='utf-8') as f:
                rcnn_data = json.load(f)
            rcnn_panels_raw = rcnn_data.get('panels') or []
            width = rcnn_data.get('image_width', 100)
            height = rcnn_data.get('image_height', 100)
        except Exception:
            pass
            
    # 2. Load VLM data
    vlm_panels_raw = []
    overall_summary = ""
    if os.path.exists(vlm_path):
        try:
            with open(vlm_path, 'r', encoding='utf-8') as f:
                vlm_data = json.load(f)
            vlm_panels_raw = vlm_data.get('panels') or []
            overall_summary = vlm_data.get('overall_summary') or vlm_data.get('summary', {}).get('plot', '')
            if width == 100:
                width = vlm_data.get('image_width', 100)
                height = vlm_data.get('image_height', 100)
        except Exception:
            pass
            
    # 3. Extract and scale VLM coordinates
    vlm_panels = []
    for idx, p in enumerate(vlm_panels_raw):
        if not isinstance(p, dict):
            continue
        box2d = p.get('box_2d')
        bbox_px = None
        if isinstance(box2d, (list, tuple)) and len(box2d) == 4 and all(v is not None for v in box2d):
            try:
                y1, x1, y2, x2 = [float(v) for v in box2d]
                px = x1 / 1000.0 * width
                py = y1 / 1000.0 * height
                pw = (x2 - x1) / 1000.0 * width
                ph = (y2 - y1) / 1000.0 * height
                bbox_px = [px, py, pw, ph]
            except (ValueError, TypeError):
                bbox_px = None
            
        text_parts = []
        desc = p.get('description', '').strip()
        if desc:
            text_parts.append(desc)
        for tc in (p.get('text_content') or []):
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
            
    # 5. Spatial alignment (greedy matching)
    aligned_panels = []
    matched_rcnn_indices = set()
    matched_vlm_indices = set()
    
    matches = []
    for v_p in vlm_panels:
        if v_p['vlm_box_px'] is None:
            continue
        for r_p in rcnn_panels:
            iou = box_iou(v_p['vlm_box_px'], r_p['bbox'])
            if iou > 0.1:
                matches.append((iou, v_p['index'], r_p['index']))
                
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
                'modality_mask': [1.0, 1.0, 1.0],
                'match_iou': iou
            })
            matched_vlm_indices.add(v_idx)
            matched_rcnn_indices.add(r_idx)
            
    for r_p in rcnn_panels:
        if r_p['index'] not in matched_rcnn_indices:
            aligned_panels.append({
                'rcnn_bbox': r_p['bbox'],
                'vlm_box_2d': None,
                'vlm_box_px': None,
                'text': "",
                'modality_mask': [1.0, 0.0, 1.0],
                'match_iou': 0.0
            })
            
    for v_p in vlm_panels:
        if v_p['index'] not in matched_vlm_indices:
            aligned_panels.append({
                'rcnn_bbox': None,
                'vlm_box_2d': v_p['vlm_box_2d'],
                'vlm_box_px': v_p['vlm_box_px'],
                'text': v_p['text'],
                'modality_mask': [0.0, 1.0, 0.0],
                'match_iou': 0.0
            })
            
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
        'sequence_index': seq_idx,
        'canonical_id': vlm_cid,
        'cluster_id': cluster_id,
        'overall_summary': overall_summary,
        'image_width': width,
        'image_height': height,
        'vlm_panels_count': len(vlm_panels),
        'rcnn_panels_count': len(rcnn_panels),
        'num_panels': len(aligned_panels),
        'panels': aligned_panels
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconcile VLM and RCNN panel geometries with Resume Support")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of pages to process")
    parser.add_argument('--workers', type=int, default=4, help="Number of thread pool workers")
    parser.add_argument('--output', type=str, default=DEFAULT_OUT, help="Path to save aligned JSON metadata")
    parser.add_argument('--save_every', type=int, default=5000, help="Save progress incrementally every N pages")
    args = parser.parse_args()
    
    # 1. Load existing file for Resuming
    existing_cids = set()
    aligned_dataset = []
    
    if os.path.exists(args.output):
        print(f"Found existing output file: {args.output}")
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                aligned_dataset = json.load(f)
            # Gather completed canonical IDs
            existing_cids = {item['canonical_id'] for item in aligned_dataset if 'canonical_id' in item}
            print(f"  Resuming: {len(existing_cids):,} pages already processed.")
        except Exception as e:
            print(f"  Error loading {args.output} for resume: {e}. Starting fresh.")
            aligned_dataset = []
            
    # 2. Load page list to align
    print(f"Loading page list from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    if args.limit:
        print(f"Applying limit of first {args.limit} pages.")
        df = df.head(args.limit)
        
    records = df.to_dict('records')
    
    # Filter out already processed pages
    if existing_cids:
        records = [r for r in records if r['vlm_canonical_id'] not in existing_cids]
        
    total = len(records)
    print(f"Remaining pages to align: {total:,}")
    if total == 0:
        print("All pages already processed. Done!")
        return
        
    temp_results = []
    
    print(f"\nStarting ThreadPool alignment using {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Stream tasks lazily to avoid memory leaks/bloat
        results = tqdm(
            executor.map(align_panels_for_page, records),
            total=total,
            desc="Aligning panels"
        )
        
        for idx, res in enumerate(results):
            if res is not None:
                temp_results.append(res)
                
            # Incremental save checkpoint
            if (idx + 1) % args.save_every == 0:
                aligned_dataset.extend(temp_results)
                temp_results = []
                
                # Sort by sequence_index to maintain array order structure
                aligned_dataset.sort(key=lambda x: x['sequence_index'])
                
                # Write to temp file first to prevent corruption during writes
                temp_out = args.output + ".tmp"
                with open(temp_out, 'w', encoding='utf-8') as f:
                    json.dump(aligned_dataset, f, indent=2)
                # Atomic swap
                if os.path.exists(args.output):
                    os.remove(args.output)
                os.rename(temp_out, args.output)
                
                print(f"\n  Checkpoint: Saved {len(aligned_dataset):,} completed pages to disk.")
                sys.stdout.flush()
                
    # Final flush
    if temp_results:
        aligned_dataset.extend(temp_results)
        aligned_dataset.sort(key=lambda x: x['sequence_index'])
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(aligned_dataset, f, indent=2)
            
    print(f"\nLoop complete! Successfully aligned {len(aligned_dataset):,} records.")
    print("Dataset generation complete!")

if __name__ == '__main__':
    main()
