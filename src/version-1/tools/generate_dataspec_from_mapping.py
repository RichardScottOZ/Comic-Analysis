#!/usr/bin/env python3
"""
Generate DataSpec JSONs for training using canonical mapping.

MUCH SIMPLER than v2 - just uses the pre-built canonical mapping CSV
to emit DataSpec JSONs with guaranteed correct image paths.

CLAUDE FIXES - 2025-10-28:
1. Line 251: Changed from panel_ratio == 1.0 (floating point comparison)
   to rcnn_panels == vlm_panel_count (integer comparison)
   
2. Line 210-212: CRITICAL - Added category_id == 1 filter!
   COCO has multiple categories (1=panel, 2=character, 3=balloon, 7=face)
   We were counting ALL detections instead of just PANELS
   This caused only ~3,600 "perfect matches" instead of expected ~80,000
   
   Example:
   - Page has 6 panels (VLM)
   - COCO has 50 annotations (6 panels + 20 chars + 15 balloons + 9 faces)
   - BEFORE: Compared 50 != 6 → NO MATCH
   - AFTER: Compared 6 == 6 → PERFECT MATCH!
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

def load_coco_data(coco_file):
    """Load COCO predictions to get detection boxes."""
    print(f"Loading COCO data from: {coco_file}")
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Build image_id -> detections mapping
    image_detections = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id:
            if img_id not in image_detections:
                image_detections[img_id] = []
            image_detections[img_id].append(ann)
    
    print(f"Loaded {len(image_detections):,} images with detections")
    return image_detections, coco_data

def load_vlm_json(vlm_path):
    """Load a VLM JSON file."""
    try:
        with open(vlm_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load VLM JSON {vlm_path}: {e}")
        return None

def extract_vlm_panels(vlm_data):
    """Extract panels list from VLM JSON (handles various formats)."""
    if not vlm_data:
        return []
    
    if isinstance(vlm_data, dict):
        if 'panels' in vlm_data and isinstance(vlm_data['panels'], list):
            return vlm_data['panels']
        for key in ['result', 'page', 'data']:
            if key in vlm_data and isinstance(vlm_data[key], dict):
                if 'panels' in vlm_data[key]:
                    return vlm_data[key]['panels']
    elif isinstance(vlm_data, list) and vlm_data:
        if isinstance(vlm_data[0], dict) and 'panels' in vlm_data[0]:
            return vlm_data[0]['panels']
    
    return []

def aggregate_panel_text(panel):
    """Extract all text from a VLM panel into dialogue/narration/sfx categories."""
    text_dict = {'dialogue': [], 'narration': [], 'sfx': []}
    
    if not isinstance(panel, dict):
        return text_dict
    
    # Helper to add text to a category
    def add_text(val, category='dialogue'):
        if isinstance(val, str) and val.strip():
            text_dict[category].append(val.strip())
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, str) and v.strip():
                    text_dict[category].append(v.strip())
    
    # Extract from 'text' field
    text_field = panel.get('text')
    if isinstance(text_field, dict):
        add_text(text_field.get('dialogue'), 'dialogue')
        add_text(text_field.get('narration'), 'narration')
        add_text(text_field.get('caption'), 'narration')
        add_text(text_field.get('sfx'), 'sfx')
    elif isinstance(text_field, str):
        add_text(text_field, 'dialogue')
    
    # Extract from 'speakers' field
    speakers = panel.get('speakers')
    if isinstance(speakers, list):
        for speaker in speakers:
            if isinstance(speaker, dict):
                speech_type = str(speaker.get('speech_type', '')).lower()
                dialogue = speaker.get('dialogue') or speaker.get('text')
                if 'narration' in speech_type or 'caption' in speech_type:
                    add_text(dialogue, 'narration')
                elif 'sfx' in speech_type or 'sound' in speech_type:
                    add_text(dialogue, 'sfx')
                else:
                    add_text(dialogue, 'dialogue')
    
    # Other fields
    add_text(panel.get('caption'), 'narration')
    add_text(panel.get('description'), 'narration')
    
    return text_dict

def main():
    parser = argparse.ArgumentParser(description='Generate DataSpec JSONs from canonical mapping')
    parser.add_argument('--mapping_csv', required=True, 
                       help='Canonical mapping CSV (e.g., key_mapping_report_calibre.csv)')
    parser.add_argument('--coco_file', required=True,
                       help='COCO predictions JSON')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for DataSpec JSONs')
    parser.add_argument('--subset', choices=['perfect', 'perfect_with_text', 'near_perfect', 'high_quality', 'medium_quality', 'low_quality', 'all'], default='perfect',
                       help='Which subset to generate (default: perfect)')
    parser.add_argument('--min_text_coverage', type=float, default=0.8,
                       help='Minimum text coverage for perfect_with_text (default: 0.8)')
    parser.add_argument('--min_score', type=float, default=0.5,
                       help='Minimum detection score (default: 0.5)')
    parser.add_argument('--require_equal_counts', action='store_true',
                       help='Only emit when RCNN panel count == VLM panel count')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of DataSpec JSONs to generate (for testing)')
    parser.add_argument('--list_output', default=None,
                       help='Output .txt file with list of generated DataSpec JSON paths (for training)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATASPEC GENERATOR FROM CANONICAL MAPPING")
    print("=" * 80)
    
    # Load canonical mapping
    print(f"\nLoading canonical mapping from: {args.mapping_csv}")
    df = pd.read_csv(args.mapping_csv)
    print(f"Loaded {len(df):,} records")
    
    # Filter to valid records only
    valid = df[
        (df['image_exists'] == True) & 
        (df['coco_id_exists'] == True) & 
        (df['vlm_json_exists'] == True)
    ].copy()
    print(f"Valid records (all three exist): {len(valid):,}")
    
    if len(valid) == 0:
        print("ERROR: No valid records found! Check your mapping CSV.")
        return
    
    # Apply subset filter (based on panel count ratio - we'll calculate it)
    print(f"\nApplying subset filter: {args.subset}")
    
    # Load COCO data
    image_detections, coco_data = load_coco_data(args.coco_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate DataSpec JSONs
    print(f"\nGenerating DataSpec JSONs...")
    
    records_to_process = valid.head(args.limit) if args.limit else valid
    
    generated = 0
    skipped = 0
    errors = 0
    generated_paths = []  # Track all generated paths for .txt output
    
    for idx, row in tqdm(records_to_process.iterrows(), total=len(records_to_process), desc="Generating DataSpec"):
        try:
            # Get the ACTUAL image path (guaranteed to exist)
            image_path = row['image_path']
            
            # Get predictions.json ID
            predictions_id = row['predictions_json_id']
            
            # Get VLM JSON path
            vlm_json_path = row['vlm_json_path']
            
            # Load VLM data
            vlm_data = load_vlm_json(vlm_json_path)
            if not vlm_data:
                skipped += 1
                continue
            
            # Extract VLM panels
            vlm_panels = extract_vlm_panels(vlm_data)
            if not vlm_panels:
                skipped += 1
                continue
            
            # Get RCNN detections
            detections = image_detections.get(predictions_id, [])
            if not detections:
                skipped += 1
                continue
            
            # CLAUDE FIX: Filter by category_id == 1 (panel) AND score
            # CRITICAL: COCO has multiple categories (panel, character, balloon, face)
            # We only want to count PANEL detections for matching with VLM panel count!
            boxes = []
            for det in detections:
                # Only count category_id == 1 (panel)
                if det.get('category_id') != 1:
                    continue
                    
                score = det.get('score', 1.0)
                if score >= args.min_score:
                    bbox = det.get('bbox')
                    if bbox and len(bbox) == 4:
                        boxes.append(bbox)
            
            # Sort boxes top-to-bottom, left-to-right
            boxes.sort(key=lambda b: (b[1], b[0]))
            
            if not boxes:
                skipped += 1
                continue
            
            # Check count matching if required
            if args.require_equal_counts and len(boxes) != len(vlm_panels):
                skipped += 1
                continue
            
            # Calculate quality metrics
            rcnn_panels = len(boxes)
            vlm_panel_count = len(vlm_panels)
            panel_ratio = rcnn_panels / vlm_panel_count if vlm_panel_count > 0 else 0.0
            
            # Calculate text coverage
            vlm_panels_with_text = 0
            for panel in vlm_panels:
                text = aggregate_panel_text(panel)
                if any(text.get(k) for k in ['dialogue', 'narration', 'sfx']):
                    vlm_panels_with_text += 1
            
            text_coverage = vlm_panels_with_text / vlm_panel_count if vlm_panel_count > 0 else 0.0
            
            # Calculate quality score (matching v2 logic)
            quality_score = 0
            if 0.9 <= panel_ratio <= 1.1:
                quality_score += 2
            elif 0.8 <= panel_ratio <= 1.2:
                quality_score += 1
            
            if text_coverage >= args.min_text_coverage:
                quality_score += 1
            elif text_coverage >= max(0.0, args.min_text_coverage - 0.2):
                quality_score += 0.5
            
            if 1 <= vlm_panel_count <= 15:
                quality_score += 1
            elif 1 <= vlm_panel_count <= 20:
                quality_score += 0.5
            
            # CLAUDE FIX: Use integer comparison, not floating point!
            # Perfect match = exact same count (not ratio == 1.0 which has FP precision issues)
            is_perfect_match = (rcnn_panels == vlm_panel_count)
            
            # Apply subset filter
            if args.subset == 'perfect' and not is_perfect_match:
                skipped += 1
                continue
            elif args.subset == 'perfect_with_text' and not (is_perfect_match and text_coverage >= args.min_text_coverage):
                skipped += 1
                continue
            elif args.subset == 'near_perfect' and quality_score < 3.5:
                skipped += 1
                continue
            elif args.subset == 'high_quality' and quality_score < 2.5:
                skipped += 1
                continue
            elif args.subset == 'medium_quality' and (quality_score < 1.5 or quality_score >= 2.5):
                skipped += 1
                continue
            elif args.subset == 'low_quality' and quality_score >= 1.5:
                skipped += 1
                continue
            # 'all' accepts everything
            
            # Build DataSpec panels
            K = min(len(boxes), len(vlm_panels))
            panels_out = []
            
            for i in range(K):
                x, y, w, h = boxes[i]
                text = aggregate_panel_text(vlm_panels[i]) if i < len(vlm_panels) else {'dialogue': [], 'narration': [], 'sfx': []}
                
                panels_out.append({
                    'panel_coords': [int(x), int(y), max(1, int(w)), max(1, int(h))],
                    'text': text
                })
            
            # Create DataSpec object with CORRECT image_path
            dataspec = {
                'page_image_path': image_path,  # THE ACTUAL CORRECT PATH!
                'panels': panels_out
            }
            
            # Generate output path (preserve directory structure)
            rel_path = row['image_rel_path']
            # Replace extension with .json
            json_rel_path = os.path.splitext(rel_path)[0] + '.json'
            output_path = os.path.join(args.output_dir, json_rel_path)
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write DataSpec JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataspec, f, indent=2, ensure_ascii=False)
            
            generated += 1
            generated_paths.append(output_path)
            
        except Exception as e:
            print(f"\nError processing {row.get('image_rel_path', 'unknown')}: {e}")
            errors += 1
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Generated: {generated:,} DataSpec JSONs")
    print(f"Skipped: {skipped:,} (didn't meet criteria)")
    print(f"Errors: {errors:,}")
    print(f"\nOutput directory: {args.output_dir}")
    
    # Write list file if requested
    if args.list_output:
        list_path = args.list_output
    else:
        # Auto-generate list filename based on output_dir and subset
        list_path = f"{args.output_dir}_{args.subset}_list.txt"
    
    print(f"\nWriting list of paths to: {list_path}")
    with open(list_path, 'w', encoding='utf-8') as f:
        for path in generated_paths:
            f.write(path + '\n')
    print(f"Wrote {len(generated_paths):,} paths to list file")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
