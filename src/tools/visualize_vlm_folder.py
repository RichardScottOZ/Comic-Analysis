#!/usr/bin/env python3
"""
Visualize VLM Folder (Manifest Driven)
Generates bounding box visualizations for VLM JSONs corresponding to a manifest.
Guarantees correct image-to-json pairing.
"""

import os
import argparse
import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def draw_boxes(img_path, json_path, out_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        objects = []
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box and len(box)==4:
                    objects.append({'label': f"P{p.get('panel_number')}", 'box': box, 'color': 'blue'})
        
        if 'objects' in data:
            for obj in data['objects']:
                box = obj.get('box_2d') or obj.get('box')
                if box and len(box)==4:
                    label = obj.get('label', 'obj')
                    color = 'green' if 'text' in label else 'red' if 'person' in label else 'blue'
                    objects.append({'label': label, 'box': box, 'color': color})

        if not objects:
            return

        for obj in objects:
            ymin, xmin, ymax, xmax = obj['box']
            color = obj['color']
            
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=4)
            draw.text((abs_xmin+5, abs_ymin+5), obj['label'], fill=color, font=font)
            
        img.save(out_path)
        
    except Exception as e:
        # print(f"Error processing {json_path}: {e}")
        pass

def process_record(record, input_dir, output_dir):
    canonical_id = record['canonical_id']
    image_path = record['absolute_image_path']
    
    # Resolve JSON path (try flat or nested)
    # 1. Flat: output_dir/Canonical_ID.json
    # 2. Nested: output_dir/Folder/File.json
    
    # Assume flat first (as per batch script behavior on Windows often)
    json_path = Path(input_dir) / f"{canonical_id}.json"
    
    if not json_path.exists():
        # Try finding it if canonical_id has slashes
        if '/' in canonical_id or '\\' in canonical_id:
             # Just strict append
             pass 
        else:
             return

    if not json_path.exists():
        return

    # Check image exists
    if not os.path.exists(image_path):
        return

    # Create output filename
    # Flatten the ID for the filename so all viz are in one folder
    flat_name = canonical_id.replace('/', '_').replace('\\', '_')
    out_path = Path(output_dir) / f"viz_{{flat_name}}.jpg"
    
    if out_path.exists():
        return

    draw_boxes(image_path, json_path, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--input-dir', required=True, help='Directory containing VLM JSONs')
    parser.add_argument('--output-dir', required=True, help='Directory to save visualizations')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading manifest: {args.manifest}")
    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    if args.limit:
        records = records[:args.limit]
        
    print(f"Processing {len(records)} records...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_record, r, args.input_dir, args.output_dir) for r in records]
        for _ in tqdm(as_completed(futures), total=len(records)):
            pass

if __name__ == "__main__":
    main()