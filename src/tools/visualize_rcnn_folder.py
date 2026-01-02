#!/usr/bin/env python3
"""
Visualize R-CNN Folder (Scan-Driven)
Generates bounding box visualizations for Faster R-CNN JSONs.
"""

import os
import argparse
import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_manifest_lookup(manifest_path):
    print(f"Loading manifest: {manifest_path}")
    lookup = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            img_path = row['absolute_image_path']
            lookup[cid] = img_path
            # Store stem lookup
            lookup[Path(cid).name] = img_path
            lookup[Path(cid).stem] = img_path
    return lookup

def draw_rcnn_boxes(img_path, json_path, out_path):
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

        # Scaling logic
        json_size = data.get('image_size_wh', [1, 1])
        json_w, json_h = json_size
        scale_x = width / json_w
        scale_y = height / json_h

        detections = data.get('detections', [])
        
        # Sort by Area Descending (Draw big stuff first, small stuff on top)
        def get_area(det):
            b = det.get('box_xyxy', [0,0,0,0])
            return (b[2] - b[0]) * (b[3] - b[1])
            
        detections.sort(key=get_area, reverse=True)

        colors = {
            'panel': 'blue',
            'text': 'green',
            'character': 'red',
            'face': 'magenta'
        }

        # Draw
        for i, det in enumerate(detections):
            score = det.get('score', 0)
            if score < 0.5: continue
            
            label = det.get('label')
            raw_box = det.get('box_xyxy')
            
            # Apply Scale
            box = [
                raw_box[0] * scale_x,
                raw_box[1] * scale_y,
                raw_box[2] * scale_x,
                raw_box[3] * scale_y
            ]
            
            color = colors.get(label, 'yellow')
            draw.rectangle(box, outline=color, width=4)
            
            label_text = f"{label} {score:.2f}"
            
            # Draw text with stroke
            txt_x = max(0, min(box[0] + 5, width - 80))
            txt_y = max(0, min(box[1] + 5, height - 15))
            
            try:
                draw.text((txt_x, txt_y), label_text, fill='white', font=font, stroke_width=1, stroke_fill='black')
            except:
                draw.text((txt_x, txt_y), label_text, fill=color, font=font)

        img.save(out_path)
        
    except Exception as e:
        # print(f"Error processing {json_path}: {e}")
        pass

def process_file(json_file, manifest_lookup, output_dir):
    stem = json_file.stem 
    img_path = manifest_lookup.get(stem)
    
    if not img_path:
        # Try finding canonical_id in JSON
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cid = data.get('canonical_id')
                if cid:
                    img_path = manifest_lookup.get(cid)
                    if not img_path:
                        img_path = manifest_lookup.get(Path(cid).name)
        except: pass

    if not img_path or not os.path.exists(img_path):
        return

    # Flatten name for viz folder
    flat_name = json_file.name.replace('.json', '.jpg')
    out_path = Path(output_dir) / f"viz_{flat_name}"
    
    if out_path.exists():
        return

    draw_rcnn_boxes(img_path, json_file, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    lookup = load_manifest_lookup(args.manifest)
    
    print(f"Scanning JSONs in {args.input_dir}...")
    json_files = list(Path(args.input_dir).rglob("*.json"))
    
    if args.limit:
        json_files = json_files[:args.limit]
        
    print(f"Processing {len(json_files)} files...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, f, lookup, args.output_dir) for f in json_files]
        for _ in tqdm(as_completed(futures), total=len(json_files)):
            pass

if __name__ == "__main__":
    main()
