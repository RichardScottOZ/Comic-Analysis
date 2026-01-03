#!/usr/bin/env python3
"""
Visualize VLM Folder (Scan-Driven)
Generates bounding box visualizations for VLM JSONs in bulk.
Supports piped labels and custom coordinate orders.
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
            lookup[Path(cid).name] = img_path
            lookup[Path(cid).stem] = img_path
    return lookup

def draw_vlm_boxes(image_path, json_path, output_path, order='ymin_first'):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        objects = []
        
        # 1. Extract from 'panels'
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    objects.append({
                        'label': f"panel|{p.get('panel_number', '?')}",
                        'box': box
                    })
        
        # 2. Extract from 'objects'
        if 'objects' in data:
            for obj in data['objects']:
                box = obj.get('box_2d') or obj.get('box')
                if box:
                    objects.append({
                        'label': obj.get('label', 'obj'),
                        'box': box
                    })

        if not objects:
            return

        colors = {
            'panel': 'blue', 
            'person': 'red', 
            'text': 'green', 
            'face': 'magenta',
            'car': 'cyan',
            'building': 'orange'
        }

        for obj in objects:
            full_label = obj['label'].lower()
            base_label = full_label.split('|')[0].strip()
            color = colors.get(base_label, 'yellow')
            
            coords = obj['box']
            if len(coords) != 4: continue
            
            if order == 'xmin_first':
                xmin, ymin, xmax, ymax = coords
            else:
                ymin, xmin, ymax, xmax = coords
            
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=4)
            
            txt_x = max(0, min(abs_xmin + 5, width - 100))
            txt_y = max(0, min(abs_ymin + 5, height - 20))
            
            try:
                draw.text((txt_x, txt_y), full_label, fill='white', font=font, stroke_width=1, stroke_fill='black')
            except:
                draw.text((txt_x, txt_y), full_label, fill=color, font=font)

        img.save(output_path)
        
    except Exception:
        pass

def process_file(json_file, manifest_lookup, output_dir, order):
    stem = json_file.stem
    img_path = manifest_lookup.get(stem)
    
    if not img_path:
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

    flat_name = f"viz_{json_file.name.replace('.json', '.jpg')}"
    out_path = Path(output_dir) / flat_name
    
    if out_path.exists():
        return

    draw_vlm_boxes(img_path, json_file, out_path, order)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--order', choices=['ymin_first', 'xmin_first'], default='ymin_first')
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
        futures = [executor.submit(process_file, f, lookup, args.output_dir, args.order) for f in json_files]
        for _ in tqdm(as_completed(futures), total=len(json_files)):
            pass

if __name__ == "__main__":
    main()
