#!/usr/bin/env python3
"""
Visualize Zhipu/GLM VLM Folder (Scan-Driven)
Generates bounding box visualizations for Zhipu/GLM-4V JSONs.
Defaults to [xmin, ymin, xmax, ymax] coordinate order.
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
            # Map canonical_id to image path
            lookup[cid] = img_path
            # Map clean filename to image path
            lookup[Path(cid).name] = img_path
            # Map stem (no ext) to image path
            lookup[Path(cid).stem] = img_path
            # Map with replaced slashes
            lookup[cid.replace('/', '_')] = img_path
    return lookup

def draw_vlm_boxes(image_path, json_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        objects = []
        
        # 1. Extract from 'panels' (Zhipu Analysis style)
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    objects.append({
                        'label': f"panel {p.get('panel_number', '?')}",
                        'box': box
                    })
        
        # 2. Extract from 'objects' (Zhipu Grounding style)
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
            full_label = str(obj['label'])
            base_label = full_label.split(' ')[0].lower().strip()
            if '|' in base_label: base_label = base_label.split('|')[0]
            
            color = colors.get(base_label, 'yellow')
            
            coords = obj['box']
            if len(coords) != 4: continue
            
            # Zhipu/GLM uses [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = coords
            
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=3)
            
            txt_x = max(0, min(abs_xmin + 5, width - 100))
            txt_y = max(0, min(abs_ymin + 5, height - 20))
            
            # Draw label background for readability
            bbox = draw.textbbox((txt_x, txt_y), full_label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((txt_x, txt_y), full_label, fill='white', font=font)

        img.save(output_path)
        
    except Exception as e:
        # print(f"Error drawing {json_path}: {e}")
        pass

def process_file(json_file, manifest_lookup, output_dir):
    stem = json_file.stem
    
    # Try multiple ways to find the image
    img_path = manifest_lookup.get(stem)
    
    if not img_path:
        # Try inside JSON
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

    # Use flat output filename to avoid nested folders
    # e.g. "Guardian_001_p003.jpg"
    flat_name = f"viz_{json_file.name.replace('.json', '.jpg')}"
    out_path = Path(output_dir) / flat_name
    
    if out_path.exists():
        return

    draw_vlm_boxes(img_path, json_file, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Manifest
    lookup = load_manifest_lookup(args.manifest)
    print(f"Manifest loaded with {len(lookup)} entries.")
    
    # 2. Find JSONs
    print(f"Scanning JSONs in {args.input_dir}...")
    json_files = list(Path(args.input_dir).rglob("*.json"))
    
    if args.limit:
        json_files = json_files[:args.limit]
        
    print(f"Processing {len(json_files)} files...")
    
    # 3. Process
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, f, lookup, args.output_dir) for f in json_files]
        for _ in tqdm(as_completed(futures), total=len(json_files)):
            pass
            
    print(f"Done. Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
