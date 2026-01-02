#!/usr/bin/env python3
"""
Visualize VLM Folder (Optimized Scan-Driven)
Scans the JSON folder first, then looks up images in the manifest.
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
            # Store full ID
            lookup[cid] = img_path
            # Also store by filename stem for robust matching if JSONs are flat
            # e.g. "Page001" -> path
            stem = Path(cid).name # file.jpg
            lookup[stem] = img_path
            # And without extension
            stem_no_ext = Path(cid).stem
            lookup[stem_no_ext] = img_path
            
    return lookup

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

def process_file(json_file, manifest_lookup, output_dir):
    # Determine Canonical ID from JSON filename
    # E:\...\Folder\File.json -> Folder/File ?
    # Or just File.json -> File
    
    stem = json_file.stem # "Folder_File" or "File"
    
    # Try direct lookup
    img_path = manifest_lookup.get(stem)
    
    # Try parsing canonical_id from JSON content (Safest!)
    if not img_path:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cid = data.get('canonical_id')
                if cid:
                    img_path = manifest_lookup.get(cid)
                    if not img_path:
                        # Try fuzzy match on CID stem
                        img_path = manifest_lookup.get(Path(cid).name)
        except:
            pass

    if not img_path:
        return

    # Create output filename
    out_name = f"viz_{json_file.name.replace('.json', '.jpg')}"
    out_path = Path(output_dir) / out_name
    
    if out_path.exists():
        return

    if os.path.exists(img_path):
        draw_boxes(img_path, json_file, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    lookup = load_manifest_lookup(args.manifest)
    
    print(f"Scanning JSONs in {args.input_dir}...")
    json_files = list(Path(args.input_dir).rglob("*.json"))
    print(f"Found {len(json_files)} JSONs to process.")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, f, lookup, args.output_dir) for f in json_files]
        for _ in tqdm(as_completed(futures), total=len(json_files)):
            pass

if __name__ == "__main__":
    main()
