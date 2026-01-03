#!/usr/bin/env python3
"""
Visualize PaddleOCR Folder (Scan-Driven with S3 Download)
Generates visualizations for PaddleOCR JSONs, downloading them from S3 if needed.
"""

import os
import argparse
import json
import csv
import boto3
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
    return lookup

def draw_paddle_boxes(image_path, json_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = data.get('paddleocr_results', [])
        if not results:
            return

        for res in results:
            # PaddleOCR: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            poly = res.get('bbox') 
            text = res.get('text', '')
            conf = res.get('confidence', 0.0)
            
            if not poly: continue
            
            # Draw Polygon
            points = [(p[0], p[1]) for p in poly]
            draw.polygon(points, outline='red', width=2)
            
            # Label
            txt_x, txt_y = points[0]
            label = f"{text[:15]}..." if len(text) > 15 else text
            
            if hasattr(draw, "textbbox"):
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                w, h = right - left, bottom - top
            else:
                w, h = draw.textsize(label, font=font)
                
            draw.rectangle([txt_x, txt_y - h - 4, txt_x + w + 4, txt_y], fill='red')
            draw.text((txt_x + 2, txt_y - h - 2), label, fill='white', font=font)

        img.save(output_path)
        
    except Exception as e:
        # print(f"Error: {e}")
        pass

def download_json(s3, bucket, key, local_path):
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception:
        return False

def process_row(row, input_dir, output_dir, s3_bucket, s3_prefix, s3_client):
    cid = row['canonical_id']
    img_path = row['absolute_image_path']
    
    if not img_path or not os.path.exists(img_path):
        return

    # Expected JSON name: {cid}_ocr.json
    # Note: canonical_id might be a path. We want the filename part for the local cache?
    # Or should we replicate the full structure locally?
    # Let's mirror the structure locally to be safe.
    
    local_rel_path = f"{cid}_ocr.json"
    local_json = Path(input_dir) / local_rel_path
    
    # Check if exists locally
    if not local_json.exists():
        if s3_client:
            # Download from S3
            # S3 Key: prefix / canonical_id + "_ocr.json"
            # Note: canonical_id matches S3 structure?
            # If cid is "CalibreComics_extracted/amazon/..."
            # And prefix is "ocr_results_paddleocr"
            # Key is "ocr_results_paddleocr/CalibreComics_extracted/amazon/..._ocr.json"
            
            s3_key = f"{s3_prefix}/{cid}_ocr.json"
            
            if not download_json(s3_client, s3_bucket, s3_key, local_json):
                # Try fallback without _ocr
                s3_key_alt = f"{s3_prefix}/{cid}.json"
                if not download_json(s3_client, s3_bucket, s3_key_alt, local_json):
                    return # Not found on S3

    if not local_json.exists():
        return

    # Visualization Output
    # Flatten name for viz folder to avoid deep nesting issues
    flat_name = f"viz_{local_json.name.replace('.json', '.jpg')}"
    out_path = Path(output_dir) / flat_name
    
    if out_path.exists():
        return

    draw_paddle_boxes(img_path, local_json, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True, help='Local cache for JSONs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--s3-bucket', default='calibrecomics-extracted')
    parser.add_argument('--s3-prefix', default='ocr_results_paddleocr')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files')
    args = parser.parse_args()
    
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading manifest: {args.manifest}")
    rows = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if args.limit:
        rows = rows[:args.limit]
        
    print(f"Processing {len(rows)} pages...")
    
    s3 = boto3.client('s3')
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_row, row, args.input_dir, args.output_dir, args.s3_bucket, args.s3_prefix, s3) for row in rows]
        for _ in tqdm(as_completed(futures), total=len(rows)):
            pass

if __name__ == "__main__":
    main()