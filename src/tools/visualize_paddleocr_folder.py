#!/usr/bin/env python3
"""
Visualize PaddleOCR Folder (Strict Manifest Mode)
Matches the logic of batch_ocr_paddleocr_lithops_batched.py
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

def draw_paddle_boxes(image_path, json_path, output_path):
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
            
        # Support both 'paddleocr_results' and 'OCRResult -> text_regions'
        results = data.get('paddleocr_results')
        if not results:
            results = data.get('OCRResult', {}).get('text_regions', [])
            
        if not results:
            print(f"[NO TEXT] {Path(json_path).name}")
            return

        for res in results:
            text = res.get('text', '')
            
            # 1. Try Polygon (Best for rotated text)
            poly = res.get('polygon')
            if poly:
                points = [(p[0], p[1]) for p in poly]
                draw.polygon(points, outline='red', width=3)
                txt_x, txt_y = points[0]
            else:
                # 2. Try BBox [xmin, ymin, xmax, ymax]
                bbox = res.get('bbox')
                if bbox and len(bbox) == 4:
                    xmin, ymin, xmax, ymax = bbox
                    draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
                    txt_x, txt_y = xmin, ymin
                else:
                    continue
            
            label = text
            
            # Background box for text
            if hasattr(draw, "textbbox"):
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                w, h = right - left, bottom - top
            else:
                w, h = draw.textsize(label, font=font)
                
            draw.rectangle([txt_x, txt_y - h - 4, txt_x + w + 4, txt_y], fill='red')
            draw.text((txt_x + 2, txt_y - h - 2), label, fill='white', font=font)

        img.save(output_path)
        print(f"✅ Saved {output_path.name}")
        
    except Exception as e:
        print(f"[DRAW ERROR] {e}")

def load_local_lookup(local_manifest_path):
    print(f"Loading local manifest: {local_manifest_path}")
    lookup = {}
    with open(local_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize ID (slashes) just in case
            cid = row['canonical_id'].replace('\\', '/')
            lookup[cid] = row['absolute_image_path']
    print(f"Loaded {len(lookup)} entries into local lookup.")
    return lookup

def process_row(row, local_lookup, input_dir, output_dir, s3_bucket, s3_prefix, s3_client):
    cid = row['canonical_id'] # From S3 manifest
    
    # 1. Find Local Image using Lookup
    # Try exact match
    local_img_path = local_lookup.get(cid)
    
    if not local_img_path:
        # Try stripping first folder if it's "CalibreComics_extracted"
        if '/' in cid:
            short_cid = cid.split('/', 1)[1]
            local_img_path = local_lookup.get(short_cid)
            if not local_img_path:
                 # Try matching just the filename?
                 # This is risky but useful for debug
                 # stem = Path(cid).stem
                 pass

    if not local_img_path:
        print(f"[NO IMG MATCH] {cid}")
        return

    if not os.path.exists(local_img_path):
        print(f"[IMG NOT FOUND] {local_img_path}")
        return

    # 2. Resolve/Download JSON
    local_json_path = Path(input_dir) / f"{Path(cid).name}_ocr.json"
    
    if not local_json_path.exists():
        s3_key = f"{s3_prefix}/{cid}_ocr.json"
        try:
            local_json_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(s3_bucket, s3_key, str(local_json_path))
        except Exception as e:
            print(f"[S3 FAIL] {s3_key} -> {e}")
            return

    # 3. Visualize
    flat_name = f"viz_{Path(cid).name}.jpg"
    out_path = Path(output_dir) / flat_name
    
    if out_path.exists():
        return

    draw_paddle_boxes(local_img_path, local_json_path, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3-manifest', required=True, help='calibrecomics-extracted_manifest.csv')
    parser.add_argument('--local-manifest', required=True, help='master_manifest_20251229.csv')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--s3-bucket', default='calibrecomics-extracted')
    parser.add_argument('--s3-prefix', default='ocr_results_paddleocr')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files')
    args = parser.parse_args()
    
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    local_lookup = load_local_lookup(args.local_manifest)
    
    print(f"Loading S3 manifest: {args.s3_manifest}")
    rows = []
    with open(args.s3_manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
                
    print(f"Processing {len(rows)} pages...")
    s3 = boto3.client('s3')
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_row, row, local_lookup, args.input_dir, args.output_dir, args.s3_bucket, args.s3_prefix, s3) for row in rows]
        for _ in tqdm(as_completed(futures), total=len(rows)):
            pass

if __name__ == "__main__":
    main()