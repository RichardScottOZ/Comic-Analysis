#!/usr/bin/env python3
"""
Visualize PaddleOCR Folder (Manifest-Driven with S3 Download)
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

def load_manifest_rows(manifest_path, limit=None):
    print(f"Loading manifest: {manifest_path}")
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows

def download_json(s3, bucket, key, local_path):
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception:
        return False

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
            poly = res.get('bbox') 
            text = res.get('text', '')
            
            if not poly: continue
            
            points = [(p[0], p[1]) for p in poly]
            draw.polygon(points, outline='red', width=2)
            
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
        # print(f"Draw Error: {e}")
        pass

def process_row(row, input_dir, output_dir, s3_bucket, s3_prefix, s3_client):
    cid = row['canonical_id']
    
    # Image Path from Manifest (e.g. E:\...)
    # Warning: Manifest has s3:// paths? Or local paths?
    # User said "manifest has CalibreComics_extracted/amazon/... , s3://..."
    # If the user is running locally, we need the LOCAL image path.
    # If the manifest only has S3 paths, we can't draw on the image unless we download it too.
    
    # Let's assume absolute_image_path is correct for the local machine OR we need to map it.
    # Since previous scripts worked, let's assume it points to E:\...
    # Wait, the sample user pasted showed: "s3://calibrecomics-extracted/..."
    
    # If absolute_image_path is s3://, we CANNOT visualize without downloading the image.
    # But user implied they have images locally ("E:\amazon...").
    
    # Let's try to infer local path from CID if absolute_path is S3
    img_path = row['absolute_image_path']
    if img_path.startswith('s3://'):
        # Try to map to E:\CalibreComics_extracted or E:\amazon
        # Heuristic based on user's setup
        if 'amazon' in cid:
            # cid: CalibreComics_extracted/amazon/Title/Page
            # local: E:\amazon\Title\Page.jpg ??
            # This is risky.
            # Let's try to assume the user mounted it or has it at E:\
            pass 
    
    # Check if image exists
    if not os.path.exists(img_path):
        # Try mapping E:\CalibreComics_extracted
        # Construct local path from CID
        # cid: CalibreComics_extracted/amazon/3 Guns/3 Guns - p000
        # img: E:\CalibreComics_extracted\CalibreComics_extracted\amazon... ? No.
        
        # Let's try prepending E:\ to the relative part
        # Remove bucket prefix if present?
        pass

    # If we can't find the image, we can't draw.
    if not os.path.exists(img_path):
        # print(f"Missing Image: {img_path}")
        return

    # JSON Path
    # Structure in S3: s3_prefix / cid + "_ocr.json"
    # Local Cache: input_dir / cid + "_ocr.json" (Mirroring structure)
    
    local_json_path = Path(input_dir) / f"{cid}_ocr.json"
    
    if not local_json_path.exists():
        if s3_client:
            s3_key = f"{s3_prefix}/{cid}_ocr.json"
            if not download_json(s3_client, s3_bucket, s3_key, local_json_path):
                 # Try without _ocr
                 s3_key = f"{s3_prefix}/{cid}.json"
                 download_json(s3_client, s3_bucket, s3_key, local_json_path)
    
    if not local_json_path.exists():
        # print(f"Missing JSON: {local_json_path}")
        return

    # Output Path
    # Use flat name for easy viewing
    flat_name = f"viz_{Path(cid).name}.jpg"
    out_path = Path(output_dir) / flat_name
    
    if out_path.exists():
        return

    draw_paddle_boxes(img_path, local_json_path, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--s3-bucket', default='calibrecomics-extracted')
    parser.add_argument('--s3-prefix', default='ocr_results_paddleocr')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, help='Limit number of files')
    args = parser.parse_args()
    
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    rows = load_manifest_rows(args.manifest, args.limit)
    print(f"Processing {len(rows)} pages...")
    
    s3 = boto3.client('s3')
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_row, row, args.input_dir, args.output_dir, args.s3_bucket, args.s3_prefix, s3) for row in rows]
        for _ in tqdm(as_completed(futures), total=len(rows)):
            pass

if __name__ == "__main__":
    main()
