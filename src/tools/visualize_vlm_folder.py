#!/usr/bin/env python3
"""
Visualize VLM Folder (Batch Mode)
Generates bounding box visualizations for an entire folder of VLM JSON results.
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_image_path(canonical_id, image_root):
    # Try various extensions and path styles
    candidates = [
        Path(image_root) / f"{canonical_id}.jpg",
        Path(image_root) / f"{canonical_id}.png",
        Path(image_root) / f"{canonical_id}.jpeg",
        # Handle flattened vs nested
        Path(image_root) / Path(canonical_id).name
    ]
    
    # Handle the '#' vs '/' issue in windows paths if needed
    clean_id = canonical_id.replace('/', os.sep)
    candidates.append(Path(image_root) / f"{clean_id}")
    
    for c in candidates:
        if c.exists():
            return c
    return None

def process_single_file(json_path, image_root, output_dir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        canonical_id = data.get('canonical_id')
        if not canonical_id:
            return # Skip invalid JSON
            
        img_path = get_image_path(canonical_id, image_root)
        if not img_path:
            # Fallback: try to guess from JSON filename
            # e.g. .../Folder_File.json -> Folder/File
            return 

        # Create output path
        out_name = f"viz_{Path(json_path).stem}.jpg"
        out_path = Path(output_dir) / out_name
        
        if out_path.exists():
            return

        # Load Image
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Load Font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Extract Boxes
        objects = []
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    objects.append({'label': f"P{p.get('panel_number')}", 'box': box, 'color': 'blue'})
        
        if 'objects' in data:
            for obj in data['objects']:
                box = obj.get('box_2d') or obj.get('box')
                if box:
                    color = 'green' if 'text' in obj['label'] else 'red' if 'person' in obj['label'] else 'blue'
                    objects.append({'label': obj['label'], 'box': box, 'color': color})

        if not objects:
            return

        # Draw
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
        print(f"Error processing {json_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory containing VLM JSONs')
    parser.add_argument('--image-root', required=True, help='Root directory of original images')
    parser.add_argument('--output-dir', required=True, help='Directory to save visualizations')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = list(Path(args.input_dir).rglob("*.json"))
    if args.limit:
        files = files[:args.limit]
        
    print(f"Processing {len(files)} files...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, f, args.image_root, args.output_dir) for f in files]
        for _ in tqdm(as_completed(futures), total=len(files)):
            pass

if __name__ == "__main__":
    from concurrent.futures import as_completed
    main()
