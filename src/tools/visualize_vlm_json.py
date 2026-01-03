#!/usr/bin/env python3
"""
Visualize VLM JSON Analysis (Unified)
Draws bounding boxes from standard VLM analysis JSONs with robust label and coordinate handling.
"""

import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def draw_vlm_boxes(image_path, json_path, output_path, order='ymin_first'):
    print(f"Loading VLM JSON: {json_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        objects = []
        
        # 1. Extract from 'panels' list
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    objects.append({
                        'label': f"panel|{p.get('panel_number', '?')}",
                        'box': box
                    })
        
        # 2. Extract from 'objects' list
        if 'objects' in data:
            for obj in data['objects']:
                box = obj.get('box_2d') or obj.get('box')
                if box:
                    objects.append({
                        'label': obj.get('label', 'obj'),
                        'box': box
                    })

        if not objects:
            print("No bounding boxes found in this JSON.")
            return

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = ImageFont.load_default()

        # Color Map
        colors = {
            'panel': 'blue', 
            'person': 'red', 
            'text': 'green', 
            'face': 'magenta',
            'car': 'cyan',
            'building': 'orange'
        }

        print(f"Drawing {len(objects)} boxes...")
        
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
            
            # Normalize 0-1000 to pixels
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            # Draw
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=4)
            # Label
            draw.text((abs_xmin + 5, abs_ymin + 5), full_label, fill=color, font=font)

        img.save(output_path)
        print(f"✅ Saved visualization: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to original image')
    parser.add_argument('--json', required=True, help='Path to VLM analysis JSON')
    parser.add_argument('--output', default='viz_vlm_output.jpg', help='Output image path')
    parser.add_argument('--order', choices=['ymin_first', 'xmin_first'], default='ymin_first', help='Coordinate order')
    args = parser.parse_args()
    
    draw_vlm_boxes(args.image, args.json, args.output, args.order)