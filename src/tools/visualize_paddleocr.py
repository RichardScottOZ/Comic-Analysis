#!/usr/bin/env python3
"""
Visualize PaddleOCR Results
Draws bounding boxes and text from PaddleOCR JSONs onto the source image.
"""

import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_paddle_boxes(image_path, json_path, output_path):
    print(f"Loading PaddleOCR JSON: {json_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = data.get('paddleocr_results', [])
        if not results:
            print("No OCR results found.")
            return

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font = ImageFont.load_default()

        print(f"Drawing {len(results)} text regions...")
        
        for res in results:
            # PaddleOCR returns a 4-point polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            poly = res.get('bbox') 
            text = res.get('text', '')
            conf = res.get('confidence', 0.0)
            
            if not poly: continue
            
            # Flatten to list of tuples for polygon drawing
            # [(x1,y1), (x2,y2), ...]
            points = [(p[0], p[1]) for p in poly]
            
            # Draw Polygon (Red outline)
            draw.polygon(points, outline='red', width=3)
            
            # Draw Text Label at top-left of polygon
            # Use a small background box for readability
            txt_x, txt_y = points[0]
            label = f"{text[:20]} ({conf:.2f})" # Truncate long text
            
            if hasattr(draw, "textbbox"):
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                w, h = right - left, bottom - top
            else:
                w, h = draw.textsize(label, font=font)
                
            draw.rectangle([txt_x, txt_y - h, txt_x + w, txt_y], fill='red')
            draw.text((txt_x, txt_y - h), label, fill='white', font=font)

        img.save(output_path)
        print(f"✅ Saved visualization: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to original image')
    parser.add_argument('--json', required=True, help='Path to PaddleOCR JSON')
    parser.add_argument('--output', default='viz_paddleocr.jpg', help='Output image path')
    args = parser.parse_args()
    
    draw_paddle_boxes(args.image, args.json, args.output)
