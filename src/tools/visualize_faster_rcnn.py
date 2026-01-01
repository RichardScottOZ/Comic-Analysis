#!/usr/bin/env python3
"""
Visualize Faster R-CNN Results
"""
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import os

def draw_rcnn_boxes(image_path, json_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Loaded Image: {image_path}")
        print(f"Dimensions: {img.width} x {img.height}")
        
        draw = ImageDraw.Draw(img)
        
        # Load smaller font
        try:
            # Windows path
            font = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            try:
                # Linux path
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except OSError:
                print("Warning: Could not load TrueType font. Using default (tiny) font.")
                font = ImageFont.load_default()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        json_size = data.get('image_size_wh', [1, 1]) # [w, h]
        json_w, json_h = json_size
        
        # Calculate scaling factor
        scale_x = img.width / json_w
        scale_y = img.height / json_h
        print(f"Scaling detection boxes: X={scale_x:.3f}, Y={scale_y:.3f}")

        detections = data.get('detections', [])
        
        # Sort by Area Descending
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
        
        print(f"Drawing {len(detections)} detections...")
        
        for i, det in enumerate(detections):
            label = det.get('label')
            raw_box = det.get('box_xyxy')
            score = det.get('score', 0)
            
            if score < 0.5: 
                continue
            
            # Apply Scale
            box = [
                raw_box[0] * scale_x,
                raw_box[1] * scale_y,
                raw_box[2] * scale_x,
                raw_box[3] * scale_y
            ]
            
            color = colors.get(label, 'yellow')
            
            # 1. Draw box
            draw.rectangle(box, outline=color, width=4)
            
            # 2. Prepare Label Text
            label_text = f"{label} {score:.2f}"
            
            # 3. Draw text with stroke
            txt_x = box[0] + 5
            txt_y = box[1] + 5
            
            # Clamp
            txt_x = max(0, min(txt_x, img.width - 80))
            txt_y = max(0, min(txt_y, img.height - 15))
            
            try:
                draw.text((txt_x, txt_y), label_text, fill='white', font=font, stroke_width=1, stroke_fill='black')
            except:
                draw.text((txt_x, txt_y), label_text, fill=color, font=font)
            
        img.save(output_path)
        print(f"Saved R-CNN visualization: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
            
        img.save(output_path)
        print(f"Saved R-CNN visualization: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to original image')
    parser.add_argument('--json', required=True, help='Path to Faster R-CNN detection JSON')
    parser.add_argument('--output', default='viz_faster_rcnn.jpg', help='Output image path')
    args = parser.parse_args()
    
    draw_rcnn_boxes(args.image, args.json, args.output)
