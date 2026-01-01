#!/usr/bin/env python3
"""
Test & Visualize VLM Grounding
Queries models for bounding boxes and immediately renders them onto the image.
Saves RAW and CLEAN versions of both JSON and Visualization.
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw

# Target Page (Default)
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        header = image_file.read(12)
        image_file.seek(0)
        content = image_file.read()
        
    # Detect MIME type
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        mime_type = 'image/png'
    elif header.startswith(b'\xff\xd8\xff'):
        mime_type = 'image/jpeg'
    elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        mime_type = 'image/webp'
    else:
        mime_type = 'image/jpeg' # Fallback

    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"

def get_grounding_prompt():
    return """Analyze this comic page.
Identify the bounding box for:
1. Every PANEL (labelled 'panel')
2. Every CHARACTER (labelled 'person')
3. Every FACE (labelled 'face')
4. Every SPEECH BUBBLE (labelled 'text')

STRICT RULES:
- Return ONLY ONE label per distinct region. 
- DO NOT assign multiple labels (e.g., both 'panel' and 'person') to the same or nearly identical coordinates.
- If a character or text bubble takes up an entire region, prioritize the 'person' or 'text' label over 'panel'.

Return ONLY a JSON object with this format:
{
  "objects": [
    {"label": "panel", "box_2d": [ymin, xmin, ymax, xmax]},
    {"label": "person", "box_2d": [ymin, xmin, ymax, xmax]},
    {"label": "face", "box_2d": [ymin, xmin, ymax, xmax]},
    {"label": "text", "box_2d": [ymin, xmin, ymax, xmax]}
  ]
}
Coordinates should be normalized 0-1000.
"""

def deduplicate_objects(objects):
    """
    Cleans up duplicate detections where the model assigns multiple labels 
    (e.g., panel + person) to the exact same box.
    Priority: text > face > person > panel
    """
    if not objects: return []
    
    # Sort by priority score (higher is better)
    priority = {'text': 4, 'face': 3, 'person': 2, 'panel': 1}
    
    # Helper to calculate IoU
    def get_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[1], boxB[1])
        yA = max(boxA[0], boxB[0])
        xB = min(boxA[3], boxB[3])
        yB = min(boxA[2], boxB[2])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[0])
        boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[0])
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    # Group by box coordinates (approximate matching)
    cleaned_objects = []
    # Sort objects so we process highest priority first
    objects.sort(key=lambda x: priority.get(x.get('label', ''), 0), reverse=True)
    
    for obj in objects:
        is_duplicate = False
        box = obj.get('box_2d') or obj.get('box')
        if not box or len(box) != 4: continue
        
        for kept in cleaned_objects:
            kept_box = kept.get('box_2d') or kept.get('box')
            iou = get_iou(box, kept_box)
            
            # If boxes are nearly identical (IoU > 0.95), keep the one with higher priority
            # Since we sorted by priority, 'kept' is already the higher priority one.
            if iou > 0.95:
                is_duplicate = True
                break
        
        if not is_duplicate:
            cleaned_objects.append(obj)
            
    return cleaned_objects

def draw_boxes(image_path, objects, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Colors
        colors = {'panel': 'blue', 'person': 'red', 'text': 'green', 'face': 'magenta'}
        
        for obj in objects:
            label = obj.get('label', 'unknown').lower()
            oid = obj.get('id', '?')
            # Handle inconsistent keys from some models (box vs box_2d)
            box = obj.get('box_2d', [])
            if not box:
                box = obj.get('box', [])
            
            if len(box) != 4: continue
            
            ymin, xmin, ymax, xmax = box
            
            # Normalize 0-1000 -> Pixels
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            # Fix swapped coordinates (e.g. xmin > xmax)
            x0, x1 = sorted([abs_xmin, abs_xmax])
            y0, y1 = sorted([abs_ymin, abs_ymax])
            
            color = colors.get(label, 'yellow')
            draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
            
            # Draw Label Text
            try:
                # Use default font
                label_text = f"#{oid} {label}"
                draw.text((x0 + 5, y0 + 5), label_text, fill=color)
            except:
                pass # Fallback if font issues

        img.save(output_path)
        print(f"✅ Saved visualization: {output_path}")
        
    except Exception as e:
        print(f"Error drawing boxes: {e}")

def run_test(model, api_key, image_path, temperature=None):
    print(f"\n--- Testing {model} ---")
    
    # 1. API Call
    try:
        uri = encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "Comic Grounding Test"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": get_grounding_prompt()}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 4000
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return

        res_json = response.json()
        
        # Robust Choice Extraction
        try:
            content = res_json['choices'][0]['message']['content']
        except (KeyError, IndexError):
            print(f"❌ Could not find content in response. Keys: {list(res_json.keys())}")
            print(f"Full Response: {str(res_json)[:500]}")
            return
        
        # 2. Parse JSON
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        try:
            data = json.loads(content.strip(), strict=False)
        except json.JSONDecodeError:
            # Try a quick repair for common truncation
            if content.strip().startswith('{') and not content.strip().endswith('}'):
                 content += ']}\n' # Very basic attempt
            try:
                data = json.loads(content.strip())
            except:
                print(f"❌ JSON Parse Error. Raw start: {content[:100]}")
                return

        raw_objects = data.get('objects', [])
        print(f"Found {len(raw_objects)} raw objects.")
        
        # Add IDs to raw objects
        for i, obj in enumerate(raw_objects):
            obj['id'] = i
            
        safe_model = model.replace('/', '_').replace(':', '_')
        
        # Save Raw JSON
        raw_json_out = f"grounding_output_{safe_model}_raw.json"
        with open(raw_json_out, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved Raw JSON: {raw_json_out}")
        
        # Visualize Raw
        out_name_raw = f"grounding_viz_{safe_model}_raw.jpg"
        draw_boxes(image_path, raw_objects, out_name_raw)
        
        # Cleanup
        clean_objects = deduplicate_objects(raw_objects)
        
        # Re-assign IDs for clean list
        for i, obj in enumerate(clean_objects):
            obj['id'] = i
            
        print(f"Cleaned to {len(clean_objects)} unique objects.")
        
        # Save Clean JSON
        clean_data = data.copy()
        clean_data['objects'] = clean_objects
        clean_json_out = f"grounding_output_{safe_model}_clean.json"
        with open(clean_json_out, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2)
        print(f"✅ Saved Clean JSON: {clean_json_out}")
        
        # Visualize Clean
        out_name_clean = f"grounding_viz_{safe_model}_clean.jpg"
        draw_boxes(image_path, clean_objects, out_name_clean)

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--temperature', type=float, default=None)
    args = parser.parse_args()
    
    models = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-flash-preview",
        # "google/gemini-2.0-flash-lite-001",
        # "qwen/qwen3-vl-8b-instruct",
        # "amazon/nova-lite-v1",
        # "mistralai/mistral-small-3.2-24b-instruct",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-27b-it",
        # "meta-llama/llama-4-scout"
    ]
    
    if not args.api_key:
        print("Need API Key")
    else:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
        else:
            for m in models:
                run_test(m, args.api_key, args.image, args.temperature)