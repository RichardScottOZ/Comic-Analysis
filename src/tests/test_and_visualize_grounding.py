#!/usr/bin/env python3
"""
Test & Visualize VLM Grounding
Queries models for bounding boxes and immediately renders them onto the image.
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw

# Target Page (Default)
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001_#Guardian 001 - p003.jpg.png"

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
3. Every SPEECH BUBBLE (labelled 'text')

Return ONLY a JSON object with this format:
{
  "objects": [
    {"label": "panel", "box_2d": [ymin, xmin, ymax, xmax]},
    {"label": "person", "box_2d": [ymin, xmin, ymax, xmax]},
    {"label": "text", "box_2d": [ymin, xmin, ymax, xmax]}
  ]
}
Coordinates should be normalized 0-1000.
"""

def draw_boxes(image_path, objects, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Colors
        colors = {'panel': 'blue', 'person': 'red', 'text': 'green'}
        
        for obj in objects:
            label = obj.get('label', 'unknown').lower()
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
            
            color = colors.get(label, 'yellow')
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=5)
            
            # Draw Label Text
            try:
                # Use default font
                draw.text((abs_xmin + 5, abs_ymin + 5), label, fill=color)
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
                 content += ']}' # Very basic attempt
            try:
                data = json.loads(content.strip())
            except:
                print(f"❌ JSON Parse Error. Raw start: {content[:100]}")
                return

        objects = data.get('objects', [])
        print(f"Found {len(objects)} objects.")
        
        # Save JSON for inspection
        safe_model = model.replace('/', '_').replace(':', '_')
        json_out = f"grounding_output_{safe_model}.json"
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved JSON: {json_out}")
        
        # 3. Visualize
        out_name = f"grounding_viz_{safe_model}.jpg"
        draw_boxes(image_path, objects, out_name)

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
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-3-flash-preview",
        "google/gemini-3-pro-image-preview",
        "qwen/qwen3-vl-8b-instruct",
        "amazon/nova-lite-v1"
    ]
    
    if not args.api_key:
        print("Need API Key")
    else:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
        else:
            for m in models:
                run_test(m, args.api_key, args.image, args.temperature)
