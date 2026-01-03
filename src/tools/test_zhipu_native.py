#!/usr/bin/env python3
"""
Test Zhipu AI Native API (GLM-4V-Flash)
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path

# Paths
ORIGINAL_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        header = image_file.read(12)
        image_file.seek(0)
        content = image_file.read()
        
    if header.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
    elif header.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
    else: mime = 'image/jpeg'

    return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"

def get_full_grounding_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis in JSON format.
Identify the bounding box [xmin, ymin, xmax, ymax] (0-1000) for:
1. Every PANEL (labelled 'panel')
2. Every CHARACTER (labelled 'person')
3. Every FACE (labelled 'face')
4. Every SPEECH BUBBLE (labelled 'text')

Return ONLY valid JSON with this structure:
{
  "overall_summary": "...",
  "objects": [
    {"label": "panel|person|face|text", "box_2d": [xmin, ymin, xmax, ymax]}
  ],
  "panels": [
    {
      "panel_number": 1,
      "description": "...",
      "speakers": [...]
    }
  ]
}
"""

def draw_zhipu_boxes(image_path, json_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Standard colors
        colors = {'panel': 'blue', 'person': 'red', 'text': 'green', 'face': 'magenta'}
        
        # Check both 'objects' and 'panels'
        all_objects = data.get('objects', [])
        
        # Add panels from the panels list if not in objects
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    all_objects.append({'label': 'panel', 'box_2d': box})

        print(f"Drawing {len(all_objects)} objects...")
        
        for obj in all_objects:
            label = obj.get('label', 'obj').lower()
            box = obj.get('box_2d') or obj.get('box')
            
            if not box or len(box) != 4: continue
            
            # Zhipu: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            
            # Normalize
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            color = colors.get(label, 'yellow')
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=4)
            
            # Label
            draw.text((abs_xmin+5, abs_ymin+5), label, fill=color)

        img.save(output_path)
        print(f"✅ Saved viz: {output_path}")
        
    except Exception as e:
        print(f"Vis Error: {e}")

def run_test(model, api_key):
    print(f"\n--- Testing Zhipu Full Grounding: {model} ---")
    
    try:
        uri = encode_image(ORIGINAL_IMAGE)
        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": get_full_grounding_prompt()}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 4095,
            "temperature": 0.1,
            "thinking": {"type": "enabled"}
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            
            filename = f"experiment_zhipu_full_{model}.json"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved to {filename}")
            
            out_viz = f"viz_zhipu_full_{model}.jpg"
            draw_zhipu_boxes(ORIGINAL_IMAGE, filename, out_viz)
        else:
            print(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("Z_API_KEY"), help="Zhipu API Key (defaults to Z_API_KEY env var)")
    parser.add_argument('--model', default="glm-4.6v-flash") 
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: Z_API_KEY environment variable not set and --api-key not provided.")
    else:
        run_test(args.model, args.api_key)
