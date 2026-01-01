#!/usr/bin/env python3
"""
Test Integrated Analysis + Grounding
Can VLMs perform narrative analysis AND bounding box detection in a single pass?
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

def get_integrated_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis in JSON format.

REQUIREMENTS:
1. Identify every panel. For each panel, provide its BOUNDING BOX [ymin, xmin, ymax, xmax] (0-1000).
2. Describe the visual content and action.
3. Transcribe all dialogue and attribute it to characters.

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "Detailed panel description",
      "speakers": [
        {
          "character": "Character name",
          "dialogue": "What they say"
        }
      ]
    }
  ]
}
"""

from PIL import Image, ImageDraw

def draw_integrated_boxes(image_path, json_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        panels = data.get('panels', [])
        print(f"Drawing {len(panels)} panels...")
        
        for p in panels:
            box = p.get('box_2d')
            if not box or len(box) != 4: continue
            
            ymin, xmin, ymax, xmax = box
            
            # Normalize
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            # Draw Blue Box
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline='blue', width=5)
            
            # Label
            label = f"P{p.get('panel_number', '?')}"
            draw.text((abs_xmin+5, abs_ymin+5), label, fill='white')

        img.save(output_path)
        print(f"✅ Saved viz: {output_path}")
        
    except Exception as e:
        print(f"Vis Error: {e}")

def run_test(model, api_key):
    print(f"\n--- Testing Integrated Analysis: {model} ---")
    
    try:
        uri = encode_image(ORIGINAL_IMAGE)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "Integrated VLM Test"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": get_integrated_prompt()}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 8192,
            "temperature": 0.0
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
            elif 'error' in res_json:
                print(f"API Error: {res_json['error']}")
                return
            else:
                print(f"Unexpected: {str(res_json)[:200]}")
                return

            # Clean JSON (Markdown)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            # Save Output
            safe_model = model.replace('/', '_').replace(':', '_')
            filename = f"experiment_integrated_{safe_model}.json"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved JSON to {filename}")
            
            # Visualize
            out_viz = f"viz_integrated_{safe_model}.jpg"
            draw_integrated_boxes(ORIGINAL_IMAGE, filename, out_viz)
            
        else:
            print(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()
    
    # Models
    models = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "amazon/nova-lite-v1",
        "qwen/qwen3-vl-8b-instruct"
    ]
    
    if not args.api_key:
        print("Need API Key")
    else:
        for m in models:
            run_test(m, args.api_key)
