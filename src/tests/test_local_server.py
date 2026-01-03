#!/usr/bin/env python3
"""
Test Local Llama Server (GLM-4V-Flash)
Connects to a running llama-server instance (OpenAI compatible).

Usage:
1. Start server: 
   llama-server.exe -m model.gguf --mmproj mmproj.gguf --port 8080 -ngl 99
   
2. Run this script:
   python src/tests/test_local_server.py
"""

import argparse
import base64
import requests
import json
import os

# Paths
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
SERVER_URL = "http://localhost:80/v1/chat/completions"

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
      "caption": "Panel title/description",
      "description": "Detailed panel description",
      "speakers": [
        {
          "character": "Character name",
          "dialogue": "What they say",
          "speech_type": "dialogue|thought|narration"
        }
      ],
      "key_elements": ["element1", "element2"],
      "actions": ["action1", "action2"]
    }
  ],
  "summary": {
    "characters": ["Character1", "Character2"],
    "setting": "Setting description",
    "plot": "Plot summary",
    "dialogue": ["Line1", "Line2"]
  }
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
            box = p.get('box_2d') or p.get('box')
            if not box or len(box) != 4: continue
            
            # Local GLM GGUF (like Zhipu) usually does [xmin, ymin, xmax, ymax]
            # Let's detect based on which dimension is bigger, or just use [xmin, ymin...]
            xmin, ymin, xmax, ymax = box
            
            # Normalize 0-1000 -> Pixels
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

def run_test(image_path):
    print(f"Connecting to local server: {SERVER_URL}")
    print(f"Processing image: {image_path}")
    
    try:
        uri = encode_image(image_path)
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": "gpt-3.5-turbo", # Dummy name, server ignores it usually
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": get_integrated_prompt()},
                        {"type": "image_url", "image_url": {"url": uri}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=600)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            
            # Clean JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            print("\n--- Model Output ---")
            print(content)
            
            out_json = "experiment_local_server_output.json"
            with open(out_json, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\n✅ Saved to {out_json}")
            
            # Visualize
            out_viz = "viz_local_server.jpg"
            draw_integrated_boxes(image_path, out_json, out_viz)
            
        else:
            print(f"Server Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to localhost:8080. Is llama-server running?")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
    else:
        run_test(args.image)
