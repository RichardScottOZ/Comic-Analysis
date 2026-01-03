#!/usr/bin/env python3
"""
Test Zhipu AI Native API (GLM-4V-Flash)
Supports multiple modes: Analysis, Grounding, Full (Integrated).
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

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

# --- Prompts ---

def get_analysis_prompt():
    return """Analyze this comic page and provide a detailed structured analysis in JSON format. Focus on:
1. **Panel Analysis**: Identify and describe each panel
2. **Character Identification**: Note characters, their actions, and dialogue
3. **Story Elements**: Plot points, setting, mood
4. **Visual Elements**: Art style, colors, composition
5. **Text Elements**: Speech bubbles, captions, sound effects

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
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

def get_grounding_prompt():
    return """Identify the bounding box [xmin, ymin, xmax, ymax] (0-1000) for:
1. Every PANEL (labelled 'panel')
2. Every CHARACTER (labelled 'person')
3. Every FACE (labelled 'face')
4. Every SPEECH BUBBLE (labelled 'text')

STRICT RULES:
- Return ONLY ONE label per distinct region. 
- Prioritize specific labels (face/text) over general ones.

Return ONLY valid JSON:
{
  "objects": [
    {"label": "panel|person|face|text", "box_2d": [xmin, ymin, xmax, ymax]}
  ]
}
"""

def get_full_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis AND bounding boxes in JSON format.

REQUIREMENTS:
1. Identify every panel. For each panel, provide its BOUNDING BOX [xmin, ymin, xmax, ymax] (0-1000).
2. Identify characters, faces, and text bubbles with bounding boxes.
3. Describe the visual content and action.
4. Transcribe all dialogue and attribute it to characters.

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "objects": [
    {"label": "panel|person|face|text", "box_2d": [xmin, ymin, xmax, ymax]}
  ],
  "panels": [
    {
      "panel_number": 1,
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

def draw_zhipu_boxes(image_path, json_path, output_path):
    try:
        # 1. Load Image first to get dimensions
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # 2. Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 3. Setup Colors
        colors = {
            'panel': 'blue', 
            'person': 'red', 
            'text': 'green', 
            'face': 'magenta',
            'car': 'cyan',
            'building': 'orange'
        }
        
        # 4. Collect Objects
        all_objects = data.get('objects', [])
        if 'panels' in data:
            for p in data['panels']:
                box = p.get('box_2d') or p.get('box')
                if box:
                    all_objects.append({'label': 'panel', 'box_2d': box})

        if not all_objects:
            print(f"No objects to draw for {output_path}")
            return

        print(f"Drawing {len(all_objects)} objects...")
        
        # 5. Draw
        for obj in all_objects:
            full_label = obj.get('label', 'obj').lower()
            base_label = full_label.split('|')[0].strip()
            
            box = obj.get('box_2d') or obj.get('box')
            
            if not box or len(box) != 4: continue
            
            # Zhipu: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box
            
            # Normalize
            abs_xmin = (xmin / 1000) * width
            abs_ymin = (ymin / 1000) * height
            abs_xmax = (xmax / 1000) * width
            abs_ymax = (ymax / 1000) * height
            
            color = colors.get(base_label, 'yellow')
            draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline=color, width=4)
            
            # Label
            draw.text((abs_xmin+5, abs_ymin+5), full_label, fill=color)

        img.save(output_path)
        print(f"✅ Saved viz: {output_path}")
        
    except Exception as e:
        print(f"Vis Error: {e}")

def run_test(model, api_key, temperature=None, mode='full'):
    temp_str = f"temp-{temperature}" if temperature is not None else "temp-default"
    print(f"\n--- Testing Zhipu {mode.upper()}: {model} ({temp_str}) ---")
    
    prompt = ""
    if mode == 'analysis': prompt = get_analysis_prompt()
    elif mode == 'grounding': prompt = get_grounding_prompt()
    elif mode == 'full': prompt = get_full_prompt()
    
    try:
        uri = encode_image(ORIGINAL_IMAGE)
        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 4095,
            "thinking": {"type": "enabled"}
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            
            filename = f"experiment_zhipu_{mode}_{model}_{temp_str}.json"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved to {filename}")
            
            if mode in ['grounding', 'full']:
                out_viz = f"viz_zhipu_{mode}_{model}_{temp_str}.jpg"
                draw_zhipu_boxes(ORIGINAL_IMAGE, filename, out_viz)
        else:
            print(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("Z_API_KEY"), help="Zhipu API Key")
    parser.add_argument('--model', default="glm-4.6v-flash") 
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--mode', choices=['analysis', 'grounding', 'full', 'all'], default='full')
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: Z_API_KEY required.")
    else:
        if args.mode == 'all':
            modes = ['analysis', 'grounding', 'full']
            with ThreadPoolExecutor(max_workers=3) as executor:
                for m in modes:
                    executor.submit(run_test, args.model, args.api_key, args.temperature, m)
        else:
            run_test(args.model, args.api_key, args.temperature, args.mode)