#!/usr/bin/env python3
"""
Test R-CNN Enhanced Analysis
Feeds Faster R-CNN detection data into the VLM prompt to improve accuracy.
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path

# Paths
ORIGINAL_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
RCNN_JSON = "E:\\Comic_Analysis_Results_v2\\detections\\#Guardian 001\\#Guardian 001 - p003.json"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        header = image_file.read(12)
        image_file.seek(0)
        content = image_file.read()
        
    if header.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
    elif header.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
    else: mime = 'image/jpeg'

    return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"

def get_enhanced_prompt(rcnn_data):
    # Simplify R-CNN data for the prompt to save tokens
    detections = []
    if rcnn_data and 'detections' in rcnn_data:
        for d in rcnn_data['detections']:
            if d['score'] > 0.5:
                detections.append({
                    "label": d['label'],
                    "box": [int(x) for x in d['box_xyxy']]
                })
    
    json_context = json.dumps(detections, indent=None)

    return f"""Analyze this comic page and provide a detailed structured analysis in JSON format. 

GUIDANCE:
I have pre-detected the following objects using a specialized model (Faster R-CNN):
{json_context}

Use these detections to guide your analysis. Focus on:
1. **Panel Analysis**: Identify and describe each detected panel
2. **Character Identification**: Note characters, their actions, and dialogue
3. **Story Elements**: Plot points, setting, mood
4. **Visual Elements**: Art style, colors, composition
5. **Text Elements**: Speech bubbles, captions, sound effects

Return ONLY valid JSON with this structure:
{{
  "overall_summary": "Brief description of the page",
  "panels": [
    {{
      "panel_number": 1,
      "caption": "Panel title/description",
      "description": "Detailed panel description",
      "speakers": [
        {{
          "character": "Character name",
          "dialogue": "What they say",
          "speech_type": "dialogue|thought|narration"
        }}
      ],
      "key_elements": ["element1", "element2"],
      "actions": ["action1", "action2"]
    }}
  ],
  "summary": {{
    "characters": ["Character1", "Character2"],
    "setting": "Setting description",
    "plot": "Plot summary",
    "dialogue": ["Line1", "Line2"]
  }}
}}"""
def run_test(model, api_key):
    print(f"\n--- Testing R-CNN Enhanced: {model} ---")
    
    # Load R-CNN Data
    try:
        with open(RCNN_JSON, 'r', encoding='utf-8') as f:
            rcnn_data = json.load(f)
    except Exception as e:
        print(f"Error loading R-CNN JSON: {e}")
        return

    prompt_text = get_enhanced_prompt(rcnn_data)
    
    try:
        uri = encode_image(ORIGINAL_IMAGE)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "R-CNN Enhanced Test"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 4000,
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

            # Save Output
            safe_model = model.replace('/', '_').replace(':', '_')
            filename = f"experiment_rcnn_enhanced_{safe_model}.json"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved to {filename}")
            
        else:
            print(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()
    
    # Cheap Models to Boost
    models = [
        "amazon/nova-lite-v1",
        "google/gemini-2.0-flash-lite-001",
        "google/gemma-3-12b-it",
        "meta-llama/llama-4-scout",
        "qwen/qwen3-vl-8b-instruct"
    ]
    
    if not args.api_key:
        print("Need API Key")
    else:
        for m in models:
            run_test(m, args.api_key)
