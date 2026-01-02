#!/usr/bin/env python3
"""
Test Visual Prompting vs JSON Prompting
Can providing Faster R-CNN results (visual or text) improve VLM grounding?
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path

# Paths (Update if needed)
ORIGINAL_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
ANNOTATED_IMAGE = "viz_faster_rcnn_p003.jpg" # Created by visualize_faster_rcnn.py
RCNN_JSON = "E:\\Comic_Analysis_Results_v2\\detections\\#Guardian 001\\#Guardian 001 - p003.json"

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

def run_json_prompt_test(model, api_key):
    print(f"\n--- Testing JSON Prompting ({model}) ---")
    
    # Load R-CNN JSON
    try:
        with open(RCNN_JSON, 'r', encoding='utf-8') as f:
            rcnn_data = json.load(f)
            # Minimize tokens by stripping unnecessary fields
            detections = [{"label": d['label'], "box": d['box_xyxy']} for d in rcnn_data.get('detections', []) if d['score'] > 0.5]
            json_context = json.dumps(detections, indent=None)
    except Exception as e:
        print(f"Failed to load R-CNN JSON: {e}")
        return

    prompt = f"""Analyze this comic page and provide a detailed structured analysis in JSON format.

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
    
    _call_api(model, api_key, ORIGINAL_IMAGE, prompt, "json_prompt")

def run_visual_prompt_test(model, api_key):
    print(f"\n--- Testing Visual Prompting ({model}) ---")
    
    if not os.path.exists(ANNOTATED_IMAGE):
        print(f"Error: Annotated image {ANNOTATED_IMAGE} not found. Run visualize_faster_rcnn.py first.")
        return

    prompt = """Analyze this ANNOTATED comic page.
Bounding boxes have been drawn for Panels (Blue), Characters (Red), and Text (Green).

GUIDANCE:
Use these visual annotations to guide your analysis.
1. Read the labels on the image (e.g. 'panel 0.99', 'text 0.99').
2. Treat each Blue Box as a panel.
3. Transcribe text inside Green Boxes.
4. Describe characters inside Red Boxes.

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
    _call_api(model, api_key, ANNOTATED_IMAGE, prompt, "visual_prompt")

def _call_api(model, api_key, image_path, prompt, tag):
    try:
        uri = encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "Visual Prompt Test"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 8192
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            res_json = response.json()
            
            # Robust extraction
            if 'choices' in res_json:
                content = res_json['choices'][0]['message']['content']
            elif 'error' in res_json:
                print(f"API Returned Error: {res_json['error']}")
                return
            else:
                print(f"Unexpected Response Structure. Keys: {list(res_json.keys())}")
                print(f"Dump: {str(res_json)[:200]}")
                return
            
            # Save raw output
            safe_model = model.replace('/', '_')
            filename = f"experiment_{tag}_{safe_model}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Saved output to {filename}")
            print(f"Preview: {content[:200]}...")
        else:
            print(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args()
    
    # Models to Test
    models = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-2.5-flash-image-preview",
        "google/gemini-3-flash-preview",
        "amazon/nova-lite-v1",
        "z-ai/glm-4.6v",
        "bytedance-seed/seed-1.6",
        # "qwen/qwen3-vl-8b-instruct",
        # "google/gemini-pro-1.5",
        # "anthropic/claude-3-haiku",
        # "meta-llama/llama-4-scout"
    ]
    
    if not args.api_key:
        print("Need API Key")
    else:
        for m in models:
            run_json_prompt_test(m, args.api_key)
            run_visual_prompt_test(m, args.api_key)
