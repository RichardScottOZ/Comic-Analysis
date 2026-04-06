#!/usr/bin/env python3
"""
Test Gemini 3.1 Flash Lite: Full Multimodal Grounding & Narrative
Combines structural grounding (boxes) with deep narrative analysis (dialogue, descriptions).
"""

import os
import argparse
import base64
import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw

# Target Page
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p002.jpg.png"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    return f"data:image/jpeg;base64,{base64.b64encode(content).decode('utf-8')}"

def get_full_analysis_prompt():
    return """Analyze this comic page in detail.
For every distinct PANEL, identify its bounding box and describe its content.
Include all dialogue, captions, and character identities.

Return ONLY a JSON object with this exact structure:
{
  "overall_summary": "Summary of the whole page",
  "panels": [
    {
      "panel_number": 1,
      "box_2d": [ymin, xmin, ymax, xmax],
      "description": "Visual description of the panel",
      "text_content": [
        {"label": "dialogue|caption|thought", "speaker": "Name", "text": "Literal text", "box_2d": [ymin, xmin, ymax, xmax]}
      ],
      "characters": [
        {"name": "Name", "box_2d": [ymin, xmin, ymax, xmax]}
      ]
    }
  ]
}

STRICT RULES:
- Coordinates MUST be normalized (0-1000).
- box_2d is [ymin, xmin, ymax, xmax].
- Ensure every panel, text bubble, and character has a corresponding box_2d.
"""

def repair_json(s):
    s = s.strip()
    if not s: return "{}"
    if s.endswith(','): s = s[:-1]
    brace_diff = s.count('{') - s.count('}')
    bracket_diff = s.count('[') - s.count(']')
    if bracket_diff > 0: s += ']' * bracket_diff
    if brace_diff > 0: s += '}' * brace_diff
    return s

def draw_analysis(image_path, data, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # We'll just draw the Panel boxes for this viz
        panels = data.get('panels', [])
        for p in panels:
            box = p.get('box_2d', [])
            if len(box) != 4: continue
            
            ymin, xmin, ymax, xmax = box
            x0, y0, x1, y1 = (xmin/1000)*width, (ymin/1000)*height, (xmax/1000)*width, (ymax/1000)*height
            
            # Draw Panel Box (Blue)
            draw.rectangle([x0, y0, x1, y1], outline='blue', width=8)
            
            # Draw Text Boxes (Green)
            for t in p.get('text_content', []):
                tbox = t.get('box_2d', [])
                if len(tbox) == 4:
                    ty0, tx0, ty1, tx1 = (tbox[0]/1000)*height, (tbox[1]/1000)*width, (tbox[2]/1000)*height, (tbox[3]/1000)*width
                    draw.rectangle([tx0, ty0, tx1, ty1], outline='green', width=3)

        img.save(output_path)
        print(f"✅ Saved visualization: {output_path}")
    except Exception as e:
        print(f"Error drawing: {e}")

def run_test(model, api_key, image_path):
    print(f"\n--- Testing Full Analysis: {model} ---")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": get_full_analysis_prompt()}, {"type": "image_url", "image_url": {"url": encode_image(image_path)}}]}],
        "max_tokens": 8192
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=240)
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return

    res_json = response.json()
    if 'choices' not in res_json:
        print(f"❌ API returned an unexpected response format. Missing 'choices'.")
        print(f"Full Response: {json.dumps(res_json, indent=2)}")
        return
        
    content = res_json['choices'][0]['message']['content']
    print(f"Response Length: {len(content)}")

    # Cleanup Markdown
    start = content.find('{')
    end = content.rfind('}')
    clean_json = content[start:end+1] if (start != -1 and end != -1) else content

    try:
        data = json.loads(clean_json.strip(), strict=False)
    except:
        print("⚠️ Truncated response detected. Repairing...")
        try:
            data = json.loads(repair_json(clean_json))
        except Exception as e:
            print(f"❌ Fatal JSON Parse Error: {e}")
            print(f"Raw Model Output:\n{content}")
            return

    safe_model = model.replace('/', '_').replace(':', '_')
    with open(f"full_analysis_{safe_model}.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved Full JSON: full_analysis_{safe_model}.json")
    
    draw_analysis(image_path, data, f"full_analysis_viz_{safe_model}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--model', default="google/gemini-3.1-flash-lite-preview")
    args = parser.parse_args()
    
    if not args.api_key: print("Need API Key")
    elif not os.path.exists(args.image): print(f"Image not found: {args.image}")
    else: run_test(args.model, args.api_key, args.image)
