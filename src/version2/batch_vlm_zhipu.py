#!/usr/bin/env python3
"""
Manifest-driven Zhipu AI (GLM-4V-Flash) Batch Processing
Enhanced version with Guided Mode (R-CNN), Integrated Grounding, and JSON Repair.

Supports:
- Structured Analysis (Characters, Dialogue, Plot)
- Integrated Grounding (Bounding Boxes for Panels/Text)
- Guided Mode (Feeding Faster R-CNN boxes into the VLM)
- WSL/Windows path interoperability
"""

import os
import argparse
import csv
import json
import base64
import requests
import time
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Path Interoperability ---

def convert_to_wsl_path(path_str):
    """Convert Windows path to WSL path if running on Linux."""
    if os.name == 'posix' and ':' in path_str:
        drive, rest = path_str.split(':', 1)
        if len(drive) == 1:
            rest = rest.replace('\\', '/')
            return Path(f"/mnt/{drive.lower()}{rest}")
    return Path(path_str)

# --- Prompts ---

def create_structured_prompt():
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
}"""

def get_integrated_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis in JSON format.

REQUIREMENTS:
1. Identify every panel. For each panel, provide its BOUNDING BOX [xmin, ymin, xmax, ymax] (0-1000).
2. Describe the visual content and action.
3. Transcribe all dialogue and attribute it to characters.

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
      "box_2d": [xmin, ymin, xmax, ymax],
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

def get_enhanced_prompt(rcnn_data):
    detections = []
    if rcnn_data and 'detections' in rcnn_data:
        for d in rcnn_data['detections']:
            if d['score'] > 0.5:
                detections.append({"label": d['label'], "box": [int(x) for x in d['box_xyxy']]})
    
    json_context = json.dumps(detections, indent=None)
    return f"""Analyze this comic page and provide a detailed structured analysis in JSON format.

GUIDANCE:
I have pre-detected the following objects using a specialized model (Faster R-CNN):
{json_context}

Use these detections to guide your analysis. Return ONLY valid JSON with the standard structure (panels, summary, overall_summary)."""

# --- JSON Repair ---

def repair_json(json_str):
    json_str = json_str.strip()
    if json_str.startswith('```'):
        json_str = re.sub(r'^```(?:json)?', '', json_str)
        json_str = re.sub(r'```$', '', json_str)
    json_str = json_str.strip()
    start = json_str.find('{')
    if start == -1: return json_str
    end = json_str.rfind('}')
    if end != -1 and end > start: json_str = json_str[start:end+1]
    else: json_str = json_str[start:]
    
    # Simple balancing
    if json_str.count('"') % 2 != 0: json_str += '"'
    if json_str.count('[') > json_str.count(']'): json_str += ']'
    if json_str.count('{') > json_str.count('}'): json_str += '}'
    return json_str

# --- API Interaction ---

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            header = image_file.read(12)
            image_file.seek(0)
            content = image_file.read()
        mime = 'image/png' if header.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
        return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"
    except: return None

def process_single_image(args, row):
    canonical_id = row['canonical_id']
    image_path_raw = row['absolute_image_path']
    
    # 1. Resolve Local Path
    local_path = convert_to_wsl_path(args.image_root) / image_path_raw if args.image_root else convert_to_wsl_path(image_path_raw)
    out_dir = convert_to_wsl_path(args.output_dir)
    out_path = out_dir / f"{canonical_id}.json"
    
    if out_path.exists(): return {'canonical_id': canonical_id, 'status': 'skipped'}
    if not local_path.exists(): return {'canonical_id': canonical_id, 'status': 'error', 'error': f'Not found: {local_path}'}

    # 2. Resolve R-CNN JSON (Guided Mode)
    rcnn_prompt = None
    if args.rcnn_dir:
        rcnn_root = convert_to_wsl_path(args.rcnn_dir)
        p_exact = rcnn_root / f"{canonical_id}.json"
        p_no_ext = rcnn_root / f"{canonical_id.rsplit('.', 1)[0]}.json"
        p_final = p_exact if p_exact.exists() else (p_no_ext if p_no_ext.exists() else None)
        if p_final:
            try:
                with open(p_final, 'r', encoding='utf-8') as f:
                    rcnn_prompt = get_enhanced_prompt(json.load(f))
            except: pass

    # 3. Select Prompt
    if rcnn_prompt: prompt = rcnn_prompt
    elif args.include_grounding: prompt = get_integrated_prompt()
    else: prompt = create_structured_prompt()

    # 4. API Call
    try:
        uri = encode_image(local_path)
        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": args.max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            try:
                # Attempt to parse
                clean_content = repair_json(content)
                json_content = json.loads(clean_content, strict=False)
                
                # Metadata
                json_content['canonical_id'] = canonical_id
                json_content['model'] = args.model
                json_content['processed_at'] = time.time()
                if rcnn_prompt: json_content['guided_by_rcnn'] = True
                
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2)
                return {'canonical_id': canonical_id, 'status': 'success', 'guided': bool(rcnn_prompt)}
            except Exception as e:
                 return {'canonical_id': canonical_id, 'status': 'error', 'error': f"JSON Error: {str(e)} | Content: {content[:100]}..."}
        else:
            return {'canonical_id': canonical_id, 'status': 'error', 'error': f"API {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {'canonical_id': canonical_id, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Enhanced Batch Zhipu AI Processing')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-root', required=False)
    parser.add_argument('--rcnn-dir', required=False, help='Path to R-CNN JSONs for Guided Mode')
    parser.add_argument('--include-grounding', action='store_true', help='Include bounding box grounding prompt')
    parser.add_argument('--api-key', default=os.environ.get("Z_API_KEY"))
    parser.add_argument('--model', default="glm-4.6v-flash") 
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        print("Error: API Key required.")
        return

    with open(args.manifest, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.limit: rows = rows[:args.limit]
        
    print(f"Loaded {len(rows)} items. Starting {args.workers} threads...")
    
    success = 0
    errors = 0
    skipped = 0
    guided = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {executor.submit(process_single_image, args, row): row for row in rows}
        pbar = tqdm(as_completed(future_to_row), total=len(rows), desc="Processing")
        for future in pbar:
            result = future.result()
            if result['status'] == 'success':
                success += 1
                if result.get('guided'): guided += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                errors += 1
                if errors <= 5: print(f"\n[ERROR] {result['canonical_id']}: {result.get('error')}")
            
            pbar.set_description(f"Succ: {success} | Guided: {guided} | Skip: {skipped} | Err: {errors}")

    print(f"\n--- Complete ---\nSuccess: {success} (Guided: {guided})\nSkipped: {skipped}\nErrors:  {errors}")

if __name__ == "__main__":
    main()
