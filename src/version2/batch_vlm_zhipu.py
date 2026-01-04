#!/usr/bin/env python3
"""
Manifest-driven Zhipu AI (GLM-4V-Flash) Batch Processing
Uses Z.ai API to analyze comic pages. Supports Analysis and Grounding modes.

Usage:
    python src/version2/batch_vlm_zhipu.py \
        --manifest manifests/calibrecomics-extracted_manifest.csv \
        --output-dir E:/Comic_Analysis_Results_v2/zhipu_results \
        --image-root E:/CalibreComics_extracted \
        --api-key YOUR_KEY \
        --workers 3
"""

import os
import argparse
import csv
import json
import base64
import requests
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Prompts ---
def get_prompt(mode):
    if mode == 'grounding':
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
    elif mode == 'analysis':
        return """Analyze this comic page. Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
      "description": "Visual description",
      "speakers": [{"character": "Name", "dialogue": "Text"}],
      "actions": ["Action1"]
    }
  ]
}
"""
    else: # Full / Integrated
        return """Analyze this comic page. Provide structured analysis AND bounding boxes.

REQUIREMENTS:
1. Identify every panel with BOUNDING BOX [xmin, ymin, xmax, ymax] (0-1000).
2. Identify characters, faces, and text bubbles with bounding boxes.
3. Transcribe dialogue.

Return ONLY valid JSON:
{
  "objects": [{"label": "panel|person|face|text", "box_2d": [xmin, ymin, xmax, ymax]}],
  "panels": [{"panel_number": 1, "description": "...", "speakers": [...]}]
}
"""

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            header = image_file.read(12)
            image_file.seek(0)
            content = image_file.read()
            
        if header.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
        elif header.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
        else: mime = 'image/jpeg'

        return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"
    except Exception as e:
        return None

def convert_to_wsl_path(path_str):
    """Convert Windows path to WSL path if running on Linux."""
    if os.name == 'posix' and ':' in path_str:
        # Check for drive letter (e.g. E:\ or E:/)
        drive, rest = path_str.split(':', 1)
        if len(drive) == 1:
            # Convert backslashes to forward slashes
            rest = rest.replace('\\', '/')
            return Path(f"/mnt/{drive.lower()}{rest}")
    return Path(path_str)

def process_single_image(args, row):
    canonical_id = row['canonical_id']
    image_path_raw = row['absolute_image_path']
    local_path = None

    # Resolve Path
    if args.image_root:
        if image_path_raw.startswith('s3://'):
            try:
                relative_path = image_path_raw.split('/', 3)[3]
                # Convert root if needed
                root = convert_to_wsl_path(args.image_root)
                local_path = root / relative_path
            except:
                return {'canonical_id': canonical_id, 'status': 'error', 'error': 'Invalid S3 URI'}
        else:
            root = convert_to_wsl_path(args.image_root)
            local_path = root / image_path_raw
    else:
        local_path = convert_to_wsl_path(image_path_raw)

    # Convert output dir too
    out_dir = convert_to_wsl_path(args.output_dir)
    out_path = out_dir / f"{canonical_id}.json"
    
    # Check if exists (Skip)
    if out_path.exists():
        return {'canonical_id': canonical_id, 'status': 'skipped'}

    if not local_path.exists():
        return {'canonical_id': canonical_id, 'status': 'error', 'error': 'Image not found'}

    # API Call
    try:
        uri = encode_image(local_path)
        if not uri: return {'canonical_id': canonical_id, 'status': 'error', 'error': 'Encoding failed'}
        
        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
        
        prompt = get_prompt(args.mode)
        
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": uri}}] }],
            "max_tokens": 4095,
            "temperature": 0.1 # Low temp for structured JSON
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # Clean Markdown
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            # Parse to ensure valid JSON before saving
            try:
                json_content = json.loads(content)
                # Inject metadata
                json_content['canonical_id'] = canonical_id
                json_content['model'] = args.model
                json_content['processed_at'] = time.time()
                
                # Save
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2)
                
                return {'canonical_id': canonical_id, 'status': 'success'}
            except json.JSONDecodeError:
                 return {'canonical_id': canonical_id, 'status': 'error', 'error': 'Invalid JSON response'}
        else:
            return {'canonical_id': canonical_id, 'status': 'error', 'error': f"API {response.status_code}: {response.text}"}

    except Exception as e:
        return {'canonical_id': canonical_id, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Batch Zhipu AI Processing')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-root', required=False)
    parser.add_argument('--api-key', default=os.environ.get("Z_API_KEY"))
    parser.add_argument('--model', default="glm-4.6v-flash") # or glm-4v-flash
    parser.add_argument('--mode', choices=['analysis', 'grounding', 'full'], default='full')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        print("Error: API Key required via --api-key or Z_API_KEY env var.")
        return

    # Load Manifest
    rows = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if args.limit:
        rows = rows[:args.limit]
        
    print(f"Loaded {len(rows)} items. Starting {args.workers} threads...")
    
    # Processing
    success = 0
    errors = 0
    skipped = 0
    
    pbar = tqdm(as_completed(future_to_row), total=len(rows), desc="Processing")
    for future in pbar:
        result = future.result()
        if result['status'] == 'success':
            success += 1
        elif result['status'] == 'skipped':
            skipped += 1
        else:
            errors += 1
            if errors <= 5:
                print(f"\n[ERROR] {result['canonical_id']}: {result.get('error')}")
        
        pbar.set_description(f"Success: {success} | Skipped: {skipped} | Errors: {errors}")

    print(f"\n--- Complete ---")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")

if __name__ == "__main__":
    main()
