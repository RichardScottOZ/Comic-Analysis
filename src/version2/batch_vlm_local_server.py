#!/usr/bin/env python3
"""
Batch VLM Analysis using a Local Inference Server (OpenAI Compatible)
Compatible with: Llama.cpp --server, Ollama, vLLM, LM Studio.

Usage:
    python src/version2/batch_vlm_local_server.py \
        --manifest manifests/nowhereman_manifest.csv \
        --output-dir E:/vlm_recycling_staging \
        --image-root E:/ \
        --base-url http://localhost:8080/v1
"""

import os
import argparse
import csv
import json
import base64
import time
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

# Disable decompression bomb limits
Image.MAX_IMAGE_PIXELS = None

def get_detailed_prompt():
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

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            header = image_file.read(12)
            image_file.seek(0)
            content = image_file.read()
        mime = 'image/png' if header.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
        return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"
    except Exception as e:
        print(f"[ENCODE ERROR] {image_path}: {e}")
        return None

def process_image(args, row):
    cid = row['canonical_id']
    path_raw = row['absolute_image_path']
    
    # 1. Resolve Path
    if args.image_root:
        if path_raw.startswith('s3://'):
            try:
                relative_path = path_raw.split('/', 3)[3]
                local_path = Path(args.image_root) / relative_path.replace('/', os.sep)
            except:
                return {'canonical_id': canonical_id, 'status': 'error', 'error': 'Invalid S3 URI'}
        else:
            local_path = Path(args.image_root) / path_raw
    else:
        local_path = Path(path_raw)

    out_path = Path(args.output_dir) / f"{cid}.json"
    
    if out_path.exists():
        return {'status': 'skipped'}

    if not local_path.exists():
        return {'status': 'error', 'error': f'Image not found: {local_path}'}

    # 2. Call Local Server
    try:
        uri = encode_image(local_path)
        if not uri: 
            return {'status': 'error', 'error': f'Encoding failed for {local_path}'}
        
        headers = {"Content-Type": "application/json"}
        if args.api_key: headers["Authorization"] = f"Bearer {args.api_key}"
        
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": get_detailed_prompt()},
                        {"type": "image_url", "image_url": {"url": uri}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        url = f"{args.base_url.rstrip('/')}/chat/completions"
        res = requests.post(url, json=payload, timeout=args.timeout)
        
        if res.status_code == 200:
            content = res.json()['choices'][0]['message']['content']
            
            if '```json' in content: content = content.split('```json')[1].split('```')[0]
            elif '```' in content: content = content.split('```')[1].split('```')[0]
            
            data = json.loads(content.strip())
            data['canonical_id'] = cid
            data['model'] = f"local-{args.model}"
            data['processed_at'] = time.time()
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return {'status': 'success'}
        else:
            print(f"\n[API ERROR] {cid}")
            print(f"  URL: {url}")
            print(f"  Status: {res.status_code}")
            print(f"  Response: {res.text[:200]}")
            return {'status': 'error', 'error': f"HTTP {res.status_code}"}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-root', default=None)
    parser.add_argument('--base-url', default="http://localhost:8080/v1")
    parser.add_argument('--api-key', default="sk-no-key-required")
    parser.add_argument('--model', default="glm-4v-9b") 
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    rows = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    print(f"Starting batch with {len(rows)} items against {args.base_url}...")
    
    success, errors, skipped = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, args, row): row for row in rows}
        pbar = tqdm(as_completed(futures), total=len(rows))
        for f in pbar:
            res = f.result()
            if res['status'] == 'success': success += 1
            elif res['status'] == 'skipped': skipped += 1
            else: 
                errors += 1
            pbar.set_description(f"Succ: {success} | Skip: {skipped} | Err: {errors}")

if __name__ == "__main__":
    main()