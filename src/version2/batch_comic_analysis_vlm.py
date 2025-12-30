#!/usr/bin/env python3
"""
Manifest-driven VLM Analysis (API-based)

This script processes comic pages using the master manifest CSV and sends them to an external VLM API 
(e.g., Google Gemini via OpenRouter or Vertex AI) using your established prompt structure.

It is designed to reconcile your messy existing data by:
1. Reading the canonical manifest.
2. Checking if a result already exists.
3. If not, mapping the S3 path to your local E: drive.
4. Sending the image to the API.
5. Saving the result as {canonical_id}.json.

Usage:
    python src/version2/batch_comic_analysis_vlm.py \
        --manifest manifests/calibrecomics-extracted_manifest.csv \
        --output-dir E:/CalibreComics_analysis/vlm \
        --image-root E:/CalibreComics_extracted \
        --model google/gemini-pro-vision \
        --workers 8
"""

import os
import argparse
import csv
import json
import base64
import time
import requests
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration & Prompt ---

def create_structured_prompt():
    """Your standard structured prompt for comic analysis."""
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

def encode_image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def repair_json(json_str):
    """
    Attempts to repair broken JSON strings using simple heuristics.
    Useful for models that truncate output or mess up formatting.
    """
    json_str = json_str.strip()
    
    # 1. Try to find the first '{' and the last '}'
    start = json_str.find('{')
    end = json_str.rfind('}')
    
    if start != -1 and end != -1:
        json_str = json_str[start:end+1]
    elif start != -1:
        # If we have a start but no end, it's likely truncated.
        # We can try to close it blindly (risky but better than nothing for some cases)
        json_str = json_str[start:] + '}' 
    
    return json_str

def analyze_comic_page(image_path, model, api_key, timeout=120):
    """Sends image to OpenRouter API (or compatible)."""
    try:
        image_data_uri = encode_image_to_data_uri(image_path)
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/comicanalysis", # Required by OpenRouter
            "X-Title": "Comic Analysis"
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": create_structured_prompt()}, 
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            "max_tokens": 4000
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=data, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            
            # Clean Markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1]
                if '```' in content:
                    content = content.split('```')[0]
            elif '```' in content:
                content = content.split('```')[1]
            
            content = content.strip()
            
            try:
                return {'status': 'success', 'content': json.loads(content)}
            except json.JSONDecodeError:
                # Attempt simple repair
                try:
                    repaired = repair_json(content)
                    return {'status': 'success', 'content': json.loads(repaired)}
                except json.JSONDecodeError as e:
                     return {'status': 'error', 'error': f"JSON Parse Error: {str(e)}"}

        else:
            return {'status': 'error', 'error': f"API Error {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# --- Worker Function ---

def process_single_record(args):
    """Worker function to process one manifest record."""
    record, output_dir, image_root, model, api_key, timeout = args
    
    canonical_id = record['canonical_id']
    path_raw = record['absolute_image_path']
    
    # 1. Check if output exists (Resume capability)
    output_path = Path(output_dir) / f"{canonical_id}.json"
    if output_path.exists():
        return {'status': 'skipped', 'canonical_id': canonical_id}
    
    # 2. Resolve Local Path
    local_path = None
    if image_root:
        if path_raw.startswith('s3://'):
            try:
                # s3://bucket/path/to/file -> image_root/path/to/file
                relative_path = path_raw.split('/', 3)[3]
                local_path = Path(image_root) / relative_path
            except IndexError:
                return {'status': 'error', 'canonical_id': canonical_id, 'error': f"Invalid S3 URI: {path_raw}"}
        else:
            local_path = Path(image_root) / path_raw
    else:
        # No root provided, assume path is already local
        if path_raw.startswith('s3://'):
             return {'status': 'error', 'canonical_id': canonical_id, 'error': "Cannot process S3 URI without --image-root"}
        local_path = Path(path_raw)

    if not local_path.exists():
        return {'status': 'error', 'canonical_id': canonical_id, 'error': f"File not found: {local_path}"}

    # 3. Call API
    result = analyze_comic_page(local_path, model, api_key, timeout)
    
    # 4. Save Result
    if result['status'] == 'success':
        out_data = result['content']
        # Add metadata
        out_data['canonical_id'] = canonical_id
        out_data['model'] = model
        out_data['processed_at'] = time.time()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        return {'status': 'success', 'canonical_id': canonical_id}
    else:
        return {'status': 'error', 'canonical_id': canonical_id, 'error': result['error']}

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Batch VLM Analysis via API (Manifest Driven)')
    parser.add_argument('--manifest', required=True, help='Path to master_manifest.csv')
    parser.add_argument('--output-dir', required=True, help='Root directory for output JSONs')
    parser.add_argument('--image-root', required=False, help='Local root folder (optional). Required if manifest has S3 URIs.')
    parser.add_argument('--model', default='google/gemini-pro-1.5', help='OpenRouter model ID')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"), help='API Key')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process (useful for testing)')
    parser.add_argument('--timeout', type=int, default=120, help='API Timeout in seconds')
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key or OPENROUTER_API_KEY env var required.")
        return

    # 1. Load Manifest
    print(f"Loading manifest: {args.manifest}")
    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    print(f"Loaded {len(records)} records.")

    # Apply limit if specified
    if args.limit:
        records = records[:args.limit]
        print(f"Limited to {len(records)} records for testing.")

    # 2. Prepare Tasks
    tasks = []
    for rec in records:
        tasks.append((rec, args.output_dir, args.image_root, args.model, args.api_key, args.timeout))

    # 3. Process
    print(f"Starting processing with {args.workers} workers...")
    
    successful = 0
    skipped = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_record, t) for t in tasks]
        
        # Monitor
        with tqdm(total=len(futures), desc="VLM Analysis") as pbar:
            for future in as_completed(futures):
                res = future.result()
                
                if res['status'] == 'success':
                    successful += 1
                elif res['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    # Log failure to a separate file so we don't lose track
                    with open('vlm_failures.log', 'a') as log:
                        log.write(f"{res['canonical_id']}: {res.get('error')}\n")
                
                pbar.set_description(f"Success: {successful} | Skip: {skipped} | Fail: {failed}")
                pbar.update(1)

    print("\nProcessing Complete.")
    print(f"Success: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed} (See vlm_failures.log)")

if __name__ == "__main__":
    main()
