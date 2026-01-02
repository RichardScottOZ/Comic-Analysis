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
    """Encodes image to base64 with correct MIME type detected from file content."""
    with open(image_path, "rb") as image_file:
        header = image_file.read(12)
        image_file.seek(0)
        content = image_file.read()
        
    # Detect MIME type from magic numbers
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        mime_type = 'image/png'
    elif header.startswith(b'\xff\xd8\xff'):
        mime_type = 'image/jpeg'
    elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        mime_type = 'image/webp'
    else:
        # Fallback to extension if magic number unknown
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
             mime_type = 'image/jpeg' # Final fallback

    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"

def repair_json(json_str):
    """
    Attempts to repair broken JSON strings using aggressive heuristics.
    Handles truncated responses, unclosed quotes, and leading/trailing chatter.
    """
    import re
    json_str = json_str.strip()
    
    # 1. Basic Markdown Cleanup
    if json_str.startswith('```'):
        json_str = re.sub(r'^```(?:json)?', '', json_str)
        json_str = re.sub(r'```$', '', json_str)
    json_str = json_str.strip()

    # 2. Find the start and end of the JSON object
    # This strips special tokens like <|begin_of_box|> or lead-in text
    start = json_str.find('{')
    if start == -1:
        return json_str
    
    end = json_str.rfind('}')
    if end != -1 and end > start:
        json_str = json_str[start:end+1]
    else:
        json_str = json_str[start:]

    # 3. Close unclosed quotes accurately
    def is_balanced_quotes(s):
        count = 0
        escaped = False
        for char in s:
            if char == '\\':
                escaped = not escaped
            elif char == '"' and not escaped:
                count += 1
                escaped = False
            else:
                escaped = False
        return count % 2 == 0

    if not is_balanced_quotes(json_str):
        json_str += '"'

    # 4. Fix missing commas between fields and elements
    json_str = re.sub(r'(")\s*\n?\s*(")', r'\1,\n\2', json_str)
    json_str = re.sub(r'(\})\s*\n?\s*(")', r'\1,\n\2', json_str)
    json_str = re.sub(r'(\])\s*\n?\s*(")', r'\1,\n\2', json_str)

    # 5. Balance Brackets and Braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)

    # 6. Final cleanup
    json_str = json_str.replace("\\'", "'")
    
    return json_str

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

def analyze_comic_page(image_path, model, api_key, temperature=None, timeout=120, rcnn_json_path=None):
    """Sends image to OpenRouter API (or compatible)."""
    try:
        # Determine prompt strategy
        prompt_text = create_structured_prompt()
        
        if rcnn_json_path and os.path.exists(rcnn_json_path):
            try:
                with open(rcnn_json_path, 'r', encoding='utf-8') as f:
                    rcnn_data = json.load(f)
                prompt_text = get_enhanced_prompt(rcnn_data)
            except Exception as e:
                print(f"Warning: Failed to load R-CNN JSON {rcnn_json_path}: {e}")

        image_data_uri = encode_image_to_data_uri(image_path)
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis", # Required by OpenRouter
            "X-Title": "Comic Analysis"
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text}, 
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            "max_tokens": 8192
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=data, 
            timeout=timeout
        )
        
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' not in res_json:
                return {'status': 'error', 'error': f"KeyError: 'choices' missing. Keys found: {list(res_json.keys())} | Full: {str(res_json)[:200]}"}
            
            content = res_json['choices'][0]['message']['content']
            
            # Clean Markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1]
                if '```' in content:
                    content = content.split('```')[0]
            elif '```' in content:
                content = content.split('```')[1]
            
            content = content.strip()
            
            try:
                return {'status': 'success', 'content': json.loads(content, strict=False)}
            except json.JSONDecodeError:
                # Attempt simple repair
                try:
                    repaired = repair_json(content)
                    return {'status': 'success', 'content': json.loads(repaired, strict=False)}
                except json.JSONDecodeError as e:
                     # Log the raw content for debugging (truncated)
                     preview = content[:500].replace('\n', '\\n')
                     return {'status': 'error', 'error': f"JSON Parse Error: {str(e)} | Raw: {preview}"}

        else:
            return {'status': 'error', 'error': f"API Error {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# --- Worker Function ---

def process_single_record(args):
    """Worker function to process one manifest record."""
    record, output_dir, image_root, model, api_key, temperature, timeout, rcnn_dir = args
    
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

    # 3. Resolve R-CNN JSON Path
    rcnn_json_path = None
    if rcnn_dir:
        # Strategy A: Exact mirror of detection script logic
        p_exact = Path(rcnn_dir) / f"{canonical_id}.json"
        
        # Strategy B: If ID has .jpg/.png, try without it
        clean_id = canonical_id
        if clean_id.lower().endswith(('.jpg', '.png', '.jpeg')):
            clean_id = clean_id.rsplit('.', 1)[0]
        p_no_ext = Path(rcnn_dir) / f"{clean_id}.json"
        
        # Strategy C: If manifest used underscores but detections used slashes
        p_slashed = Path(rcnn_dir) / f"{clean_id.replace('_', '/')}.json"

        if p_exact.exists(): rcnn_json_path = p_exact
        elif p_no_ext.exists(): rcnn_json_path = p_no_ext
        elif p_slashed.exists(): rcnn_json_path = p_slashed

    # 4. Call API
    is_guided = rcnn_json_path is not None and rcnn_json_path.exists()
    result = analyze_comic_page(local_path, model, api_key, temperature, timeout, rcnn_json_path)
    
    # 5. Save Result
    if result['status'] == 'success':
        out_data = result['content']
        # Add metadata
        out_data['canonical_id'] = canonical_id
        out_data['model'] = model
        out_data['processed_at'] = time.time()
        if is_guided:
            out_data['guided_by_rcnn'] = True
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
            
        return {'status': 'success', 'canonical_id': canonical_id, 'guided': is_guided}
    else:
        return {'status': 'error', 'canonical_id': canonical_id, 'error': result['error']}

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='Batch VLM Analysis via API (Manifest Driven)')
    parser.add_argument('--manifest', required=True, help='Path to master_manifest.csv')
    parser.add_argument('--output-dir', required=True, help='Root directory for output JSONs')
    parser.add_argument('--image-root', required=False, help='Local root folder (optional). Required if manifest has S3 URIs.')
    parser.add_argument('--rcnn-dir', required=False, help='Root directory for R-CNN JSONs (optional). If provided, enables Guided Mode.')
    parser.add_argument('--model', default='google/gemini-pro-1.5', help='OpenRouter model ID')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"), help='API Key')
    parser.add_argument('--limit', type=int, help='Limit the number of images to process (useful for testing)')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature (omit for model default, 0.0 for deterministic)')
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
        tasks.append((rec, args.output_dir, args.image_root, args.model, args.api_key, args.temperature, args.timeout, args.rcnn_dir))

    # 3. Process
    print(f"Starting processing with {args.workers} workers...")
    
    successful = 0
    guided = 0
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
                    if res.get('guided'):
                        guided += 1
                elif res['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    # Log failure to a separate file so we don't lose track
                    with open('vlm_failures.log', 'a', encoding='utf-8') as log:
                        log.write(f"{res['canonical_id']}: {res.get('error')}\n")
                
                pbar.set_description(f"Succ: {successful} | Guided: {guided} | Fail: {failed}")
                pbar.update(1)

    print("\nProcessing Complete.")
    print(f"Success: {successful} (Guided: {guided})")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed} (See vlm_failures.log)")

if __name__ == "__main__":
    main()
