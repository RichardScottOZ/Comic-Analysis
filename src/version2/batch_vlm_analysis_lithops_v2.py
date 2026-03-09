#!/usr/bin/env python3
"""
Production VLM Analysis Orchestrator (Lithops v2)

Features:
- Gemini 2.5 Flash Lite default, with 3.1 option.
- Manifest-driven S3-to-S3 processing.
- Robust skip logic (checks S3 before launching).
- Rate-limit aware (exponential backoff inside worker).
- Records token usage and detailed failures.
"""

import os
import argparse
import csv
import json
import base64
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import boto3
import lithops
from tqdm import tqdm

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence urllib3 connection pool warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.HTTPWarning)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# --- Path Resolution ---

def resolve_s3_uri(local_path, bucket_name="calibrecomics-extracted"):
    """Converts local E:\ paths to S3 URIs."""
    path_str = str(local_path).replace('\\', '/')
    
    if 'amazon' in path_str:
        rel_parts = path_str.split('amazon/')[-1]
        return f"s3://{bucket_name}/CalibreComics_extracted/amazon/{rel_parts}"
    elif 'CalibreComics_extracted_20251107' in path_str:
        rel_parts = path_str.split('CalibreComics_extracted_20251107/')[-1]
        return f"s3://{bucket_name}/CalibreComics_extracted_20251107/{rel_parts}"
    elif 'CalibreComics_extracted' in path_str:
        rel_parts = path_str.split('CalibreComics_extracted/')[-1]
        return f"s3://{bucket_name}/CalibreComics_extracted/{rel_parts}"
    elif 'NeonIchiban' in path_str:
        rel_parts = path_str.split('NeonIchiban/')[-1]
        return f"s3://{bucket_name}/NeonIchiban/{rel_parts}"
        
    return path_str

# --- VLM Logic (Runs inside Lambda) ---

def get_integrated_prompt():
    """Detailed prompt for grounding and narrative."""
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
      ]
    }
  ]
}"""

def process_page_vlm(task_data):
    """
    Worker function executed in AWS Lambda.
    """
    import json
    import base64
    import requests
    import boto3
    import time
    from pathlib import Path
    
    # Internal JSON repair
    def inner_repair_json(json_str):
        import re
        json_str = json_str.strip()
        
        # 1. Remove markdown code fences
        if json_str.startswith('```'):
            lines = json_str.split('\n')
            if lines[0].startswith('```'): lines = lines[1:]
            if lines and lines[-1].startswith('```'): lines = lines[:-1]
            json_str = '\n'.join(lines).strip()
        
        # 2. Find the first { and remove anything before it
        start = json_str.find('{')
        if start != -1:
            json_str = json_str[start:]
        
        # 3. Fix missing commas between properties (common issue)
        # Pattern: "key": value"next_key" -> "key": value, "next_key"
        # This handles: }" -> }, "  and  ]" -> ], "  and  number" -> number, "
        json_str = re.sub(r'(\d)"(\s*")', r'\1, \2', json_str)  # After number
        json_str = re.sub(r'(\])"(\s*")', r'\1, \2', json_str)  # After ]
        json_str = re.sub(r'(})"(\s*")', r'\1, \2', json_str)  # After }
        json_str = re.sub(r'(true)"(\s*")', r'\1, \2', json_str)  # After true
        json_str = re.sub(r'(false)"(\s*")', r'\1, \2', json_str)  # After false
        json_str = re.sub(r'(null)"(\s*")', r'\1, \2', json_str)  # After null
        
        # 4. Fix missing commas after string values before next key
        # Pattern: "value"\n"key" -> "value",\n"key"
        json_str = re.sub(r'(")("\s*\n\s*")', r'\1,\2', json_str)
        
        # 5. Balance quotes, braces, brackets
        if json_str.count('"') % 2 != 0:
            json_str += '"'
        
        ob = json_str.count('{')
        cb = json_str.count('}')
        ok = json_str.count('[')
        ck = json_str.count(']')
        
        if ok > ck: json_str += ']' * (ok - ck)
        if ob > cb: json_str += '}' * (ob - cb)
        
        # 6. Handle "Extra data" - truncate after first complete JSON object
        # Find where the outermost object closes
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(json_str):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # Found end of first complete object
                        return json_str[:i+1]
        
        return json_str

    canonical_id = task_data['canonical_id']
    s3_uri = task_data['image_path']
    output_bucket = task_data['output_bucket']
    output_prefix = task_data['output_prefix']
    model = task_data['model']
    api_key = task_data['api_key']
    timeout = task_data.get('timeout', 180)
    
    s3_client = boto3.client('s3')
    
    try:
        # 1. Fetch from S3
        parts = s3_uri[5:].split('/', 1)
        bucket, key = parts[0], parts[1]
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        
        # 2. Encode
        ext = Path(key).suffix.lower()
        mime = 'image/png' if ext == '.png' else 'image/jpeg'
        base64_image = base64.b64encode(image_content).decode('utf-8')
        data_uri = f"data:{mime};base64,{base64_image}"
        
        # 3. Call API with Retries for BOTH network AND JSON parse errors
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "Comic Analysis"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": get_integrated_prompt()}, {"type": "image_url", "image_url": {"url": data_uri}}]}],
            "max_tokens": 8192,
            "response_format": {"type": "json_object"}
        }
        
        max_retries = 3
        api_res = None
        last_error = "Unknown Error"
        
        # We only retry for 429 Rate Limits, NOT for JSON parse errors to save money.
        for attempt in range(max_retries):
            try:
                api_res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                if api_res.status_code == 429:
                    last_error = "HTTP 429 Rate Limit"
                    time.sleep(15 * (attempt + 1))
                    continue
                elif api_res.status_code != 200:
                    last_error = f"HTTP {api_res.status_code}: {api_res.text[:500]}"
                    break # Don't retry auth errors
                    
                # Success HTTP
                last_error = None
                break

            except Exception as e:
                last_error = f"Request Exception: {str(e)}"
                time.sleep(5)
                continue

        if last_error:
            return {
                'status': 'error', 
                'canonical_id': canonical_id, 
                'error': f"API Failure: {last_error}"
            }

        res_json = api_res.json()
        content = res_json['choices'][0]['message']['content']
        final_usage = res_json.get('usage', {})
        
        # 4. Parse & Save (NO RETRIES ON FAILURE)
        repaired = inner_repair_json(content)
        try:
            out_data = json.loads(repaired, strict=False)
        except Exception as parse_e:
            raw_snippet = content[:500].replace('\n', '\\n')
            return {
                'status': 'error', 
                'canonical_id': canonical_id, 
                'error': f"JSON Parse Failure ({str(parse_e)}) | Raw: {raw_snippet}", 
                'usage': final_usage,
                'raw_content': content
            }
            
        # Save to S3
        out_data['canonical_id'] = canonical_id
        out_data['model'] = model
        out_data['usage'] = final_usage
        out_data['processed_at'] = time.time()
        
        s3_client.put_object(
            Bucket=output_bucket, Key=f"{output_prefix}/{canonical_id}.json",
            Body=json.dumps(out_data, indent=2), ContentType='application/json'
        )
        return {'status': 'success', 'canonical_id': canonical_id, 'usage': final_usage}
        
    except Exception as e:
        return {'status': 'error', 'canonical_id': canonical_id, 'error': str(e)}

# --- Orchestrator ---

def main():
    parser = argparse.ArgumentParser(description='Production Lithops VLM')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-bucket', default='calibrecomics-extracted')
    parser.add_argument('--output-prefix', default='vlm_analysis_production')
    parser.add_argument('--model', default='google/gemini-2.5-flash-lite')
    parser.add_argument('--workers', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument('--config', default='~/.lithops/config_comic_vlm_v2')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, help="Limit total records to process")
    args = parser.parse_args()

    logger.info(f"Checking S3 for existing progress...")
    s3 = boto3.client('s3')
    existing = set()
    
    if not args.dry_run:
        paginator = s3.get_paginator('list_objects_v2')
        prefix = args.output_prefix.rstrip('/')+'/'
        for page in paginator.paginate(Bucket=args.output_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    existing.add(obj['Key'][len(prefix):-5])
    
    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['canonical_id'] not in existing:
                records.append(row)
    
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limited to {args.limit} records.")
    
    if args.dry_run:
        print("\n=== DRY RUN: Path Mapping Audit ===")
        for i, r in enumerate(records[:5]):
            s3_in = resolve_s3_uri(r['absolute_image_path'], args.output_bucket)
            out_key = f"{args.output_prefix.rstrip('/')}/{r['canonical_id']}.json"
            print(f"Record {i+1}:\n  S3 IN:  {s3_in}\n  S3 OUT: s3://{args.output_bucket}/{out_key}\n")
        return

    logger.info(f"To process: {len(records)} | Skipped: {len(existing)}")
    if not records: return

    fexec = lithops.FunctionExecutor(config_file=os.path.expanduser(args.config), workers=args.workers)
    
    for i in range(0, len(records), args.batch_size):
        chunk = records[i:i+args.batch_size]
        logger.info(f"Launching chunk {i//args.batch_size + 1}...")
        
        tasks = [{'task_data': {
            'canonical_id': r['canonical_id'], 
            'image_path': resolve_s3_uri(r['absolute_image_path'], args.output_bucket),
            'output_bucket': args.output_bucket, 
            'output_prefix': args.output_prefix,
            'model': args.model, 
            'api_key': args.api_key
        }} for r in chunk]
        
        futures = fexec.map(process_page_vlm, tasks)
        results = fexec.get_result(futures)
        
        success = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        if failed > 0:
            with open('vlm_production_failures.log', 'a', encoding='utf-8') as f:
                for r in results:
                    if r['status'] == 'error':
                        f.write(f"{r['canonical_id']}: {r.get('error')} | Usage: {r.get('usage')}\n")
        
        logger.info(f"Chunk results: Success={success}, Fail={failed}")

if __name__ == "__main__":
    main()
