#!/usr/bin/env python3
"""
Distributed VLM Analysis with Lithops (Serverless)

This script uses Lithops to run comic page analysis in parallel across AWS Lambda workers.
It reads from an S3-based manifest, checks for existing results in S3 to skip,
and sends images to OpenRouter VLMs.

Features:
- Massive parallelization (100+ concurrent workers)
- S3-to-S3 workflow (no local image downloads required on the orchestrator)
- Resume capability (skips existing .json files in S3)
- Robust JSON repair for VLM outputs
"""

import os
import argparse
import csv
import json
import base64
import time
import logging
from pathlib import Path
from datetime import datetime

import boto3
import lithops

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VLM Prompt & Logic (Runs inside Lambda) ---

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

def repair_json(json_str):
    """Attempts to repair broken JSON strings from VLMs."""
    import re
    json_str = json_str.strip()
    start = json_str.find('{')
    if start == -1: return json_str
    json_str = json_str[start:]
    
    # Close unclosed quotes
    quotes = len(re.findall(r'(?<!\\)"', json_str))
    if quotes % 2 != 0: json_str += '"'

    # Balance braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    if open_brackets > close_braces: json_str += ']' * (open_brackets - close_brackets)
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces: json_str += '}' * (open_braces - close_braces)
    
    # Fix missing commas
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    json_str = json_str.replace("\'", "'"")
    return json_str

def process_page_vlm(task_data):
    """
    Worker function executed in AWS Lambda.
    """
    import json
    import base64
    import requests
    import boto3
    import os
    from pathlib import Path
    
    canonical_id = task_data['canonical_id']
    s3_uri = task_data['image_path']
    output_bucket = task_data['output_bucket']
    output_prefix = task_data['output_prefix']
    model = task_data['model']
    api_key = task_data['api_key']
    timeout = task_data.get('timeout', 120)
    
    s3_client = boto3.client('s3')
    
    try:
        # 1. Download image from S3
        if not s3_uri.startswith('s3://'):
            return {'status': 'error', 'canonical_id': canonical_id, 'error': f"Invalid S3 URI: {s3_uri}"}
            
        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        
        # 2. Prepare Data URI
        ext = Path(key).suffix.lower()
        mime_type = 'image/jpeg'
        if ext == '.png': mime_type = 'image/png'
        elif ext == '.webp': mime_type = 'image/webp'
        
        base64_image = base64.b64encode(image_content).decode('utf-8')
        image_data_uri = f"data:{mime_type};base64,{base64_image}"
        
        # 3. Call OpenRouter
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RichardScottOZ/Comic-Analysis",
            "X-Title": "Comic Analysis Lithops"
        }
        
        payload = {
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
        
        api_res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        if api_res.status_code != 200:
            return {'status': 'error', 'canonical_id': canonical_id, 'error': f"API Error {api_res.status_code}: {api_res.text}"}
            
        content = api_res.json()['choices'][0]['message']['content']
        
        # Clean Markdown
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        content = content.strip()
        
        # Parse and potentially repair
        try:
            out_data = json.loads(content, strict=False)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(content)
                out_data = json.loads(repaired, strict=False)
            except:
                return {'status': 'error', 'canonical_id': canonical_id, 'error': f"JSON Parse Failure | Raw: {content[:200]}"}
        
        # 4. Save to S3
        out_data['canonical_id'] = canonical_id
        out_data['model'] = model
        out_data['processed_at'] = time.time()
        
        out_key = f"{output_prefix}/{canonical_id}.json"
        s3_client.put_object(
            Bucket=output_bucket,
            Key=out_key,
            Body=json.dumps(out_data, indent=2),
            ContentType='application/json'
        )
        
        return {'status': 'success', 'canonical_id': canonical_id}
        
    except Exception as e:
        return {'status': 'error', 'canonical_id': canonical_id, 'error': str(e)}

# --- Orchestrator Logic ---

def run_vlm_analysis_lithops(manifest_path, output_bucket, output_prefix, model, workers, batch_size, limit, api_key):
    # 1. Load Manifest
    logger.info(f"Loading manifest: {manifest_path}")
    all_records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
    
    if limit:
        all_records = all_records[:limit]
        logger.info(f"Limited to {limit} records.")

    # 2. FAST Skip check (Check S3 for existing files)
    logger.info(f"Checking S3 for existing results in s3://{output_bucket}/{output_prefix}/...")
    s3_client = boto3.client('s3')
    existing_ids = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=output_bucket, Prefix=output_prefix + '/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # e.g. vlm_results/comic_page_1.json -> comic_page_1
            fname = Path(key).stem
            existing_ids.add(fname)
            
    to_process = [r for r in all_records if r['canonical_id'] not in existing_ids]
    skipped = len(all_records) - len(to_process)
    logger.info(f"Skipped {skipped} existing results. To process: {len(to_process)}")
    
    if not to_process:
        logger.info("Nothing to do!")
        return

    # 3. Process in batches to avoid overwhelming orchestrator or API
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    
    total_success = 0
    total_failed = 0
    
    failure_log = 'vlm_lithops_failures.log'
    
    # Initialize Lithops Executor with minimal runtime
    # Note: Using 128MB (absolute minimum) to maximize cost efficiency during API wait times.
    fexec = lithops.FunctionExecutor(
        backend='aws_lambda', 
        runtime='comic-vlm-lite',
        runtime_memory=128
    )
    
    for i in range(total_batches):
        batch = to_process[i*batch_size : (i+1)*batch_size]
        logger.info(f"Batch {i+1}/{total_batches} - Processing {len(batch)} items...")
        
        task_list = []
        for rec in batch:
            task_list.append({
                'canonical_id': rec['canonical_id'],
                'image_path': rec['absolute_image_path'],
                'output_bucket': output_bucket,
                'output_prefix': output_prefix,
                'model': model,
                'api_key': api_key
            })
            
        # Map
        futures = fexec.map(process_page_vlm, task_list)
        results = fexec.get_result(futures)
        
        # Count
        batch_success = sum(1 for r in results if r['status'] == 'success')
        batch_failed = sum(1 for r in results if r['status'] == 'error')
        
        total_success += batch_success
        total_failed += batch_failed
        
        # Log failures
        if batch_failed > 0:
            with open(failure_log, 'a', encoding='utf-8') as f:
                for r in results:
                    if r['status'] == 'error':
                        f.write(f"{r['canonical_id']}: {r['error']}\n")
        
        logger.info(f"Batch {i+1} results: Success={batch_success}, Fail={batch_failed}")
        logger.info(f"Overall Progress: {total_success + total_failed}/{len(to_process)}")

    logger.info(f"Processing Complete. Success: {total_success}, Failed: {total_failed}")
    if total_failed > 0:
        logger.info(f"Failures logged to {failure_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lithops-based VLM Analysis')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--output-bucket', default='calibrecomics-extracted', help='S3 bucket for output')
    parser.add_argument('--output-prefix', default='vlm_analysis', help='S3 prefix for JSON results')
    parser.add_argument('--model', default='google/gemini-2.0-flash-lite-001', help='Model slug')
    parser.add_argument('--workers', type=int, default=100, help='Max concurrency (not strictly used by map, but for info)')
    parser.add_argument('--batch-size', type=int, default=500, help='Items per Lithops map call')
    parser.add_argument('--limit', type=int, help='Limit total records')
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"), help='OpenRouter API Key')
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("OPENROUTER_API_KEY is required.")
    else:
        run_vlm_analysis_lithops(
            manifest_path=args.manifest,
            output_bucket=args.output_bucket,
            output_prefix=args.output_prefix,
            model=args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit,
            api_key=args.api_key
        )
