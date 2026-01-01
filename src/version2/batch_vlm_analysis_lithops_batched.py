#!/usr/bin/env python3
"""
Batched VLM Analysis with Lithops (Serverless) - _batched version
Based on the proven PaddleOCR batching strategy.

Features:
- Handles millions of pages by processing in chunks
- Memory Escalation: Retries OOM/failed tasks with higher memory
- S3 skip logic (checks existing results)
- Robust JSON repair for VLM outputs (Aggressive Mode)
- Adjustable Temperature
"""

import os
import argparse
import csv
import json
import base64
import time
import logging
import itertools
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
    """
    Attempts to repair broken JSON strings using aggressive heuristics.
    Handles truncated responses, unclosed quotes, and missing delimiters.
    """
    import re
    json_str = json_str.strip()
    
    # 1. Basic Markdown Cleanup
    if json_str.startswith('```'):
        json_str = re.sub(r'^```(?:json)?', '', json_str)
        json_str = re.sub(r'```$', '', json_str)
    json_str = json_str.strip()

    # 2. Find the start of the JSON object
    start = json_str.find('{')
    if start == -1:
        return json_str
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

    # 4. Fix missing commas between fields
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
    
    # Re-check braces after adding brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)

    # 6. Final cleanup
    json_str = json_str.replace("\'", "'")
    
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
    import time
    from pathlib import Path
    
    canonical_id = task_data['canonical_id']
    s3_uri = task_data['image_path']
    output_bucket = task_data['output_bucket']
    output_prefix = task_data['output_prefix']
    model = task_data['model']
    api_key = task_data['api_key']
    timeout = task_data.get('timeout', 120)
    temperature = task_data.get('temperature', 0.0)
    
    s3_client = boto3.client('s3')
    
    try:
        # 1. Download image from S3
        if not s3_uri.startswith('s3://'):
            return {'status': 'error', 'canonical_id': canonical_id, 'error': f"Invalid S3 URI: {s3_uri}"}
            
        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_content = response['Body'].read()
        except Exception as e:
             return {'status': 'error', 'canonical_id': canonical_id, 'error': f"S3 Download Error: {str(e)}"}

        # 2. Prepare Data URI
        ext = Path(key).suffix.lower()
        mime_type = 'image/jpeg'
        if ext == '.png': mime_type = 'image/png'
        elif ext == '.webp': mime_type = 'image/webp'
        elif ext == '.gif': mime_type = 'image/gif'
        
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
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": create_structured_prompt()},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            "max_tokens": 8192
        }
        
        try:
            api_res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            return {'status': 'error', 'canonical_id': canonical_id, 'error': "API Timeout"}
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'canonical_id': canonical_id, 'error': f"API Request Error: {str(e)}"}
            
        if api_res.status_code != 200:
            return {'status': 'error', 'canonical_id': canonical_id, 'error': f"API Error {api_res.status_code}: {api_res.text}"}
            
        try:
            content = api_res.json()['choices'][0]['message']['content']
        except (KeyError, IndexError, json.JSONDecodeError):
             return {'status': 'error', 'canonical_id': canonical_id, 'error': f"Invalid API Response: {api_res.text[:200]}"}

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
        return {'status': 'error', 'canonical_id': canonical_id, 'error': f"Unhandled Worker Exception: {str(e)}"}

# --- Orchestrator Logic ---

def run_vlm_analysis_batched(manifest_path, output_bucket, output_prefix, model, workers, batch_size, limit, api_key, backend='aws_lambda', use_default_runtime=False, temperature=0.0):
    # Memory Escalation Levels (MB)
    MEMORY_LEVELS = [128, 192, 256, 384, 512, 640, 768, 896, 1024]
    if backend == 'localhost':
         MEMORY_LEVELS = [512]

    logger.info(f"Loading manifest: {manifest_path}")
    
    all_records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
    
    if limit:
        all_records = all_records[:limit]
        logger.info(f"Limited to {limit} records.")

    # 2. FAST Skip check
    logger.info(f"Checking S3 for existing results in s3://{output_bucket}/{output_prefix}/...")
    s3_client = boto3.client('s3')
    existing_ids = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    
    prefix = output_prefix.rstrip('/') + '/'
    for page in paginator.paginate(Bucket=output_bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                # Robustly extract ID: remove prefix and .json
                # key: vlm_results/Folder/Image.json -> Folder/Image
                relative_path = key[len(prefix):-5] 
                existing_ids.add(relative_path)
            
    to_process = [r for r in all_records if r['canonical_id'] not in existing_ids]
    skipped = len(all_records) - len(to_process)
    logger.info(f"Skipped {skipped} existing results. To process: {len(to_process)}")
    
    if not to_process:
        logger.info("Nothing to do!")
        return

    # 3. Process in Batches
    total_batches = (len(to_process) + batch_size - 1) // batch_size
    
    total_success = 0
    total_failed = 0
    failure_log = 'vlm_lithops_failures_batched.log'
    
    # Configure Runtime
    runtime = 'comic-vlm-lite'
    if use_default_runtime:
        runtime = None # Lithops picks default
        logger.info("Using default Lithops runtime.")
    elif backend == 'localhost':
        runtime = None
        if workers > 16:
            workers = 8
            logger.warning("Clamping localhost workers to 8")

    logger.info(f"Starting Batched Processing: {total_batches} batches of ~{batch_size} items")
    logger.info(f"Memory Escalation Strategy: {MEMORY_LEVELS} MB")

    for i in range(total_batches):
        batch_records = to_process[i*batch_size : (i+1)*batch_size]
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {i+1}/{total_batches} - Processing {len(batch_records)} items...")
        logger.info(f"{'='*60}")
        
        current_tasks = []
        for rec in batch_records:
            data = {
                'canonical_id': rec['canonical_id'],
                'image_path': rec['absolute_image_path'],
                'output_bucket': output_bucket,
                'output_prefix': output_prefix,
                'model': model,
                'api_key': api_key,
                'temperature': temperature
            }
            current_tasks.append({'task_data': data})

        batch_success_count = 0
        batch_failed_count = 0
        
        for mem_level in MEMORY_LEVELS:
            if not current_tasks:
                break
                
            logger.info(f"  >>> Attempting {len(current_tasks)} tasks with {mem_level}MB memory...")
            
            executor = None
            try:
                executor = lithops.FunctionExecutor(
                    backend=backend, 
                    runtime=runtime,
                    runtime_memory=mem_level,
                    workers=workers
                )
                
                futures = executor.map(process_page_vlm, current_tasks)
                results = executor.get_result(futures)
                
                next_round_tasks = []
                current_round_failures = []
                
                for task_input, res in zip(current_tasks, results):
                    is_success = False
                    is_retryable = False
                    error_msg = "Unknown"

                    if isinstance(res, Exception):
                        error_msg = str(res)
                        if 'Runtime.ExitError' in error_msg or 'MemoryError' in error_msg:
                            is_retryable = True
                        else:
                            is_retryable = True 
                    
                    elif isinstance(res, dict):
                        if res.get('status') == 'success':
                            is_success = True
                        else:
                            error_msg = res.get('error', 'Unknown Error')
                            if 'MemoryError' in error_msg:
                                is_retryable = True
                    else:
                        error_msg = f"Unexpected result type: {type(res)}"
                    
                    if is_success:
                        batch_success_count += 1
                    elif is_retryable:
                        next_round_tasks.append(task_input)
                    else:
                        batch_failed_count += 1
                        cid = task_input['task_data']['canonical_id']
                        logger.error(f"    ✗ {cid} - {error_msg}")
                        current_round_failures.append(f"{cid}: {error_msg}")

                if current_round_failures:
                    with open(failure_log, 'a', encoding='utf-8') as logf:
                        for failure in current_round_failures:
                            logf.write(failure + "\n")

                current_tasks = next_round_tasks
                executor.clean()
                del executor

            except Exception as e:
                logger.error(f"  !!! Executor orchestration failed at {mem_level}MB: {e}")
                if executor:
                     try:
                        executor.clean()
                     except:
                        pass
                continue

        if current_tasks:
            for task in current_tasks:
                batch_failed_count += 1
                cid = task['task_data']['canonical_id']
                msg = f"{cid}: Failed at all memory levels (Max {MEMORY_LEVELS[-1]}MB)"
                logger.error(f"    ✗ {msg}")
                with open(failure_log, 'a', encoding='utf-8') as logf:
                    logf.write(msg + "\n")

        total_success += batch_success_count
        total_failed += batch_failed_count
        
        logger.info(f"Batch {i+1} Summary: Success={batch_success_count}, Fail={batch_failed_count}")
        logger.info(f"Overall Progress: {total_success + total_failed}/{len(to_process)}")

    logger.info(f"\nProcessing Complete. Success: {total_success}, Failed: {total_failed}")
    if total_failed > 0:
        logger.info(f"Failures logged to {failure_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batched VLM Analysis (Lithops)')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--output-bucket', default='calibrecomics-extracted', help='S3 bucket for output')
    parser.add_argument('--output-prefix', default='vlm_analysis', help='S3 prefix for JSON results')
    parser.add_argument('--model', default='google/gemini-2.0-flash-lite-001', help='Model slug')
    parser.add_argument('--workers', type=int, default=100, help='Max concurrency per batch')
    parser.add_argument('--batch-size', type=int, default=500, help='Items per batch')
    parser.add_argument('--limit', type=int, help='Limit total records')
    parser.add_argument('--api-key', default=os.environ.get("OPENROUTER_API_KEY"), help='OpenRouter API Key')
    parser.add_argument('--backend', default='aws_lambda', help='Lithops backend')
    parser.add_argument('--use-default-runtime', action='store_true', help='Use default Lithops runtime')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0.0 for deterministic)')
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("OPENROUTER_API_KEY is required.")
    else:
        run_vlm_analysis_batched(
            manifest_path=args.manifest,
            output_bucket=args.output_bucket,
            output_prefix=args.output_prefix,
            model=args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit,
            api_key=args.api_key,
            backend=args.backend,
            use_default_runtime=args.use_default_runtime,
            temperature=args.temperature
        )
