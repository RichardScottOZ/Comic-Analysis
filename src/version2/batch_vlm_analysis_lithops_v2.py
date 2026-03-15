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
      ],
      "faces": [
        {"character": "Name", "box_2d": [ymin, xmin, ymax, xmax]}
      ]
    }
  ]
}

STRICT RULES:
- Coordinates MUST be normalized (0-1000).
- box_2d is [ymin, xmin, ymax, xmax].
- Ensure every panel, text bubble, character, and face has a corresponding box_2d.
"""

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
        if json_str is None:
            return '{}'
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
        
        # 2.5. Fix invalid escape sequences (e.g., \x -> \\x)
        # Valid escapes in JSON: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        def fix_escapes(match):
            esc = match.group(1)
            valid = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}
            if esc in valid or esc == 'u':
                return match.group(0)  # Keep valid escapes
            return '\\\\' + esc  # Escape the backslash
        json_str = re.sub(r'\\([^"\\\/bfnrtu])', fix_escapes, json_str)
        # Also fix \u not followed by exactly 4 hex digits (e.g. \uROSC, \u followed
        # by whitespace, etc.) — Python's json module raises "Invalid \uXXXX escape"
        # for these.  Double-escaping the backslash produces a literal u-sequence that
        # won't be misinterpreted.
        json_str = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', json_str)
        
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

        # 4.5. Fix unescaped double quotes and other inline value errors iteratively.
        #
        # Two sub-cases for "Expecting ',' delimiter":
        #   a) json_str[pos] == '"' → OPENING quote of unescaped inner word → escape it.
        #   b) json_str[pos] is a word char (letter/digit) → string closed prematurely
        #      somewhere before pos → scan backwards (≤200 chars) for the premature
        #      closer and escape it.
        #      Crucially, we do NOT apply the backwards search for ':' delimiter errors —
        #      those at EOF are truncation artefacts; a backwards escape there would
        #      corrupt legitimate key quotes.
        #
        # Also handles:
        #   - "Expecting value": trailing commas and Python None / JS undefined / NaN.
        try:
            import json as _json

            def _last_unescaped_quote(s, before_pos, max_lookback=200):
                """Return index of last unescaped " in s within max_lookback chars of
                before_pos, or -1.  Limiting lookback prevents accidentally escaping
                legitimate structural quotes far from the error site."""
                search_from = max(0, before_pos - max_lookback)
                for k in range(min(before_pos, len(s)) - 1, search_from - 1, -1):
                    if s[k] == '"':
                        bs = 0
                        j = k - 1
                        while j >= 0 and s[j] == '\\':
                            bs += 1
                            j -= 1
                        if bs % 2 == 0:  # unescaped
                            return k
                return -1

            for _ in range(100):
                try:
                    _json.loads(json_str, strict=False)
                    break
                except _json.JSONDecodeError as e:
                    err_msg = str(e)
                    pos = e.pos

                    if "Expecting ',' delimiter" in err_msg:
                        if 0 <= pos < len(json_str) and json_str[pos] == '"':
                            # OPENING quote of an unescaped inner word — escape directly
                            json_str = json_str[:pos] + '\\"' + json_str[pos + 1:]
                        elif 0 <= pos < len(json_str) and json_str[pos] not in ':{}[],\\"':
                            # String closed prematurely; backwards-search for the closer
                            last_q = _last_unescaped_quote(json_str, pos)
                            if last_q >= 0:
                                json_str = json_str[:last_q] + '\\"' + json_str[last_q + 1:]
                            else:
                                break
                        else:
                            break  # Structural char at pos — not an inner-quote issue

                    elif "Expecting property name enclosed in double quotes" in err_msg:
                        # This happens when an unescaped quoted word like "Doc" causes
                        # the outer string to close prematurely, a literal comma follows
                        # in the text (e.g. `"Doc", is speaking`), the parser consumes
                        # it as a key-value separator, and then finds a plain letter
                        # instead of the expected `"key"`.  The fix is the same backwards
                        # search: find the premature closing quote and escape it.
                        if 0 <= pos < len(json_str) and json_str[pos] not in ':{}[],\\"':
                            last_q = _last_unescaped_quote(json_str, pos)
                            if last_q >= 0:
                                json_str = json_str[:last_q] + '\\"' + json_str[last_q + 1:]
                            else:
                                break
                        else:
                            break

                    elif "Expecting ':' delimiter" in err_msg:
                        # Two sub-cases:
                        #  a) json_str[pos] == '"' → opening quote of an inner word
                        #     acting as a spurious key closer — escape directly.
                        #  b) json_str[pos] is a letter/non-structural char AND the error
                        #     is far from EOF → unescaped quoted word in key name, e.g.
                        #     `"character "Lex" Luthor": ...`  → backwards-search safe.
                        #     We guard with (len - pos) > 200 because genuine truncation
                        #     EOF ':' errors land at pos ≈ len(json_str) (key present but
                        #     no colon/value follows), and escaping there would corrupt keys.
                        if 0 <= pos < len(json_str) and json_str[pos] == '"':
                            json_str = json_str[:pos] + '\\"' + json_str[pos + 1:]
                        elif (0 <= pos < len(json_str)
                              and json_str[pos] not in ':{}[],\\"'
                              and (len(json_str) - pos) > 200):
                            last_q = _last_unescaped_quote(json_str, pos)
                            if last_q >= 0:
                                json_str = json_str[:last_q] + '\\"' + json_str[last_q + 1:]
                            else:
                                break
                        else:
                            break

                    elif "Expecting value" in err_msg:
                        # Trailing comma before ] or }
                        before = json_str[:pos].rstrip()
                        if before and before[-1] == ',':
                            json_str = before[:-1] + json_str[pos:]
                        # Python None / JS undefined / NaN written literally
                        elif json_str[pos:pos + 4] == 'None':
                            json_str = json_str[:pos] + 'null' + json_str[pos + 4:]
                        elif json_str[pos:pos + 9] == 'undefined':
                            json_str = json_str[:pos] + 'null' + json_str[pos + 9:]
                        elif json_str[pos:pos + 3] == 'NaN':
                            json_str = json_str[:pos] + 'null' + json_str[pos + 3:]
                        else:
                            break

                    else:
                        break  # Unknown error type — let step 5 handle it
        except Exception:
            pass  # Fall through to truncation-close step

        # 5. Close any open strings and structures in the correct nesting order.
        #
        # The old approach (count '{'s and '['s then append ']'*N + '}'*M) produces
        # wrong output for nested {[...]} truncations — e.g. a truncated panel array
        # would need "}]}" not "]}}" .  A stack-based scan emits close tokens in the
        # exact reverse order they were opened, which is always correct.
        _stk = []
        _in_s = False
        _esc = False
        for _ch in json_str:
            if _esc:
                _esc = False
                continue
            if _ch == '\\':
                _esc = True
                continue
            if _ch == '"':
                _in_s = not _in_s
                continue
            if _in_s:
                continue
            if _ch == '{':
                _stk.append('}')
            elif _ch == '[':
                _stk.append(']')
            elif _ch in ('}', ']') and _stk and _stk[-1] == _ch:
                _stk.pop()
        if _in_s:
            json_str += '"'       # close the unterminated string
        if _stk:
            json_str += ''.join(reversed(_stk))  # close open {[ in correct order
        
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
        
        # We retry on 429 (rate limit) and 5xx (transient server errors).
        # 4xx errors other than 429 are not retried (auth, bad request, etc.).
        for attempt in range(max_retries):
            try:
                api_res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
                
                if api_res.status_code == 429:
                    last_error = "HTTP 429 Rate Limit"
                    time.sleep(15 * (attempt + 1))
                    continue
                elif api_res.status_code >= 500:
                    last_error = f"HTTP {api_res.status_code}: {api_res.text[:200]}"
                    time.sleep(5 * (attempt + 1))  # brief backoff for transient errors
                    continue
                elif api_res.status_code != 200:
                    last_error = f"HTTP {api_res.status_code}: {api_res.text[:500]}"
                    break  # Don't retry 4xx auth/bad-request errors
                    
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
        choice = res_json.get('choices', [{}])[0]
        # Defensively handle API responses where 'message' or 'content' may be null
        message = choice.get('message') or {}
        content = message.get('content') or ''
            
        final_usage = res_json.get('usage', {})
        
        # 4. Parse & Save (NO RETRIES ON FAILURE)
        
        # Strip markdown safely
        clean_json = content.strip()
        if clean_json.startswith('```json'):
            clean_json = clean_json[7:]
        elif clean_json.startswith('```'):
            clean_json = clean_json[3:]
            
        if clean_json.endswith('```'):
            clean_json = clean_json[:-3]
            
        clean_json = clean_json.strip()
        
        # Find JSON start; do NOT use rfind('}') as "Extra data" errors mean
        # there are multiple objects and rfind picks up the last one.
        # inner_repair_json handles proper truncation at the first complete object.
        start = clean_json.find('{')
        if start != -1:
            clean_json = clean_json[start:]

        try:
            out_data = json.loads(clean_json, strict=False)
        except Exception as parse_e:
            # First parse failed — attempt structural repair before giving up
            repaired = inner_repair_json(clean_json)
            try:
                out_data = json.loads(repaired, strict=False)
            except Exception as repair_e:
                raw_snippet = content[:500].replace('\n', '\\n')
                return {
                    'status': 'error',
                    'canonical_id': canonical_id,
                    'error': f"JSON Parse Failure ({str(repair_e)}) | Raw: {raw_snippet}",
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
        
        # Token distribution stats (all results that have usage data)
        completion_tokens = [
            r['usage']['completion_tokens']
            for r in results
            if r.get('usage') and isinstance(r['usage'], dict) and 'completion_tokens' in r['usage']
        ]
        if completion_tokens:
            ct = sorted(completion_tokens)
            n = len(ct)
            truncated = sum(1 for t in ct if t >= 8192)
            p95_idx = min(int(n * 0.95), n - 1)
            logger.info(
                f"Token stats (n={n}): "
                f"min={ct[0]}, "
                f"mean={sum(ct)//n}, "
                f"p95={ct[p95_idx]}, "
                f"max={ct[-1]}, "
                f"truncated(>=8192)={truncated}"
            )
        
        if failed > 0:
            with open('vlm_production_failures.log', 'a', encoding='utf-8') as f:
                for r in results:
                    if r['status'] == 'error':
                        f.write(f"{r['canonical_id']}: {r.get('error')} | Usage: {r.get('usage')}\n")
        
        logger.info(f"Chunk results: Success={success}, Fail={failed}")

if __name__ == "__main__":
    main()
