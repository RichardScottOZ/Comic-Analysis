# benchmarks/detections/openrouter/batch_comic_analysis_multi_v2.py

import argparse
import multiprocessing as mp
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import requests
import json
import base64
from tqdm import tqdm
import os
from datetime import datetime

# --- API and Prompt Functions (largely unchanged) ---

def encode_image_to_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def create_structured_prompt():
    """Create a structured prompt for comic analysis."""
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

def analyze_comic_page(image_path, model, api_key, timeout):
    try:
        image_data_uri = encode_image_to_data_uri(image_path)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": create_structured_prompt()}, {"type": "image_url", "image_url": {"url": image_data_uri}}]}],
            "max_tokens": 2000
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=timeout)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # Basic JSON cleaning
            json_match = content.strip().lstrip('```json').rstrip('```')
            return {'status': 'success', 'content': json.loads(json_match)}
        else:
            return {'status': 'error', 'error': f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# --- Worker Function ---
def process_single_record(args):
    """Processes a single record from the master manifest."""
    record, output_dir, model, api_key, timeout = args
    canonical_id = record['canonical_id']
    absolute_image_path = record['absolute_image_path']

    # Construct the output path using the canonical_id
    output_path = Path(output_dir) / f"{canonical_id}.json"
    
    if output_path.exists():
        return {'status': 'skipped', 'canonical_id': canonical_id}

    if not os.path.exists(absolute_image_path):
        return {'status': 'error', 'canonical_id': canonical_id, 'error': 'Image file not found'}

    result = analyze_comic_page(absolute_image_path, model, api_key, timeout)

    if result['status'] == 'success':
        # Add the canonical_id to the JSON content before saving
        vlm_data = result['content']
        vlm_data['canonical_id'] = canonical_id
        vlm_data['source_image_path'] = absolute_image_path

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vlm_data, f, indent=2)
        return {'status': 'success', 'canonical_id': canonical_id}
    else:
        return {'status': 'error', 'canonical_id': canonical_id, 'error': result['error']}

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Manifest-driven batch comic analysis.')
    parser.add_argument('--manifest_file', type=str, required=True, help='Path to the master_manifest.csv file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for VLM analysis JSON files.')
    parser.add_argument('--model', type=str, default="google/gemini-pro-vision", help='OpenRouter model to use.')
    parser.add_argument('--api_key', type=str, default=os.environ.get("OPENROUTER_API_KEY"), help='OpenRouter API key.')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of worker processes.')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout for API requests in seconds.')
    args = parser.parse_args()

    if not args.api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set and --api_key argument not provided.")
        return

    # 1. Load the manifest
    print(f"Loading manifest from: {args.manifest_file}")
    try:
        with open(args.manifest_file, 'r', encoding='utf-8') as f:
            records = list(csv.DictReader(f))
    except Exception as e:
        print(f"Fatal: Could not load manifest file: {e}")
        return
    print(f"Loaded {len(records)} records from manifest.")

    # 2. Prepare arguments for multiprocessing
    process_args = [(record, args.output_dir, args.model, args.api_key, args.timeout) for record in records]

    # 3. Process in parallel
    print(f"Processing {len(process_args)} images with {args.max_workers} workers...")
    successful = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_record, p_args) for p_args in process_args]
        
        with tqdm(total=len(futures), desc="Analyzing Images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    successful += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    # Use tqdm.write to print errors without disturbing the progress bar
                    tqdm.write(f"\nFailed to process {result['canonical_id']}: {result['error']}")
                
                pbar.set_description(
                    f"Success: {successful}, Skipped: {skipped}, Failed: {failed}"
                )
                pbar.update(1)

    print("\n--- VLM Analysis Complete ---")
    print(f"✅ Successful: {successful}")
    print(f"⏭️ Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()
