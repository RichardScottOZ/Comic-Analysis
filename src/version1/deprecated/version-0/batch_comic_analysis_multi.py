#!/usr/bin/env python3
"""
Batch comic analysis script with multiprocessing support.
Uses the multiprocessing strategy from comictest4.py for parallel processing.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import base64
import re
from openai import OpenAI

def encode_image_to_data_uri(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_str}"

def create_structured_prompt():
    """Create a prompt that requests structured JSON output."""
    return """Please analyze this comic page and provide a detailed structured analysis. 

Split the page into individual panels and for each panel provide:
- A descriptive caption
- Detailed description of what's happening
- Any dialogue or speech bubbles with speaker identification
- Key visual elements and actions

Please format your response as a JSON object with the following structure:

{
  "overall_summary": "Brief description of the entire comic page",
  "panels": [
    {
      "panel_number": 1,
      "caption": "Descriptive caption for the panel",
      "description": "Detailed description of what's happening in the panel",
      "speakers": [
        {
          "character": "Character description or name",
          "dialogue": "What they're saying",
          "speech_type": "dialogue/thought/narration/sound_effect"
        }
      ],
      "key_elements": ["list", "of", "key", "visual", "elements"],
      "actions": ["list", "of", "key", "actions", "happening"]
    }
  ],
  "summary": {
    "characters": ["list", "of", "main", "characters"],
    "setting": "Description of the setting/environment",
    "plot": "Brief summary of the plot/story",
    "dialogue": ["list", "of", "key", "dialogue", "lines"]
  }
}

Focus on:
1. Identifying each distinct panel clearly
2. Capturing all dialogue and who's speaking
3. Describing visual elements, actions, and emotions
4. Maintaining narrative flow between panels
5. Noting any sound effects, thought bubbles, or narrative text

Please ensure your response is valid JSON format."""

def analyze_comic_page(image_path, model="qwen/qwen2.5-vl-72b-instruct:free", temperature=0.1, top_p=1.0, api_key=None):
    """Analyze a comic page and return structured JSON output."""
    
    if api_key is None:
        api_key = "bananasplitsapikey"
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # Encode the image
    image_data_uri = encode_image_to_data_uri(image_path)
    
    # Create the completion request
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "",
            "X-Title": "Comic Analysis",
        },
        extra_body={},
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": create_structured_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ],
            }
        ],
    )
    
    return completion.choices[0].message.content

def save_json_output(content, output_path):
    """Save the JSON output to a file."""
    try:
        # Try to parse the content as JSON directly
        json_data = json.loads(content)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return True
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from the response
        try:
            # Look for JSON block between ```json and ``` markers
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                json_data = json.loads(json_content)
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                # Try to find JSON object without code block markers
                # Look for the first { and last }
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_content = content[start:end+1]
                    json_data = json.loads(json_content)
                    
                    # Save to file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    return True
                else:
                    raise ValueError("No JSON content found")
                    
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return False

def process_single_image(args):
    """Process a single image - designed for multiprocessing."""
    image_path, output_path, model, temperature, top_p, api_key = args
    
    try:
        # Analyze the comic page
        result = analyze_comic_page(str(image_path), model, temperature, top_p, api_key)
        
        # Try to save as JSON
        if save_json_output(result, output_path):
            return {
                'status': 'success',
                'image_path': str(image_path),
                'output_path': str(output_path)
            }
        else:
            return {
                'status': 'json_parse_error',
                'image_path': str(image_path),
                'output_path': str(output_path),
                'raw_content': result
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'image_path': str(image_path),
            'output_path': str(output_path),
            'error': str(e)
        }

def find_image_files(root_dir):
    """Find all image files in the directory structure."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory not found: {root_dir}")
        return []
    
    print(f"Scanning directory: {root_dir}")
    
    # Walk through all subdirectories
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            print(f"Processing subdirectory: {subdir.name}")
            for image_file in subdir.glob("*.jpg"):  # Focus on jpg files based on the structure
                if image_file.is_file():
                    image_files.append(image_file)

            for image_file in subdir.glob("*.jpeg"):  # Focus on jpg files based on the structure
                if image_file.is_file():
                    image_files.append(image_file)

            for image_file in subdir.glob("*.png"):  # Focus on jpg files based on the structure
                if image_file.is_file():
                    image_files.append(image_file)

    print(f"Found {len(image_files)} image files")
    return image_files

def main():
    parser = argparse.ArgumentParser(description='Batch process comic images with multiprocessing analysis')
    parser.add_argument('--input-dir', type=str, 
                       default=r'C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\2000ad\images',
                       help='Root directory containing comic images')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmarks/detections/openrouter/analysis_results_multi',
                       help='Directory to save analysis results')
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-vl-32b-instruct:free',
                       help='Model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for generation (0.0-1.0)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip images that already have analysis results')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--start-from', type=str, default=None,
                       help='Start processing from a specific image path')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenRouter API key (default: uses environment variable or default key)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = find_image_files(args.input_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    # Filter by start-from if specified
    if args.start_from:
        start_path = Path(args.start_from)
        try:
            start_index = next(i for i, img in enumerate(image_files) if img == start_path)
            image_files = image_files[start_index:]
            print(f"Starting from image: {start_path}")
        except StopIteration:
            print(f"Start image not found: {args.start_from}")
            return
    
    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Limited to {args.max_images} images for testing")
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Prepare arguments for multiprocessing
    process_args = []
    skipped_count = 0
    
    for image_path in image_files:
        # Create output filename based on the image path
        try:
            relative_path = image_path.relative_to(Path(args.input_dir))
        except ValueError:
            # Fallback: use just the filename if relative path fails
            relative_path = Path(image_path.name)
        
        output_filename = f"{relative_path.parent}_{relative_path.stem}.json"
        output_path = output_dir / output_filename
        
        # Skip if output already exists and skip_existing is True
        if args.skip_existing and output_path.exists():
            skipped_count += 1
            continue
        
        process_args.append((
            image_path,
            output_path,
            args.model,
            args.temperature,
            args.top_p,
            args.api_key
        ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing files")
    
    if not process_args:
        print("No images to process")
        return
    
    print(f"Processing {len(process_args)} images")
    
    # Set number of workers
    max_workers = args.max_workers or min(mp.cpu_count(), len(process_args))
    print(f"Using {max_workers} worker processes")
    
    # Process images in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image, args): args for args in process_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(process_args), desc="Processing images") as pbar:
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                # Update progress bar description with current status
                success_count = sum(1 for r in results if r['status'] == 'success')
                error_count = sum(1 for r in results if r['status'] == 'error')
                parse_error_count = sum(1 for r in results if r['status'] == 'json_parse_error')
                pbar.set_description(f"Success: {success_count}, Errors: {error_count}, Parse Errors: {parse_error_count}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    parse_error_count = sum(1 for r in results if r['status'] == 'json_parse_error')
    
    print(f"\n=== Processing Summary ===")
    print(f"Total images: {len(process_args)}")
    print(f"Successful: {success_count}")
    print(f"JSON parse errors: {parse_error_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(process_args):.2f} seconds")
    
    # Print errors if any
    if error_count > 0:
        print(f"\n=== Errors ===")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['image_path']}: {result['error']}")
    
    if parse_error_count > 0:
        print(f"\n=== JSON Parse Errors ===")
        for result in results:
            if result['status'] == 'json_parse_error':
                print(f"  {result['image_path']}: JSON parsing failed")

if __name__ == "__main__":
    main() 