#!/usr/bin/env python3
"""
Multiprocessing batch comic analysis using OpenRouter API.
Processes multiple comic images in parallel and saves structured JSON output.
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import requests
import json
import base64
import time
from tqdm import tqdm
import os
from datetime import datetime

def encode_image_to_data_uri(image_path):
    """Encode image to data URI for API request."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

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

def count_existing_results(output_dir):
    """Count existing JSON result files in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    
    json_files = list(output_path.glob("*.json"))
    return len(json_files)

def analyze_comic_page(image_path, model="qwen/qwen2.5-vl-72b-instruct:free", temperature=0.1, top_p=1.0, api_key=None, debug=False):
    """Analyze a single comic page using OpenRouter API."""
    if api_key is None:
        api_key = "bananasplitsapikey"
    
    try:
        # Encode image
        image_data_uri = encode_image_to_data_uri(image_path)
        
        # Prepare request
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": create_structured_prompt()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 2000
        }
        
        # Make request
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if debug:
            print(f"API Response Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.text[:200]}...")
            else:
                print("API request successful")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return {'status': 'success', 'content': content}
        else:
            error_msg = f"Error code: {response.status_code} - {response.text}"
            return {'status': 'error', 'error': error_msg}
            
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def save_json_output(content, output_path, print_raw=False):
    """Save JSON output with robust parsing and error logging."""
    if print_raw:
        print(f"Raw content: {content}")
    
    # Try to extract JSON from the content
    json_content = None
    
    # Method 1: Try direct JSON parsing
    try:
        json_content = json.loads(content)
        return True, json_content
    except json.JSONDecodeError as e:
        if print_raw:
            print(f"Direct JSON parsing failed: {e}")
            # Try to identify the issue
            print(f"Content length: {len(content)}")
            print(f"First 100 chars: {repr(content[:100])}")
            print(f"Last 100 chars: {repr(content[-100:])}")
    
    # Method 2: Try to extract JSON from markdown code blocks
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        try:
            json_content = json.loads(json_match.group(1))
            return True, json_content
        except json.JSONDecodeError as e:
            if print_raw:
                print(f"Code block JSON parsing failed: {e}")
    
    # Method 3: Try to find JSON between curly braces
    brace_match = re.search(r'\{.*\}', content, re.DOTALL)
    if brace_match:
        try:
            json_content = json.loads(brace_match.group(0))
            return True, json_content
        except json.JSONDecodeError as e:
            if print_raw:
                print(f"Brace matching JSON parsing failed: {e}")
    
    # If all parsing methods failed, log the error
    return False, content

def process_single_image(args):
    """Process a single image - designed for multiprocessing."""
    image_path, output_path, model, temperature, top_p, api_key, print_raw, error_log_dir, debug = args
    
    try:
        # Analyze the image
        result = analyze_comic_page(image_path, model, temperature, top_p, api_key, debug)
        
        # Check if it's an error response
        if result['status'] == 'error':
            return {
                'status': 'error',
                'image_path': str(image_path),
                'output_path': str(output_path),
                'error': result['error']
            }
        
        # Get the content from successful response
        content = result['content']
        
        # Try to parse JSON
        success, json_content = save_json_output(content, output_path, print_raw)
        
        if success:
            # Save successful JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
            
            return {
                'status': 'success',
                'image_path': str(image_path),
                'output_path': str(output_path)
            }
        else:
            # Log failed JSON parsing
            error_log_path = None
            if error_log_dir:
                # Create error log filename based on image path
                image_name = Path(image_path).stem
                error_log_path = Path(error_log_dir) / f"{image_name}_json_error.txt"
                
                # Save the failed content for analysis
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Model: {model}\n")
                    f.write(f"Temperature: {temperature}\n")
                    f.write(f"Top-p: {top_p}\n")
                    f.write("=" * 80 + "\n")
                    f.write("RAW CONTENT:\n")
                    f.write("=" * 80 + "\n")
                    f.write(content)
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("JSON PARSING FAILED\n")
                    f.write("=" * 80 + "\n")
            
            return {
                'status': 'json_parse_error',
                'image_path': str(image_path),
                'output_path': str(output_path),
                'error_log_path': str(error_log_path) if error_log_path else None,
                'raw_content': content[:500] + "..." if len(content) > 500 else content
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'image_path': str(image_path),
            'output_path': str(output_path),
            'error': str(e)
        }

def find_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Find all image files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return []
    
    print(f"Scanning directory: {input_dir}")
    image_files = []
    
    # Use rglob for faster scanning (like the original)
    for ext in extensions:
        print(f"Scanning for {ext} files...")
        files = list(input_path.rglob(f"*{ext}"))
        files.extend(input_path.rglob(f"*{ext.upper()}"))
        image_files.extend(files)
        print(f"Found {len(files)} {ext} files")
    
    print(f"Found {len(image_files)} total image files")
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description='Batch comic analysis with multiprocessing')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing comic images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--model', type=str, default="qwen/qwen2.5-vl-72b-instruct:free",
                       help='OpenRouter model to use')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p sampling parameter')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenRouter API key')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from specific image index')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have output files')
    parser.add_argument('--print-raw', action='store_true',
                       help='Print raw API responses')
    parser.add_argument('--error-log-dir', type=str, default=None,
                       help='Directory to save JSON parsing error logs')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for API requests')
    
    args = parser.parse_args()
    
    print("Starting batch comic analysis...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    print("Input directory validated")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory created/validated")
    
    # Count existing results
    existing_count = count_existing_results(output_dir)
    print(f"Existing results found: {existing_count}")
    
    # Create error log directory if specified
    error_log_dir = None
    if args.error_log_dir:
        error_log_dir = Path(args.error_log_dir)
        error_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"JSON parsing errors will be logged to: {error_log_dir}")
    
    print("Starting to find image files...")
    # Find image files
    image_files = find_image_files(input_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Image files found: {len(image_files)}")
    
    # Apply limits
    if args.max_images:
        image_files = image_files[args.start_from:args.start_from + args.max_images]
        print(f"Limited to {len(image_files)} images")
    else:
        image_files = image_files[args.start_from:]
        print(f"Using {len(image_files)} images from index {args.start_from}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Prepare arguments for multiprocessing
    print("Preparing multiprocessing arguments...")
    process_args = []
    skipped_count = 0
    error_count = 0
    current_dir = None
    
    for i, image_file in enumerate(image_files):
        # Print directory changes
        file_dir = str(image_file.parent)
        if file_dir != current_dir:
            print(f"Processing directory: {file_dir}")
            current_dir = file_dir
        
        try:
            # Determine output path - flatten structure
            # Use just the filename with a unique identifier to avoid conflicts
            image_name = image_file.stem
            # Create a unique identifier from the path to avoid name conflicts
            unique_id = str(image_file.relative_to(input_dir)).replace('\\', '_').replace('/', '_').replace('.', '_')
            output_path = output_dir / f"{unique_id}.json"
            
            # Skip if output exists and skip-existing is set
            if args.skip_existing and output_path.exists():
                skipped_count += 1
                continue
            
            # Create output directory if needed (should just be the main output dir)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                # If there's a file with the same name as the directory, skip this image
                print(f"Warning: Cannot create directory for {image_file.name} - file/directory conflict")
                print(f"  Image: {image_file}")
                print(f"  Output path: {output_path}")
                print(f"  Parent dir: {output_path.parent}")
                error_count += 1
                continue
            
            process_args.append((
                image_file,
                output_path,
                args.model,
                args.temperature,
                args.top_p,
                args.api_key,
                args.print_raw,
                error_log_dir,
                args.debug
            ))
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            error_count += 1
            continue
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing files")
    
    if error_count > 0:
        print(f"Encountered {error_count} errors during preparation")
    
    if not process_args:
        print("No files to process")
        return
    
    print(f"Processing {len(process_args)} files")
    print(f"Total progress: {existing_count} completed + {len(process_args)} to process = {existing_count + len(process_args)} total")
    
    # Set number of workers
    max_workers = args.max_workers or min(mp.cpu_count(), len(process_args))
    print(f"Using {max_workers} worker processes")
    
    # Process images in parallel
    print("Starting multiprocessing...")
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image, args): args for args in process_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(process_args), desc="Processing images", ncols=100) as pbar:
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                # Update progress bar description with current status
                success_count = sum(1 for r in results if r['status'] == 'success')
                error_count = sum(1 for r in results if r['status'] == 'error')
                json_error_count = sum(1 for r in results if r['status'] == 'json_parse_error')
                pbar.set_description(f"Success: {success_count}, Errors: {error_count}, JSON Errors: {json_error_count}")
                pbar.refresh()  # Force refresh the progress bar
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    json_error_count = sum(1 for r in results if r['status'] == 'json_parse_error')
    
    print(f"\n=== Analysis Summary ===")
    print(f"Total images processed this run: {len(process_args)}")
    print(f"Successful: {success_count}")
    print(f"JSON parsing errors: {json_error_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Previously completed: {existing_count}")
    print(f"Total completed: {existing_count + success_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(process_args):.2f} seconds")
    
    if json_error_count > 0 and error_log_dir:
        print(f"\nJSON parsing errors logged to: {error_log_dir}")
        print("You can analyze these files to identify parsing patterns and improve the parser.")
    
    # Print errors if any
    if error_count > 0:
        print(f"\n=== Errors ===")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['image_path']}: {result['error']}")
    
    if json_error_count > 0:
        print(f"\n=== JSON Parsing Errors ===")
        for result in results:
            if result['status'] == 'json_parse_error':
                print(f"  {result['image_path']}: JSON parsing failed")
                if result.get('error_log_path'):
                    print(f"    Error log: {result['error_log_path']}")

if __name__ == "__main__":
    main() 