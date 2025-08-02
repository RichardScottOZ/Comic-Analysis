from openai import OpenAI
import base64
import json
import argparse
from pathlib import Path
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import os

def encode_image_to_data_uri(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    # Adjust the MIME type if your image is not PNG
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
    image_path, output_path, model, temperature, top_p, api_key, print_raw = args
    
    try:
        # Analyze the comic page
        result = analyze_comic_page(str(image_path), model, temperature, top_p, api_key)
        
        if print_raw:
            print(f"\nRaw response for {image_path}:")
            print(result)
            print("\n" + "="*50 + "\n")
        
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

def find_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Find all image files in the input directory."""
    input_path = Path(input_dir)
    image_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            image_files.append(input_path)
    else:
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description='Analyze comic pages with structured JSON output using multiprocessing')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to comic page image or directory containing images')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save JSON outputs (default: same directory as input)')
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-vl-32b-instruct:free',
                       help='Model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for generation (0.0-1.0)')
    parser.add_argument('--print-raw', action='store_true',
                       help='Print the raw response before JSON parsing')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenRouter API key (default: uses environment variable or default key)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have corresponding JSON files')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Find image files
    image_files = find_image_files(args.input_path)
    
    if not image_files:
        print(f"No image files found in: {args.input_path}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Processing first {len(image_files)} images")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.input_path).parent if Path(args.input_path).is_file() else Path(args.input_path)
    
    # Prepare arguments for multiprocessing
    process_args = []
    skipped_count = 0
    
    for image_path in image_files:
        # Determine output path
        if args.output_dir:
            output_path = output_dir / f"{image_path.stem}.json"
        else:
            output_path = image_path.with_suffix('.json')
        
        # Skip if file exists and skip-existing is set
        if args.skip_existing and output_path.exists():
            skipped_count += 1
            continue
        
        process_args.append((
            image_path,
            output_path,
            args.model,
            args.temperature,
            args.top_p,
            args.api_key,
            args.print_raw
        ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing files")
    
    if not process_args:
        print("No images to process")
        return
    
    print(f"Processing {len(process_args)} images")
    print(f"Using model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Output directory: {output_dir}")
    
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