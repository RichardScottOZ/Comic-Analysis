#!/usr/bin/env python3
"""
Panel-level multiprocessing batch comic analysis using OpenRouter API.
Processes multiple comic images in parallel and saves structured JSON output with detailed panel information.

Enhanced features:
1. Panel geometry extraction (bounding boxes)
2. Detailed character identification per panel
3. Dialogue with speaker attribution
4. Action descriptions per panel
5. Setting and environment details
6. Visual element analysis
7. Panel sequence and flow analysis
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
import re

def encode_image_to_data_uri(image_path):
    """Encode image to data URI for API request."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def create_panel_structured_prompt():
    """Create a structured prompt for detailed panel-level comic analysis."""
    return """Analyze this comic page and provide detailed panel-level analysis in JSON format.

CRITICAL REQUIREMENTS:
1. **Panel Geometry**: Provide precise bounding box coordinates [x1, y1, x2, y2] for each panel
2. **Panel Sequence**: Number panels in reading order (left-to-right, top-to-bottom)
3. **Character Details**: Identify all characters in each panel with their actions
4. **Dialogue Attribution**: Match dialogue to specific speakers
5. **Visual Elements**: Describe art style, colors, composition per panel

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the entire page",
  "page_layout": {
    "total_panels": 6,
    "reading_order": "left-to-right, top-to-bottom",
    "art_style": "detailed, dynamic",
    "color_scheme": "vibrant, high contrast"
  },
  "panels": [
    {
      "panel_number": 1,
      "bbox": [x1, y1, x2, y2],
      "caption": "Panel title/description",
      "description": "Detailed visual description",
      "characters": [
        {
          "name": "Character name",
          "role": "hero/villain/supporting",
          "description": "What they look like",
          "actions": ["action1", "action2"],
          "emotions": ["determined", "angry"]
        }
      ],
      "dialogue": [
        {
          "speaker": "Character name",
          "text": "What they say",
          "speech_type": "dialogue|thought|narration|sound_effect",
          "emotion": "angry|happy|sad|neutral"
        }
      ],
      "actions": ["web swinging", "punching", "running"],
      "setting": "New York rooftops",
      "visual_elements": {
        "colors": ["red", "blue", "green"],
        "composition": "dynamic diagonal",
        "lighting": "bright daylight",
        "perspective": "heroic low angle"
      },
      "key_elements": ["web", "costume", "buildings"],
      "mood": "tense|action|calm|dramatic"
    }
  ],
  "story_elements": {
    "characters": ["Character1", "Character2"],
    "setting": "Overall setting description",
    "plot": "What's happening in this page",
    "conflict": "Main conflict or tension",
    "dialogue_summary": ["Key line 1", "Key line 2"]
  }
}"""

def create_enhanced_panel_prompt():
    """Alternative enhanced prompt for even more detailed analysis."""
    return """Analyze this comic page with EXTREME attention to panel geometry and detail.

REQUIRED OUTPUT FORMAT:
{
  "overall_summary": "Page summary",
  "page_metadata": {
    "total_panels": 6,
    "layout_type": "grid|dynamic|splash",
    "reading_pattern": "left-to-right|top-to-bottom|zigzag",
    "art_style": "detailed|cartoon|realistic",
    "color_palette": ["primary colors"],
    "mood": "action|drama|comedy|horror"
  },
  "panels": [
    {
      "panel_number": 1,
      "bbox": [x1, y1, x2, y2],
      "size": "large|medium|small",
      "shape": "rectangle|square|irregular",
      "caption": "Panel description",
      "description": "Detailed visual description",
      "characters": [
        {
          "name": "Character name",
          "role": "protagonist|antagonist|supporting",
          "costume": "What they're wearing",
          "pose": "What they're doing",
          "facial_expression": "emotion shown",
          "actions": ["specific actions"],
          "dialogue": "What they say (if any)"
        }
      ],
      "dialogue": [
        {
          "speaker": "Character name",
          "text": "Exact dialogue",
          "type": "speech|thought|narration|sound",
          "bubble_type": "speech|thought|whisper|shout",
          "position": "top|bottom|left|right"
        }
      ],
      "actions": ["specific actions happening"],
      "setting": "Where this panel takes place",
      "background": "What's in the background",
      "foreground": "What's in the foreground",
      "visual_style": {
        "line_style": "thick|thin|varied",
        "shading": "crosshatch|solid|gradient",
        "perspective": "eye-level|low-angle|high-angle",
        "composition": "rule-of-thirds|centered|dynamic"
      },
      "colors": {
        "dominant": "main color",
        "accent": "highlight color",
        "mood": "warm|cool|neutral"
      },
      "special_effects": ["motion lines", "sound effects", "lighting"],
      "mood": "tense|action|calm|dramatic",
      "importance": "key|transition|background"
    }
  ],
  "story_analysis": {
    "characters": ["all characters"],
    "setting": "overall setting",
    "plot": "what's happening",
    "conflict": "main tension",
    "pacing": "fast|medium|slow",
    "climax": "is this a key moment?",
    "foreshadowing": "any hints of future events"
  }
}

CRITICAL: Provide accurate bounding boxes and detailed character/dialogue matching."""

def count_existing_results(output_dir):
    """Count existing JSON result files in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    
    json_files = list(output_path.glob("*.json"))
    return len(json_files)

def analyze_comic_page_panel(image_path, model="qwen/qwen2.5-vl-72b-instruct:free", 
                           temperature=0.1, top_p=1.0, api_key=None, debug=False, 
                           timeout=960, enhanced_prompt=False):
    """Analyze a single comic page with detailed panel-level information."""
    if api_key is None:
        api_key = "bananasplitsapikey"
    
    try:
        # Encode image
        image_data_uri = encode_image_to_data_uri(image_path)
        
        # Choose prompt based on enhancement level
        if enhanced_prompt:
            prompt = create_enhanced_panel_prompt()
        else:
            prompt = create_panel_structured_prompt()
        
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
                            "text": prompt
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
            "max_tokens": 3000  # Increased for detailed panel analysis
        }
        
        # Make request with configurable timeout
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        
        if debug:
            print(f"API Response Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.text[:200]}...")
            else:
                print("API request successful")
        
        if response.status_code == 200:
            result = response.json()
            if debug:
                print(f"DEBUG: Response keys: {list(result.keys())}")
                print(f"DEBUG: Response structure: {result}")
                print(f"DEBUG: Image path: {image_path}")
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # Try to parse JSON
                try:
                    json_data = json.loads(content)
                    return {
                        'success': True,
                        'content': json_data,
                        'raw_content': content
                    }
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"JSON parsing failed: {e}")
                        print(f"Raw content: {content[:500]}...")
                    
                    # Try to repair JSON
                    repaired_content = repair_json_content(content)
                    if debug:
                        print(f"DEBUG: After repair (length: {len(repaired_content)})")
                        print(f"DEBUG: First 200 chars after repair: {repaired_content[:200]}")
                    
                    try:
                        json_data = json.loads(repaired_content)
                        return {
                            'success': True,
                            'content': json_data,
                            'raw_content': repaired_content,
                            'repaired': True
                        }
                    except json.JSONDecodeError as repair_error:
                        if debug:
                            print(f"DEBUG: Repair failed: {repair_error}")
                            print(f"DEBUG: Repaired content: {repaired_content[:500]}...")
                        return {
                            'success': False,
                            'error': f'JSON parsing failed after repair: {repair_error}',
                            'raw_content': content
                        }
            else:
                return {
                    'success': False,
                    'error': 'No choices in response',
                    'raw_content': str(result)
                }
        else:
            error_msg = f"API request failed with status {response.status_code}"
            if response.text:
                error_msg += f": {response.text[:200]}"
            return {
                'success': False,
                'error': error_msg,
                'raw_content': response.text
            }
    
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': f'Request timeout after {timeout} seconds',
            'raw_content': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'raw_content': ''
        }

def repair_json_content(content):
    """Enhanced JSON repair function for panel-level content."""
    if not content:
        return content
    
    # First, try to find JSON blocks marked with ```json and ```
    json_block_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_block_pattern, content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()
        # If we found a JSON block, return it directly - it should be valid
        return content
    
    # If no JSON block found, try to extract JSON from the content
    # Remove any text before the first {
    start_idx = content.find('{')
    if start_idx != -1:
        content = content[start_idx:]
    
    # Remove any text after the last }
    end_idx = content.rfind('}')
    if end_idx != -1:
        content = content[:end_idx + 1]
    
    # Only apply minimal repairs if the JSON is clearly malformed
    # Don't apply aggressive regex patterns that might break valid JSON
    
    return content

def save_panel_json_output(content, output_path, print_raw=False):
    """Save panel-level JSON output with enhanced metadata."""
    try:
        # Add metadata
        if isinstance(content, dict):
            content['_metadata'] = {
                'analysis_type': 'panel_level',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'features': [
                    'panel_geometry',
                    'character_detection',
                    'dialogue_attribution',
                    'action_analysis',
                    'visual_elements'
                ]
            }
        
        # Save the JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        if print_raw:
            print(f"Raw content saved to: {output_path}")
        else:
            print(f"Panel analysis saved to: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error saving JSON output: {e}")
        return False

def process_single_image_panel(args):
    """Process a single image with panel-level analysis."""
    image_path, output_dir, model, temperature, top_p, api_key, debug, timeout, enhanced_prompt, error_log_dir = args
    
    try:
        # Create output filename
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{image_name}.json"
        
        # Skip if already exists
        if output_path.exists():
            return {
                'success': True,
                'skipped': True,
                'message': f'Skipped existing file: {image_name}'
            }
        
        # Analyze the image
        result = analyze_comic_page_panel(
            image_path, model, temperature, top_p, api_key, debug, timeout, enhanced_prompt
        )
        
        if result['success']:
            # Save the result
            save_success = save_panel_json_output(result['content'], output_path, debug)
            if save_success:
                return {
                    'success': True,
                    'message': f'Successfully analyzed: {image_name}',
                    'panels_found': len(result['content'].get('panels', [])),
                    'repaired': result.get('repaired', False)
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to save output for: {image_name}'
                }
        else:
            return {
                'success': False,
                'error': f'Analysis failed for {image_name}: {result["error"]}',
                'image_path': image_path,
                'raw_content': result.get('raw_content', '')
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error processing {image_path}: {str(e)}',
            'image_path': image_path
        }

def find_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Find all image files in the input directory and subdirectories."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    image_files = set()  # Use set to avoid duplicates
    for ext in extensions:
        image_files.update(input_path.rglob(f"*{ext}"))
        image_files.update(input_path.rglob(f"*{ext.upper()}"))
    
    return sorted(list(image_files))

def main():
    """Main function for panel-level batch comic analysis."""
    parser = argparse.ArgumentParser(description='Panel-level batch comic analysis with multiprocessing')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing comic images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for JSON results')
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-vl-72b-instruct:free',
                       help='OpenRouter model to use')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenRouter API key')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from this image index')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for model generation')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for model generation')
    parser.add_argument('--timeout', type=int, default=960,
                       help='Timeout for API requests in seconds')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--enhanced-prompt', action='store_true',
                       help='Use enhanced prompt for more detailed analysis')
    parser.add_argument('--panel-geometry', action='store_true',
                       help='Extract detailed panel geometry information')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip images that already have results')
    parser.add_argument('--error-log-dir', type=str, default=None,
                       help='Directory to save error logs for failed analyses')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create error log directory if specified
    if args.error_log_dir:
        error_log_path = Path(args.error_log_dir)
        error_log_path.mkdir(parents=True, exist_ok=True)
        print(f"Error logs will be saved to: {args.error_log_dir}")
    
    # Find image files
    print(f"Scanning for images in: {args.input_dir}")
    image_files = find_image_files(args.input_dir)
    
    if not image_files:
        print("No image files found!")
        return
    
    # Apply limits
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    if args.start_from > 0:
        image_files = image_files[args.start_from:]
    
    print(f"Found {len(image_files)} images to process")
    
    # Debug: Show first few files
    if args.debug:
        print(f"First 10 files to process:")
        for i, img_file in enumerate(image_files[:10]):
            print(f"  {i+1}: {img_file.name}")
    
    # Count existing results
    existing_count = count_existing_results(args.output_dir)
    print(f"Found {existing_count} existing results")
    
    # Skip existing files if requested
    if args.skip_existing:
        filtered_files = []
        for img_file in image_files:
            output_file = output_path / f"{img_file.stem}.json"
            if not output_file.exists():
                filtered_files.append(img_file)
        
        skipped_count = len(image_files) - len(filtered_files)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} existing files")
        image_files = filtered_files
        
        if args.debug:
            print(f"After filtering, {len(image_files)} files remain to process")
    
    if not image_files:
        print("No new images to process!")
        return
    
    # Prepare arguments for multiprocessing
    mp_args = [
        (str(img_file), args.output_dir, args.model, args.temperature, 
         args.top_p, args.api_key, args.debug, args.timeout, args.enhanced_prompt, args.error_log_dir)
        for img_file in image_files
    ]
    
    # Process images with multiprocessing
    max_workers = args.max_workers or max(mp.cpu_count(), 8)
    print(f"Using {max_workers} worker processes")
    
    successful = 0
    failed = 0
    skipped = 0
    json_errors = 0
    api_errors = 0
    other_errors = 0
    total_panels = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image_panel, args) for args in mp_args]
        
        with tqdm(total=len(futures), desc="Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['success']:
                    if result.get('skipped', False):
                        skipped += 1
                    else:
                        successful += 1
                        total_panels += result.get('panels_found', 0)
                        if args.debug:
                            print(f"✅ {result['message']}")
                else:
                    failed += 1
                    error_msg = result['error']
                    
                    # Categorize errors
                    if 'JSON parsing failed' in error_msg:
                        json_errors += 1
                    elif 'API request failed' in error_msg or 'timeout' in error_msg.lower():
                        api_errors += 1
                    else:
                        other_errors += 1
                    
                    # Save error log if directory is specified
                    if args.error_log_dir and 'image_path' in result:
                        try:
                            error_log_path = Path(args.error_log_dir)
                            error_file = error_log_path / f"{Path(result['image_path']).stem}_error.log"
                            
                            with open(error_file, 'w', encoding='utf-8') as f:
                                f.write(f"Image: {result['image_path']}\n")
                                f.write(f"Error: {error_msg}\n")
                                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                                if 'raw_content' in result:
                                    f.write(f"\nRaw API Response:\n{result['raw_content']}\n")
                        except Exception as e:
                            print(f"Warning: Could not save error log: {e}")
                    
                    if args.debug:
                        print(f"❌ {error_msg}")
                
                # Update progress bar description with current status
                pbar.set_description(f"Success: {successful}, Failed: {failed}, JSON Errors: {json_errors}, API Errors: {api_errors}, Other Errors: {other_errors}, Panels: {total_panels}")
                pbar.update(1)
    
    # Print summary
    print(f"\n=== Panel-Level Analysis Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total panels analyzed: {total_panels}")
    print(f"Average panels per page: {total_panels/successful if successful > 0 else 0:.1f}")
    
    if failed > 0:
        print(f"\n=== Error Breakdown ===")
        print(f"JSON parsing errors: {json_errors}")
        print(f"API errors: {api_errors}")
        print(f"Other errors: {other_errors}")

if __name__ == "__main__":
    main() 