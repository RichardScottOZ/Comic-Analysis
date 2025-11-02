#!/usr/bin/env python3
"""
Generate captions for 2000AD comic panels using Ollama with Gemma 12B model.

This script processes comic pages using Ollama's local Gemma 12B model
for comparison with cloud-based models.
"""

import os
import json
import argparse
import base64
from pathlib import Path
import requests
import time

def encode_image_to_data_uri(image_path):
    """Encode image to base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_str}"

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

def analyze_comic_page_ollama(image_path, model="gemma:12b", temperature=0.1, top_p=1.0):
    """Analyze a comic page using Ollama and return structured JSON output."""
    
    # Encode the image
    image_data_uri = encode_image_to_data_uri(image_path)
    
    # Prepare the prompt with image
    prompt = f"{create_structured_prompt()}\n\nHere is the comic page image: {image_data_uri}"
    
    # Ollama API request
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": 2048  # Limit response length
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

def save_json_output(content, output_path):
    """Save the JSON output to a file."""
    if not content:
        print("No content to save")
        return False
        
    try:
        # Try to parse the content as JSON directly
        json_data = json.loads(content)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON output saved to: {output_path}")
        return True
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from the response
        try:
            import re
            # Look for JSON block between ```json and ``` markers
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                json_data = json.loads(json_content)
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                print(f"JSON output extracted and saved to: {output_path}")
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
                    
                    print(f"JSON output extracted and saved to: {output_path}")
                    return True
                else:
                    raise ValueError("No JSON content found")
                    
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            print("Raw content:")
            print(content)
            return False

def check_ollama_available():
    """Check if Ollama is running and the model is available."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            print(f"Available Ollama models: {available_models}")
            return True
        else:
            print("Ollama is not responding properly")
            return False
    except requests.exceptions.RequestException:
        print("Ollama is not running. Please start Ollama first.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Analyze comic pages with Ollama Gemma 12B')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to the comic page image')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save the JSON output (default: same name as image with .json extension)')
    parser.add_argument('--model', type=str, default='gemma:12b',
                       help='Ollama model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for generation (0.0-1.0)')
    parser.add_argument('--print-raw', action='store_true',
                       help='Print the raw response before JSON parsing')
    
    args = parser.parse_args()
    
    # Check if Ollama is available
    if not check_ollama_available():
        return
    
    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = image_path.with_suffix('.json')
    
    print(f"Analyzing comic page: {image_path}")
    print(f"Using Ollama model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    
    # Analyze the comic page
    try:
        result = analyze_comic_page_ollama(str(image_path), args.model, args.temperature, args.top_p)
        
        if result is None:
            print("Analysis failed - no response from Ollama")
            return
        
        if args.print_raw:
            print("\nRaw response:")
            print(result)
            print("\n" + "="*50 + "\n")
        
        # Try to save as JSON
        if save_json_output(result, output_path):
            print("Analysis completed successfully!")
        else:
            print("Analysis completed but JSON parsing failed. Raw output printed above.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 