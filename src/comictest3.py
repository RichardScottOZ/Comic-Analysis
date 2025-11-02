from openai import OpenAI
import base64
import json
import argparse
from pathlib import Path
import re

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

def analyze_comic_page(image_path, model="qwen/qwen2.5-vl-32b-instruct:free", temperature=0.1, top_p=1.0):
    """Analyze a comic page and return structured JSON output."""
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="bananasplitsapikey",
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
        
        print(f"JSON output saved to: {output_path}")
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

def main():
    parser = argparse.ArgumentParser(description='Analyze comic pages with structured JSON output')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to the comic page image')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save the JSON output (default: same name as image with .json extension)')
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-vl-32b-instruct:free',
                       help='Model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for generation (0.0-1.0)')
    parser.add_argument('--print-raw', action='store_true',
                       help='Print the raw response before JSON parsing')
    
    args = parser.parse_args()
    
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
    print(f"Using model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    
    # Analyze the comic page
    try:
        result = analyze_comic_page(str(image_path), args.model, args.temperature, args.top_p)
        
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