#!/usr/bin/env python3
"""
Test Local Llama Server (GLM-4V-Flash)
Connects to a running llama-server instance (OpenAI compatible).

Usage:
1. Start server: 
   llama-server.exe -m model.gguf --mmproj mmproj.gguf --port 8080 -ngl 99
   
2. Run this script:
   python src/tests/test_local_server.py
"""

import argparse
import base64
import requests
import json
import os

# Paths
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
SERVER_URL = "http://localhost:80/v1/chat/completions"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        header = image_file.read(12)
        image_file.seek(0)
        content = image_file.read()
        
    if header.startswith(b'\x89PNG\r\n\x1a\n'): mime = 'image/png'
    elif header.startswith(b'\xff\xd8\xff'): mime = 'image/jpeg'
    else: mime = 'image/jpeg'

    return f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"

def get_integrated_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis in JSON format.

REQUIREMENTS:
1. Identify every panel. For each panel, provide its BOUNDING BOX [ymin, xmin, ymax, xmax] (0-1000).
2. Describe the visual content and action.
3. Transcribe all dialogue and attribute it to characters.

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
      "box_2d": [ymin, xmin, ymax, xmax],
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
}
"""
def run_test(image_path):
    print(f"Connecting to local server: {SERVER_URL}")
    print(f"Processing image: {image_path}")
    
    try:
        uri = encode_image(image_path)
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": "gpt-3.5-turbo", # Dummy name, server ignores it usually
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": get_integrated_prompt()},
                        {"type": "image_url", "image_url": {"url": uri}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        response = requests.post(SERVER_URL, headers=headers, json=payload, timeout=600)
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print("\n--- Model Output ---")
            print(content)
            
            with open("experiment_local_server_output.json", "w", encoding="utf-8") as f:
                f.write(content)
            print("\nSaved to experiment_local_server_output.json")
        else:
            print(f"Server Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to localhost:8080. Is llama-server running?")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
    else:
        run_test(args.image)
