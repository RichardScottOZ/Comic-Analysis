#!/usr/bin/env python3
"""
Test Local GGUF VLM (GLM-4V-Flash) via llama-cpp-python
Uses the modern from_pretrained logic.
"""

import argparse
import base64
import json
import os
from pathlib import Path

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("Error: llama-cpp-python not installed. Run: pip install llama-cpp-python")
    exit(1)

# Paths
DEFAULT_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
REPO_ID = "bartowski/zai-org_GLM-4.6V-Flash-GGUF"
MODEL_FILE = "zai-org_GLM-4.6V-Flash-Q6_K_L.gguf"
MMPROJ_FILE = "mmproj-zai-org_GLM-4.6V-Flash-f16.gguf"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"

def get_integrated_prompt():
    return """Analyze this comic page. Provide a detailed structured analysis in JSON format.
Identify every panel. For each panel, provide its BOUNDING BOX [ymin, xmin, ymax, xmax] (0-1000).
Transcribe all dialogue.
Return ONLY valid JSON."""

def run_test():
    print(f"Loading model from {REPO_ID}...")
    
    try:
        # Initialize the vision handler (Projector)
        chat_handler = Llava15ChatHandler.from_pretrained(
            repo_id=REPO_ID,
            filename=MMPROJ_FILE
        )
        
        # Initialize the main model
        llm = Llama.from_pretrained(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            chat_handler=chat_handler,
            n_ctx=8192,
            n_gpu_layers=-1 # Offload to GPU
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if not os.path.exists(DEFAULT_IMAGE):
        print(f"Image not found: {DEFAULT_IMAGE}")
        return

    print(f"Processing image: {DEFAULT_IMAGE}")
    data_uri = encode_image(DEFAULT_IMAGE)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": get_integrated_prompt()},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1
    )

    content = response['choices'][0]['message']['content']
    print("\n--- Model Output ---")
    print(content)
    
    with open("experiment_local_glm_flash.json", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    run_test()