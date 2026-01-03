#!/usr/bin/env python3
"""
Test PSS Embedding Logic (Pure Local)
Verifies models, box filtering, and embedding shapes without S3.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel
from sentence_transformers import SentenceTransformer

# Target local files for testing
TEST_IMAGE = "E:\\amazon\\#Guardian 001\\#Guardian 001 - p003.jpg"
TEST_JSON = "E:\\Comic_Analysis_Results_v2\\vlm_lite_guided\\#Guardian 001_#Guardian 001 - p003.jpg.json"

# Models (Same as production)
VISUAL_MODEL = "google/siglip-so400m-patch14-384"
TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

def filter_boxes(data):
    """Recursively removes keys related to bounding boxes/coordinates."""
    if isinstance(data, dict):
        junk_keys = {'box_2d', 'box', 'polygon', 'detections', 'coordinates', 'bbox'}
        return {k: filter_boxes(v) for k, v in data.items() if k.lower() not in junk_keys}
    elif isinstance(data, list):
        return [filter_boxes(i) for i in data]
    else:
        return data

def test_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Test JSON Filtering
    print(f"\n--- 1. Testing JSON Filter ---")
    if os.path.exists(TEST_JSON):
        with open(TEST_JSON, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Strip metadata
        raw_data.pop('canonical_id', None)
        raw_data.pop('processed_at', None)
        
        # Filter
        cleaned = filter_boxes(raw_data)
        
        # Verify box_2d is gone
        if 'box_2d' in str(cleaned):
            print("❌ FILTER FAILED: Found 'box_2d' in cleaned output.")
        else:
            print("✅ FILTER PASSED: Geometric info removed.")
            
        json_str = json.dumps(cleaned, separators=(',', ':'))
        print(f"Serialized Length: {len(json_str)} chars")
        print(f"Sample: {json_str[:100]}...")
    else:
        print(f"⚠️ Skip: JSON not found at {TEST_JSON}")
        json_str = "{}"

    # 2. Test Visual Embedding (SigLIP)
    print(f"\n--- 2. Testing SigLIP ---")
    if os.path.exists(TEST_IMAGE):
        proc = AutoProcessor.from_pretrained(VISUAL_MODEL)
        # Load ONLY the vision tower
        model = SiglipVisionModel.from_pretrained(VISUAL_MODEL).to(device).eval()
        
        img = Image.open(TEST_IMAGE).convert('RGB')
        inputs = proc(images=[img], return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model(**inputs)
            embed = out.pooler_output
            
        print(f"Visual Embedding Shape: {embed.shape} (Expected: [1, 1152])")
        if embed.shape[1] == 1152:
            print("✅ SigLIP PASSED.")
    else:
        print(f"⚠️ Skip: Image not found at {TEST_IMAGE}")

    # 3. Test Text Embedding (Qwen)
    print(f"\n--- 3. Testing Qwen ---")
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    
    with torch.no_grad():
        txt_embed = txt_model.encode([json_str], convert_to_numpy=True)
        
    print(f"Text Embedding Shape: {txt_embed.shape} (Expected: [1, 1024])")
    if txt_embed.shape[1] == 1024:
        print("✅ Qwen PASSED.")

    print("\n--- Summary ---")
    print("If all shapes and filters match, the pipeline is ready for S3.")

if __name__ == "__main__":
    test_logic()
