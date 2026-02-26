#!/usr/bin/env python3
"""
Test Stage 3 Model with REAL Data
Loads a real page (Image + JSON), extracts panels/text, and computes pairwise similarity.
Uses the proven Suffix Bridge strategy to locate files.
"""

import torch
import numpy as np
import os
import json
import csv
from PIL import Image, ImageFile
import torchvision.transforms as T
from transformers import AutoTokenizer
from stage3_panel_features_framework import PanelFeatureExtractor
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

# Configuration
FEATURE_DIM = 512
IMAGE_SIZE = 224
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JSON_ROOT = "E:/Comic_Analysis_Results_v2/stage3_json"
MASTER_MANIFEST = "manifests/master_manifest_20251229.csv"
CALIBRE_MANIFEST = "manifests/calibrecomics-extracted_manifest.csv"

# ... (inside get_real_sample) ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="checkpoints/stage3/best_model.pt", help='Path to trained model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = PanelFeatureExtractor(
            visual_backbone='both',
            visual_fusion='attention',
            feature_dim=FEATURE_DIM
        ).to(device)
        # Support both formats: raw state_dict or dict with 'model_state_dict' key
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Get Real Data
    img_path, json_path = get_real_sample()
    
    if not img_path:
        print("❌ Could not find a valid Image+JSON pair.")
        return
        
    print(f"\nAnalyzing Page:")
    print(f"  Image: {img_path}")
    print(f"  JSON:  {json_path}")
    
    # 3. Process Data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    raw_img = Image.open(img_path).convert('RGB')
    width, height = raw_img.size
    
    panels = data.get('panels', [])
    if not panels:
        print("❌ JSON has no panels.")
        return
        
    print(f"  Found {len(panels)} panels.")
    
    # Transforms
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading tokenizer (local)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
    batch_images = []
    batch_texts = []
    batch_comp = []
    batch_mask = []
    
    for i, p in enumerate(panels):
        bbox = p.get('bbox') # [x, y, w, h]
        text = p.get('text', '')
        
        print(f"    Panel {i+1}: Text='{text[:50]}...'")
        
        # Crop
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w <= 0 or h <= 0: continue
        
        crop = raw_img.crop((x, y, x+w, y+h))
        batch_images.append(transform(crop))
        batch_texts.append(text)
        
        # Comp features (placeholder)
        batch_comp.append(torch.zeros(7))
        
        # Mask
        has_text = 1.0 if len(text.strip()) > 0 else 0.0
        batch_mask.append([1.0, has_text, 1.0])

    if not batch_images:
        print("No valid panels found.")
        return

    # 4. Inference
    text_enc = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    batch = {
        'images': torch.stack(batch_images).to(device),
        'input_ids': text_enc['input_ids'].to(device),
        'attention_mask': text_enc['attention_mask'].to(device),
        'comp_feats': torch.stack(batch_comp).to(device),
        'modality_mask': torch.tensor(batch_mask).to(device)
    }
    
    with torch.no_grad():
        embeddings = model(batch) # (N, D)
        
    # 5. Analysis
    emb_np = embeddings.cpu().numpy()
    
    print("\n--- Raw Embeddings (First 10 dims) ---")
    for i in range(len(emb_np)):
        print(f"Panel {i+1}: {np.round(emb_np[i][:10], 4)}")
    
    # Compute Pairwise Similarity Matrix
    norm = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norm_emb = emb_np / (norm + 1e-8)
    sim_matrix = np.dot(norm_emb, norm_emb.T)
    
    print("\n--- Pairwise Cosine Similarity ---")
    print(np.round(sim_matrix, 2))
    
    # Check for collapse
    off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    mean_sim = np.mean(off_diag) if len(off_diag) > 0 else 0
    print(f"\nMean Off-Diagonal Similarity: {mean_sim:.4f}")
    
    if mean_sim > 0.99:
        print("❌ MODEL COLLAPSED: All panels have identical embeddings.")
    else:
        print("✅ Embeddings are distinct and discriminative.")

if __name__ == "__main__":
    main()