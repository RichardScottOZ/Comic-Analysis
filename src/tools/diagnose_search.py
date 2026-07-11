#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import zarr
from pathlib import Path

# Add version2 directory to sys.path
version2_dir = Path(__file__).resolve().parents[1] / "version2"
sys.path.append(str(version2_dir))
from stage3_panel_features_framework import PanelFeatureExtractor
from interface.search_utils_vlm import get_embedding_for_text, _get_tokenizer

def main():
    checkpoint_path = "checkpoints/stage3_vlm/best_model_vlm.pt"
    zarr_path = "E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return
    if not os.path.exists(zarr_path):
        print(f"Error: zarr not found at {zarr_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("Loading Stage 3 VLM model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PanelFeatureExtractor(
        visual_backbone="both",
        visual_fusion="attention",
        feature_dim=512,
        freeze_backbones=True,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # Encode query text
    query_text = "woman with red skin in bikini"
    print(f"Encoding text query: '{query_text}'")
    query_emb = get_embedding_for_text(model, query_text, device) # (1, 512)
    print(f"Query embedding shape: {query_emb.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_emb)}")
    print(f"Query embedding slice (first 10): {query_emb[0, :10]}")
    
    # Load Zarr
    print("Opening Zarr store...")
    root = zarr.open(zarr_path, mode='r')
    strip_embs = root['strip_embeddings']
    print(f"Database strip embeddings shape: {strip_embs.shape}")
    
    # Compute similarity manually for the first 1000 pages
    sample_size = min(1000, strip_embs.shape[0])
    db_slice = strip_embs[:sample_size].astype(np.float32)
    
    # Normalize query and db slice
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    db_norms = np.linalg.norm(db_slice, axis=1, keepdims=True) + 1e-8
    db_normalized = db_slice / db_norms
    
    similarities = np.dot(db_normalized, q.T).squeeze(-1)
    
    print("\n--- Similarity Statistics ---")
    print(f"Min similarity:  {np.min(similarities):.6f}")
    print(f"Max similarity:  {np.max(similarities):.6f}")
    print(f"Mean similarity: {np.mean(similarities):.6f}")
    print(f"Std similarity:  {np.std(similarities):.6f}")
    
    # Let's inspect the first 10 similarities
    print(f"First 10 similarities: {similarities[:10]}")
    
    # Check if there are any database rows that are all-zeros
    zero_rows = np.all(db_slice == 0, axis=1)
    print(f"Number of all-zero rows in sample: {np.sum(zero_rows)}")

if __name__ == "__main__":
    main()
