#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import zarr
from pathlib import Path

# Add version2 to sys.path
version2_dir = Path(__file__).resolve().parents[1] / "version2"
sys.path.append(str(version2_dir))
from stage4_sequence_modeling_framework import Stage4SequenceModel

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)

def main():
    checkpoint_path = "checkpoints/stage4/best_model.pt"
    stage3_zarr_path = "E:/Comic_Analysis_Results_v2/stage3_embeddings.zarr"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Stage 4 checkpoint not found at {checkpoint_path}")
        return
    if not os.path.exists(stage3_zarr_path):
        print(f"Error: Stage 3 Zarr not found at {stage3_zarr_path}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Stage 4 model
    print("Loading Stage 4 model...")
    model = Stage4SequenceModel(d_model=512, num_layers=6, nhead=8, max_panels=16).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("Stage 4 model loaded successfully.")

    # 2. Load two different page sequences from Stage 3 Zarr
    print("Loading two sample pages from Stage 3...")
    store = zarr.open(stage3_zarr_path, mode='r')
    panel_embs = store['panel_embeddings']
    masks = store['panel_masks']
    
    # Page 0 (valid panels)
    mask0 = masks[0]
    num_p0 = int(np.sum(mask0))
    p0_in = torch.from_numpy(panel_embs[0, :num_p0]).float().to(device)  # (N0, 512)
    
    # Page 1 (valid panels)
    mask1 = masks[1]
    num_p1 = int(np.sum(mask1))
    p1_in = torch.from_numpy(panel_embs[1, :num_p1]).float().to(device)  # (N1, 512)
    
    print(f"Page 0 input has {num_p0} panels. Page 1 input has {num_p1} panels.")
    print(f"Cosine similarity of raw Stage 3 inputs (mean-pooled): {cosine_similarity(np.mean(panel_embs[0, :num_p0], axis=0), np.mean(panel_embs[1, :num_p1], axis=0)):.6f}")

    # 3. Pass through Stage 4 model
    with torch.no_grad():
        out0_context, out0_strip = model.sequence_transformer(p0_in.unsqueeze(0))
        out1_context, out1_strip = model.sequence_transformer(p1_in.unsqueeze(0))
        
    out0_strip_np = out0_strip.squeeze(0).cpu().numpy()
    out1_strip_np = out1_strip.squeeze(0).cpu().numpy()
    
    print("\n--- Model Output Check ---")
    sim = cosine_similarity(out0_strip_np, out1_strip_np)
    print(f"Cosine similarity between Page 0 and Page 1 predicted strip_embeddings: {sim:.6f}")
    
    # Check if the outputs are identical
    print(f"Page 0 strip embedding first 5 values: {out0_strip_np[:5]}")
    print(f"Page 1 strip embedding first 5 values: {out1_strip_np[:5]}")
    
    # Print norms
    print(f"Page 0 strip embedding norm: {np.linalg.norm(out0_strip_np):.6f}")
    print(f"Page 1 strip embedding norm: {np.linalg.norm(out1_strip_np):.6f}")

if __name__ == "__main__":
    main()
