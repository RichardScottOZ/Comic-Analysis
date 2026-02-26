#!/usr/bin/env python3
"""
Verify Stage 3 Embeddings
Loads the generated Stage 3 Zarr and Metadata to verify data integrity,
shapes, and perform contrastive sanity checks.
"""

import zarr
import json
import numpy as np
import argparse
import random
import os

def cosine_similarity(a, b):
    # a: (N, D), b: (M, D)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', type=str, default="E:/Comic_Analysis_Results_v2/stage3_embeddings.zarr")
    parser.add_argument('--metadata', type=str, default="E:/Comic_Analysis_Results_v2/stage3_metadata.json")
    args = parser.parse_args()

    if not os.path.exists(args.zarr):
        print(f"Error: Zarr store not found at {args.zarr}")
        return
        
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata JSON not found at {args.metadata}")
        return

    print(f"Loading Zarr: {args.zarr}")
    try:
        store = zarr.open(args.zarr, mode='r')
        embeddings = store['panel_embeddings']
        masks = store['panel_masks']
        print(f"✅ Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        print(f"✅ Masks shape:      {masks.shape}, dtype: {masks.dtype}")
    except Exception as e:
        print(f"❌ Failed to load Zarr: {e}")
        return

    print(f"\nLoading Metadata: {args.metadata}")
    try:
        with open(args.metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Metadata entries: {len(metadata)}")
    except Exception as e:
        print(f"❌ Failed to load Metadata: {e}")
        return

    if len(metadata) != embeddings.shape[0]:
        print(f"⚠️ WARNING: Length mismatch! Metadata ({len(metadata)}) != Zarr ({embeddings.shape[0]})")
    else:
        print("✅ Lengths match perfectly.")
        
    if len(metadata) == 0:
        print("Dataset is empty. Run generate_stage3_embeddings.py first.")
        return

    print("\n--- Basic Statistics (First 1000 pages) ---")
    sample_size = min(1000, embeddings.shape[0])
    sample_emb = embeddings[:sample_size]
    sample_mask = masks[:sample_size]
    
    # Flatten and keep only valid panels based on boolean mask
    valid_embs = sample_emb[sample_mask]
    
    if len(valid_embs) > 0:
        print(f"Valid Panels in sample: {len(valid_embs)}")
        print(f"Mean: {np.mean(valid_embs):.4f}")
        print(f"Std:  {np.std(valid_embs):.4f}")
        print(f"Min:  {np.min(valid_embs):.4f}")
        print(f"Max:  {np.max(valid_embs):.4f}")
        
        has_nan = np.isnan(valid_embs).any()
        if has_nan:
            print("❌ WARNING: NaNs detected in embeddings!")
        else:
            print("✅ No NaNs detected.")
            
        if np.allclose(valid_embs, 0):
            print("❌ WARNING: Embeddings are all zeros!")
    else:
        print("No valid panels found in sample.")

    print("\n--- Qualitative Discriminative Check ---")
    # Pick a random sequence with > 1 panel
    valid_indices = [i for i, m in enumerate(metadata) if m['num_panels'] > 1]
    if not valid_indices:
        print("No sequences with > 1 panel found.")
        return
        
    idx1 = random.choice(valid_indices)
    
    m1 = metadata[idx1]
    e1 = embeddings[idx1]
    mask1 = masks[idx1]
    valid_e1 = e1[mask1]
    
    print(f"Page A: {m1['canonical_id']}")
    print(f"  Valid Panels: {len(valid_e1)}")
    
    sim_matrix = cosine_similarity(valid_e1, valid_e1)
    print("\n  Internal Cosine Similarity (Panel vs Panel on Page A):")
    print(np.round(sim_matrix, 2))
    
    if len(valid_indices) > 1:
        idx2 = random.choice(valid_indices)
        while idx2 == idx1:
            idx2 = random.choice(valid_indices)
            
        m2 = metadata[idx2]
        e2 = embeddings[idx2]
        mask2 = masks[idx2]
        valid_e2 = e2[mask2]
        
        print(f"\nPage B: {m2['canonical_id']}")
        
        cross_sim = cosine_similarity(valid_e1, valid_e2)
        print("\n  Cross-Page Similarity (Page A vs Page B):")
        print(np.round(cross_sim, 2))
        
        # Calculate means (excluding diagonal for internal)
        mean_internal = np.mean(sim_matrix[~np.eye(len(sim_matrix), dtype=bool)]) if len(sim_matrix) > 1 else 1.0
        mean_cross = np.mean(cross_sim)
        
        print(f"\nMean Internal Similarity (Same Page): {mean_internal:.4f}")
        print(f"Mean Cross Similarity (Diff Page):  {mean_cross:.4f}")
        
        if mean_internal > mean_cross:
            print("\n✅ SUCCESS: Contrastive separation is working (Internal > Cross).")
            print("The model has learned discriminative features.")
        else:
            print("\n⚠️ WARNING: Cross-page similarity is higher than internal.")
            print("Model might not be highly discriminative.")

if __name__ == "__main__":
    main()
