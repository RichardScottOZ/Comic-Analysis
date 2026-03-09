#!/usr/bin/env python3
"""
Stage 5: Multi-Modal Search Engine (CLI)

This script allows you to query your generated Stage 4 (or Stage 3) embeddings
using text prompts or reference images.

It leverages the Stage 3 encoders to convert your query into a 512-d vector,
then performs a fast cosine similarity search across the Zarr database.
"""

import os
import argparse
import json
import torch
import zarr
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from pathlib import Path

# Need the Stage 3 model to encode queries
from stage3_panel_features_framework import PanelFeatureExtractor

def compute_similarity(query_emb, database_embs, batch_size=10000):
    """Computes cosine similarity in batches to save RAM."""
    # query_emb: (1, D)
    # database_embs: (N, D)
    
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    similarities = np.zeros(database_embs.shape[0], dtype=np.float32)
    
    for i in range(0, database_embs.shape[0], batch_size):
        end = min(i + batch_size, database_embs.shape[0])
        batch = database_embs[i:end]
        
        batch_norm = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(batch_norm, query_norm.T).squeeze()
        similarities[i:end] = sim
        
    return similarities

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Query Encoder (Stage 3 Model)
    print(f"Loading Stage 3 Encoder from {args.stage3_checkpoint}...")
    model = PanelFeatureExtractor(
        visual_backbone='both',
        visual_fusion='attention',
        feature_dim=512
    ).to(device)
    
    try:
        checkpoint = torch.load(args.stage3_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Failed to load Stage 3 model: {e}")
        return

    # 2. Encode Query
    query_emb = None
    with torch.no_grad():
        if args.text_query:
            print(f"Encoding Text Query: '{args.text_query}'")
            try:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                
            text_enc = tokenizer([args.text_query], return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = text_enc['input_ids'].to(device)
            attn_mask = text_enc['attention_mask'].to(device)
            
            # Use the text-only branch
            query_emb = model.encode_text_only(input_ids, attn_mask)
            
        elif args.image_query:
            print(f"Encoding Image Query: '{args.image_query}'")
            image = Image.open(args.image_query).convert('RGB')
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                
            text_enc = tokenizer([""], return_tensors='pt', padding='max_length', max_length=128)
            comp_feats = torch.zeros(1, 7).to(device)
            modality_mask = torch.tensor([[1.0, 0.0, 1.0]]).to(device)
            
            batch = {
                'images': img_tensor,
                'input_ids': text_enc['input_ids'].to(device),
                'attention_mask': text_enc['attention_mask'].to(device),
                'comp_feats': comp_feats,
                'modality_mask': modality_mask
            }
            
            query_emb = model(batch)
            
        else:
            print("Error: Must provide either --text_query or --image_query")
            return

    query_emb_np = query_emb.cpu().numpy() # (1, 512)

    # 3. Load Database (Zarr)
    print(f"\nLoading Zarr Database: {args.zarr}")
    store = zarr.open(args.zarr, mode='r')
    
    # Decide which array to search
    if args.search_target == 'panel':
        print("Target: Raw Panel Embeddings")
        raw_embs = store['panel_embeddings'][:] # Load into RAM
        masks = store['panel_masks'][:]
        
        # We need a flat list of valid panels and their metadata mapping
        print("Flattening valid panels...")
        valid_embs = raw_embs[masks] # This is fast and filters out padding
        
        # Load metadata to map flattened indices back to canonical_ids
        with open(args.metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Map flat index to (canonical_id, panel_number)
        flat_to_meta = []
        for i, meta in enumerate(metadata):
            num_valid = int(masks[i].sum())
            for p_idx in range(num_valid):
                flat_to_meta.append((meta['canonical_id'], p_idx))
                
        database_embs = valid_embs
        
    elif args.search_target == 'page':
        print("Target: Mean-Pooled Page Embeddings")
        raw_embs = store['panel_embeddings'][:]
        masks = store['panel_masks'][:]
        
        print("Computing page-level strip embeddings...")
        strip_embs = []
        for i in range(raw_embs.shape[0]):
            valid_e = raw_embs[i][masks[i]]
            if len(valid_e) > 0:
                strip_embs.append(np.mean(valid_e, axis=0))
            else:
                strip_embs.append(np.zeros(raw_embs.shape[2], dtype=np.float32))
                
        database_embs = np.stack(strip_embs)
        
        with open(args.metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        flat_to_meta = [(m['canonical_id'], None) for m in metadata]
        
    else:
        print("Invalid search target.")
        return

    print(f"Database shape: {database_embs.shape}")

    # 4. Search
    print("Computing similarities...")
    similarities = compute_similarity(query_emb_np, database_embs)
    
    # Get Top K
    top_k = args.top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 5. Display Results
    print(f"\n{'='*60}")
    print(f"Top {top_k} Results for query:")
    print(f"{ '='*60}")
    
    for rank, idx in enumerate(top_indices):
        sim = similarities[idx]
        cid, p_idx = flat_to_meta[idx]
        
        # Parse canonical_id to get cleaner names
        try:
            path_obj = Path(cid)
            comic_name = path_obj.parent.name
            page_name = path_obj.stem
        except:
            comic_name = "Unknown"
            page_name = cid
            
        target_str = f"Panel {p_idx+1}" if p_idx is not None else "Entire Page"
        
        print(f"Rank {rank+1} (Score: {sim:.4f})")
        print(f"  Comic: {comic_name}")
        print(f"  Page:  {page_name}")
        print(f"  Focus: {target_str}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 Semantic Search Engine")
    
    # Query
    parser.add_argument('--text_query', type=str, help="Text to search for")
    parser.add_argument('--image_query', type=str, help="Path to image to search for")
    
    # Database
    parser.add_argument('--zarr', type=str, default="E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr")
    parser.add_argument('--metadata', type=str, default="E:/Comic_Analysis_Results_v2/stage4_metadata.json")
    parser.add_argument('--stage3_checkpoint', type=str, default="checkpoints/stage3/best_model.pt")
    
    # Search Params
    parser.add_argument('--search_target', type=str, choices=['panel', 'page'], default='panel',
                        help="Search 'panel' (contextualized) or 'page' (strip embedding)")
    parser.add_argument('--top_k', type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    main(args)
