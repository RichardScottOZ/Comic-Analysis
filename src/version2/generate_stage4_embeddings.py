#!/usr/bin/env python3
"""
Generate Stage 4 Embeddings (Contextualized & Strip-Level)

This script loads the trained Stage 4 Sequence Model and processes the
static Stage 3 embeddings to generate context-aware panel embeddings
and aggregated strip/page-level embeddings.

Inputs:
- stage3_embeddings.zarr
- stage3_metadata.json

Outputs:
- stage4_embeddings.zarr
    - 'contextualized_panels' (N_pages, max_panels, dim)
    - 'strip_embeddings' (N_pages, dim)
    - 'panel_masks' (N_pages, max_panels)
- stage4_metadata.json (usually identical to Stage 3, passed through)
"""

import os
import argparse
import json
import torch
import zarr
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from stage4_sequence_modeling_framework import Stage4SequenceModel

class InferenceDataset(Dataset):
    """Simple dataset to stream Stage 3 Zarr rows for inference."""
    def __init__(self, embeddings_path, metadata_path):
        self.embeddings = zarr.open(embeddings_path, mode='r')
        self.panel_embs = self.embeddings['panel_embeddings']
        self.masks = self.embeddings['panel_masks']
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.num_samples = len(self.metadata)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Load single row from Zarr
        emb = self.panel_embs[idx]
        mask = self.masks[idx]
        meta = self.metadata[idx]
        return {
            'panel_embeddings': torch.from_numpy(emb).float(),
            'panel_mask': torch.from_numpy(mask).float(),
            'metadata': meta
        }

def collate_inference(batch):
    return {
        'panel_embeddings': torch.stack([b['panel_embeddings'] for b in batch]),
        'panel_mask': torch.stack([b['panel_mask'] for b in batch]),
        'metadata': [b['metadata'] for b in batch]
    }

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading Stage 4 checkpoint: {args.checkpoint}")
    try:
        model = Stage4SequenceModel(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            max_panels=args.max_panels
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Setup Dataset & DataLoader
    print(f"Loading inputs from {args.input_zarr}")
    dataset = InferenceDataset(args.input_zarr, args.input_metadata)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_inference
    )
    
    total_samples = len(dataset)
    print(f"Total sequences to process: {total_samples}")

    # 3. Setup Output Zarr
    os.makedirs(os.path.dirname(args.output_zarr), exist_ok=True)
    store = zarr.DirectoryStore(args.output_zarr)
    root = zarr.group(store=store, overwrite=True)
    
    # We copy the shape config from the input dataset
    max_panels = dataset.panel_embs.shape[1]
    feature_dim = dataset.panel_embs.shape[2]
    
    context_array = root.zeros('contextualized_panels', 
                               shape=(total_samples, max_panels, feature_dim), 
                               chunks=(100, max_panels, feature_dim), dtype='float32')
                               
    strip_array = root.zeros('strip_embeddings', 
                             shape=(total_samples, feature_dim), 
                             chunks=(1000, feature_dim), dtype='float32')
                             
    mask_array = root.zeros('panel_masks', 
                            shape=(total_samples, max_panels), 
                            chunks=(100, max_panels), dtype='bool')

    # 4. Generate Embeddings
    print("Generating Stage 4 Embeddings...")
    current_idx = 0
    metadata_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            panel_embeddings = batch['panel_embeddings'].to(device)
            panel_mask = batch['panel_mask'].to(device)
            
            # Forward pass (just the base encoder, no task heads)
            outputs = model(panel_embeddings, panel_mask)
            
            # Extract
            context_np = outputs['contextualized_panels'].cpu().numpy()
            strip_np = outputs['strip_embedding'].cpu().numpy()
            mask_np = panel_mask.cpu().numpy().astype(bool)
            
            batch_size = context_np.shape[0]
            
            # Write to Zarr
            context_array[current_idx : current_idx + batch_size] = context_np
            strip_array[current_idx : current_idx + batch_size] = strip_np
            mask_array[current_idx : current_idx + batch_size] = mask_np
            
            # Collect metadata
            metadata_list.extend(batch['metadata'])
            
            current_idx += batch_size

    # 5. Save Metadata
    print(f"Saving metadata to {args.output_metadata}...")
    with open(args.output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)
        
    print("✅ Stage 4 Embedding Generation Complete!")
    print(f"Contextualized Panels & Strip Embeddings saved to: {args.output_zarr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Inputs
    parser.add_argument('--input_zarr', type=str, required=True, help="Path to stage3_embeddings.zarr")
    parser.add_argument('--input_metadata', type=str, required=True, help="Path to stage3_metadata.json")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/stage4/best_model.pt")
    
    # Model Config (Must match training!)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--max_panels', type=int, default=16)
    
    # Outputs
    parser.add_argument('--output_zarr', type=str, default="stage4_embeddings.zarr")
    parser.add_argument('--output_metadata', type=str, default="stage4_metadata.json")
    
    # Exec
    parser.add_argument('--batch_size', type=int, default=64) # Can be large, it's just transformer inference
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
