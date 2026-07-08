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
from torch.utils.data import Dataset, DataLoader, Subset

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
    total_samples = len(dataset)
    print(f"Total sequences to process: {total_samples}")

    # 3. Setup Output Zarr (with optional resume)
    progress_file = args.output_zarr.rstrip('/\\') + '.progress.json'
    meta_partial_file = args.output_zarr.rstrip('/\\') + '.metadata_partial.jsonl'
    start_idx = 0

    if args.resume and os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        start_idx = progress.get('current_idx', 0)
        print(f"Resuming from index {start_idx} / {total_samples}")
        overwrite_zarr = False
    else:
        overwrite_zarr = True
        # Clear any stale partial metadata
        if os.path.exists(meta_partial_file):
            os.remove(meta_partial_file)

    out_dir = os.path.dirname(os.path.abspath(args.output_zarr))
    os.makedirs(out_dir, exist_ok=True)

    store = zarr.DirectoryStore(args.output_zarr)
    root = zarr.group(store=store, overwrite=overwrite_zarr)

    max_panels = dataset.panel_embs.shape[1]
    feature_dim = dataset.panel_embs.shape[2]

    if overwrite_zarr:
        context_array = root.zeros('contextualized_panels',
                                   shape=(total_samples, max_panels, feature_dim),
                                   chunks=(100, max_panels, feature_dim), dtype='float32')
        strip_array = root.zeros('strip_embeddings',
                                 shape=(total_samples, feature_dim),
                                 chunks=(1000, feature_dim), dtype='float32')
        mask_array = root.zeros('panel_masks',
                                shape=(total_samples, max_panels),
                                chunks=(100, max_panels), dtype='bool')
    else:
        context_array = root['contextualized_panels']
        strip_array = root['strip_embeddings']
        mask_array = root['panel_masks']

    # Subset dataset to unprocessed rows
    if start_idx > 0:
        active_dataset = Subset(dataset, list(range(start_idx, total_samples)))
        dataloader = DataLoader(active_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_inference)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_inference)

    # 4. Generate Embeddings
    print("Generating Stage 4 Embeddings...")
    current_idx = start_idx
    meta_out = open(meta_partial_file, 'a', encoding='utf-8')

    try:
      with torch.no_grad():
        for batch in tqdm(dataloader, initial=start_idx // max(args.batch_size, 1),
                          total=(total_samples + args.batch_size - 1) // args.batch_size):
            panel_embeddings = batch['panel_embeddings'].to(device)
            panel_mask = batch['panel_mask'].to(device)
            
            outputs = model(panel_embeddings, panel_mask)
            
            context_np = outputs['contextualized_panels'].cpu().numpy()
            strip_np = outputs['strip_embedding'].cpu().numpy()
            mask_np = panel_mask.cpu().numpy().astype(bool)
            
            batch_size = context_np.shape[0]
            
            context_array[current_idx : current_idx + batch_size] = context_np
            strip_array[current_idx : current_idx + batch_size] = strip_np
            mask_array[current_idx : current_idx + batch_size] = mask_np
            
            for meta in batch['metadata']:
                meta_out.write(json.dumps(meta) + '\n')
            
            current_idx += batch_size

            # Save progress every 1000 rows
            if current_idx % 1000 < args.batch_size:
                with open(progress_file, 'w') as pf:
                    json.dump({'current_idx': current_idx}, pf)

    finally:
        meta_out.close()
        with open(progress_file, 'w') as pf:
            json.dump({'current_idx': current_idx}, pf)

    # 5. Save Metadata (consolidate from partial jsonl)
    print(f"Saving metadata to {args.output_metadata}...")
    metadata_list = []
    with open(meta_partial_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                metadata_list.append(json.loads(line))
    with open(args.output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

    # Clean up progress/partial files
    if os.path.exists(progress_file):
        os.remove(progress_file)
    if os.path.exists(meta_partial_file):
        os.remove(meta_partial_file)

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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume from last saved progress')
    
    args = parser.parse_args()
    main(args)
