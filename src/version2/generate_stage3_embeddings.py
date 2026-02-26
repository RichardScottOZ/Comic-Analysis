#!/usr/bin/env python3
"""
Generate Stage 3 Embeddings for Stage 4 Training

This script loads the trained Stage 3 PanelFeatureExtractor model and processes
the dataset to pre-compute panel embeddings. These embeddings are saved to a Zarr
store, along with a metadata JSON file, which serve as the input for Stage 4.

Features:
- Manifest-driven loading (bridges Master and Calibre manifests)
- Batch processing with DataLoader
- Zarr storage for fast, chunked reading in Stage 4
- Generates metadata.json mapping sequence indices to book/page IDs
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import zarr
import numpy as np
import csv

from stage3_panel_features_framework import PanelFeatureExtractor
from stage3_dataset import Stage3PanelDataset, collate_stage3

# --- Manifest Bridging (Reused from train_stage3.py) ---

def normalize_key(cid):
    prefixes = ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/", "CalibreComics_extracted\\", "amazon/"]
    for p in prefixes:
        if cid.startswith(p):
            cid = cid.replace(p, "")
    res = cid.lower()
    for ext in ['.jpg', '.png', '.jpeg']:
        res = res.replace(ext, '')
    return res.replace('/', '_').replace('\\', '_').strip()

def build_json_map(s3_manifest_path):
    print("Building JSON ID Map (Suffix Strategy).")
    suffix_map = {}
    with open(s3_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            if "__MACOSX" in cid: continue
            
            parts = cid.split('/')
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in suffix_map:
                    suffix_map[suffix] = cid
            
            filename = parts[-1]
            if filename not in suffix_map:
                suffix_map[filename] = cid
    return suffix_map

# ---------------------------------------------------------

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = PanelFeatureExtractor(
            visual_backbone=args.visual_backbone,
            visual_fusion=args.visual_fusion,
            feature_dim=args.feature_dim,
            freeze_backbones=args.freeze_backbones
        ).to(device)
        
        # Support both formats
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Build Dataset Maps
    json_map = build_json_map(args.s3_manifest)
    
    print(f"Loading Master Manifest: {args.manifest}")
    image_map = {}
    json_map_final = {}
    
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            master_id = row['canonical_id']
            local_path = row['absolute_image_path']
            filename = Path(local_path).name
            
            calibre_id = suffix_map.get(filename)
            if not calibre_id: calibre_id = suffix_map.get(master_id)
            if not calibre_id and filename.endswith('.jpg.png'):
                calibre_id = suffix_map.get(filename.replace('.jpg.png', '.jpg'))
                
            if calibre_id:
                key = normalize_key(master_id)
                image_map[master_id] = local_path
                json_map_final[key] = calibre_id

    # 3. Initialize Dataset
    dataset = Stage3PanelDataset(
        image_map=image_map,
        json_map=json_map_final,
        json_root=args.json_root,
        pss_labels_path=args.pss_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        limit=args.limit
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # Must be False to align metadata correctly
        num_workers=args.num_workers, 
        collate_fn=collate_stage3, 
        pin_memory=True
    )

    total_samples = len(dataset)
    if total_samples == 0:
        print("No samples to process.")
        return

    # 4. Prepare Zarr Store and Metadata
    os.makedirs(os.path.dirname(args.output_zarr), exist_ok=True)
    
    # We use zarr directly instead of xarray for simpler appending
    store = zarr.DirectoryStore(args.output_zarr)
    root = zarr.group(store=store, overwrite=True)
    
    # Shape: (Total Pages, Max Panels, Feature Dim)
    emb_array = root.zeros('panel_embeddings', shape=(total_samples, args.max_panels, args.feature_dim), 
                           chunks=(100, args.max_panels, args.feature_dim), dtype='float32')
    # Shape: (Total Pages, Max Panels) - Boolean mask for valid panels
    mask_array = root.zeros('panel_masks', shape=(total_samples, args.max_panels), 
                            chunks=(100, args.max_panels), dtype='bool')
    
    metadata_list = []
    current_idx = 0

    # 5. Extract Embeddings
    print(f"\nExtracting embeddings for {total_samples} pages...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            # Move to device
            model_batch = {
                'images': batch['images'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'comp_feats': batch['comp_feats'].to(device),
                'modality_mask': batch['modality_mask'].to(device)
            }
            
            # The model expects flattened inputs: (B*N, ...)
            B, N = batch['panel_mask'].shape
            
            flat_batch = {
                'images': model_batch['images'].view(B*N, 3, args.image_size, args.image_size),
                'input_ids': model_batch['input_ids'].view(B*N, -1),
                'attention_mask': model_batch['attention_mask'].view(B*N, -1),
                'comp_feats': model_batch['comp_feats'].view(B*N, -1),
                'modality_mask': model_batch['modality_mask'].view(B*N, 3)
            }
            
            # Forward pass -> (B*N, D)
            flat_embeddings = model(flat_batch)
            
            # Reshape back to (B, N, D)
            panel_embeddings = flat_embeddings.view(B, N, -1)
            
            # Convert to numpy
            emb_np = panel_embeddings.cpu().numpy()
            mask_np = batch['panel_mask'].cpu().numpy()
            
            batch_size_actual = emb_np.shape[0]
            
            # Write to Zarr
            emb_array[current_idx : current_idx + batch_size_actual] = emb_np
            mask_array[current_idx : current_idx + batch_size_actual] = mask_np
            
            # Collect Metadata
            for i, meta in enumerate(batch['metadata']):
                metadata_list.append({
                    "sequence_index": current_idx + i,
                    "canonical_id": meta['canonical_id'],
                    "num_panels": meta['num_panels']
                })
                
            current_idx += batch_size_actual

    # 6. Save Metadata
    print(f"\nSaving metadata to {args.output_metadata}...")
    with open(args.output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)
        
    print("✅ Stage 3 Embedding Generation Complete!")
    print(f"Embeddings saved to: {args.output_zarr}")
    print(f"Metadata saved to: {args.output_metadata}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 3 Embeddings for Stage 4")
    
    # Input Data
    parser.add_argument('--manifest', type=str, required=True, help="Master Manifest (Local Paths)")
    parser.add_argument('--s3_manifest', type=str, required=True, help="Calibre Manifest (IDs)")
    parser.add_argument('--json_root', type=str, required=True, help="Root of Stage 3 JSONs")
    parser.add_argument('--pss_labels', type=str, required=True, help="Path to PSS Labels JSON")
    
    # Model Configuration
    parser.add_argument('--checkpoint', type=str, default="checkpoints/stage3/best_model.pt", help="Path to trained Stage 3 model")
    parser.add_argument('--visual_backbone', type=str, default='both', choices=['siglip', 'resnet', 'both'])
    parser.add_argument('--visual_fusion', type=str, default='attention')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_panels', type=int, default=16)
    
    # Execution
    parser.add_argument('--output_zarr', type=str, default="stage3_embeddings.zarr", help="Output Zarr store path")
    parser.add_argument('--output_metadata', type=str, default="stage3_metadata.json", help="Output metadata JSON path")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    
    # Pass suffix_map builder into global scope to avoid undefined error in main
    suffix_map = build_json_map(args.s3_manifest)
    
    main(args)
