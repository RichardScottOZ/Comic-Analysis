#!/usr/bin/env python3
"""
Generate Stage 3 Embeddings — VLM-Backed

Identical to generate_stage3_embeddings.py but uses Stage3PanelDatasetVLM
(VLM JSON output) instead of Stage3PanelDataset (old OCR panel JSONs).

Loads the trained Stage 3 model (best_model_vlm.pt) and processes all pages
to produce a Zarr embedding store for Stage 4 training.

Output:
    {output_zarr}/panel_embeddings  shape: (N_pages, 16, 512)  float32
    {output_zarr}/panel_masks       shape: (N_pages, 16)       bool
    {output_metadata}               JSON list with sequence_index, canonical_id,
                                    num_panels, overall_summary per page

Usage:
    python generate_stage3_embeddings_vlm.py \
        --manifest manifests/master_manifest_20251229.csv \
        --vlm_cache_dir /data/vlm_cache \
        --pss_labels pss_labels_v1.json \
        --checkpoint checkpoints/stage3_vlm/best_model_vlm.pt \
        --output_zarr stage3_embeddings_vlm.zarr \
        --output_metadata stage3_metadata_vlm.json
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
from stage3_dataset_vlm import Stage3PanelDatasetVLM, collate_stage3


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = PanelFeatureExtractor(
            visual_backbone=args.visual_backbone,
            visual_fusion=args.visual_fusion,
            feature_dim=args.feature_dim,
            freeze_backbones=args.freeze_backbones
        ).to(device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Build image_map from master manifest
    print(f"Loading master manifest: {args.manifest}")
    image_map = {}
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            image_map[row['canonical_id']] = row['absolute_image_path']
    print(f"  {len(image_map):,} image paths loaded")

    # 3. Build dataset — narrative pages only (model trained on these)
    dataset = Stage3PanelDatasetVLM(
        image_map=image_map,
        vlm_cache_dir=args.vlm_cache_dir,
        pss_labels_path=args.pss_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        only_narrative=True,
        limit=args.limit
    )

    total_samples = len(dataset)
    if total_samples == 0:
        print("No samples to process.")
        return

    # 4. Resume detection — check for existing progress file
    progress_file = Path(args.output_zarr + '.progress.json')
    start_idx = 0
    metadata_list = []

    if progress_file.exists() and Path(args.output_zarr).exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_idx', 0)
        if start_idx > 0:
            print(f"▶️  Resuming from index {start_idx:,} / {total_samples:,}")
            if Path(args.output_metadata).exists():
                with open(args.output_metadata, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                print(f"   Loaded {len(metadata_list):,} existing metadata entries.")

    # 5. Create or reopen Zarr store
    store = zarr.DirectoryStore(args.output_zarr)
    if start_idx == 0:
        root_grp = zarr.group(store=store, overwrite=True)
        emb_array = root_grp.zeros(
            'panel_embeddings',
            shape=(total_samples, args.max_panels, args.feature_dim),
            chunks=(100, args.max_panels, args.feature_dim),
            dtype='float32'
        )
        mask_array = root_grp.zeros(
            'panel_masks',
            shape=(total_samples, args.max_panels),
            chunks=(100, args.max_panels),
            dtype='bool'
        )
    else:
        root_grp = zarr.open_group(store=store, mode='r+')
        emb_array = root_grp['panel_embeddings']
        mask_array = root_grp['panel_masks']

    # 6. Build dataloader from remaining samples
    if start_idx > 0:
        from torch.utils.data import Subset
        remaining = Subset(dataset, range(start_idx, total_samples))
    else:
        remaining = dataset

    print(f"Processing {len(remaining):,} pages (total={total_samples:,})...")
    dataloader = DataLoader(
        remaining,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_stage3,
        pin_memory=True
    )

    current_idx = start_idx

    # 7. Extract embeddings
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            model_batch = {
                'images': batch['images'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'comp_feats': batch['comp_feats'].to(device),
                'modality_mask': batch['modality_mask'].to(device),
            }

            B, N = batch['panel_mask'].shape
            flat_batch = {
                'images': model_batch['images'].view(B * N, 3, args.image_size, args.image_size),
                'input_ids': model_batch['input_ids'].view(B * N, -1),
                'attention_mask': model_batch['attention_mask'].view(B * N, -1),
                'comp_feats': model_batch['comp_feats'].view(B * N, -1),
                'modality_mask': model_batch['modality_mask'].view(B * N, 3),
            }

            flat_embeddings = model(flat_batch)
            panel_embeddings = flat_embeddings.view(B, N, -1)

            emb_np = panel_embeddings.cpu().numpy()
            mask_np = batch['panel_mask'].cpu().numpy()
            batch_size_actual = emb_np.shape[0]

            emb_array[current_idx: current_idx + batch_size_actual] = emb_np
            mask_array[current_idx: current_idx + batch_size_actual] = mask_np

            for i, meta in enumerate(batch['metadata']):
                metadata_list.append({
                    "sequence_index": current_idx + i,
                    "canonical_id": meta['canonical_id'],
                    "num_panels": meta['num_panels'],
                    "overall_summary": meta.get('overall_summary', ''),
                })

            current_idx += batch_size_actual

            # Save progress after every batch — crash loses at most one batch
            with open(progress_file, 'w') as f:
                json.dump({'last_idx': current_idx, 'total': total_samples}, f)

    # 8. Save metadata and clean up progress file
    print(f"Saving metadata: {args.output_metadata}")
    with open(args.output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

    progress_file.unlink(missing_ok=True)  # clean up on successful completion

    print(
        f"\n✅ Embedding generation complete!\n"
        f"   Embeddings : {args.output_zarr}  shape=({total_samples}, {args.max_panels}, {args.feature_dim})\n"
        f"   Metadata   : {args.output_metadata}  ({len(metadata_list):,} entries)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Stage 3 Embeddings — VLM-Backed")

    # Data
    parser.add_argument('--manifest', type=str, required=True,
                        help="Master manifest CSV (canonical_id + absolute_image_path)")
    parser.add_argument('--vlm_cache_dir', type=str, required=True,
                        help="Local root dir of VLM JSONs from sync_vlm_cache.py")
    parser.add_argument('--pss_labels', type=str, required=True,
                        help="PSS labels JSON — used to enumerate all pages (narrative filter off)")

    # Model
    parser.add_argument('--checkpoint', type=str, default="checkpoints/stage3_vlm/best_model_vlm.pt")
    parser.add_argument('--visual_backbone', type=str, default='both', choices=['siglip', 'resnet', 'both'])
    parser.add_argument('--visual_fusion', type=str, default='attention')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_panels', type=int, default=16)

    # Output
    parser.add_argument('--output_zarr', type=str, default="stage3_embeddings_vlm.zarr")
    parser.add_argument('--output_metadata', type=str, default="stage3_metadata_vlm.json")

    # Execution
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit pages (for testing)")

    args = parser.parse_args()
    main(args)
