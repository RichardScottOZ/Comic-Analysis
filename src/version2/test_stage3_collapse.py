#!/usr/bin/env python3
"""
Stage 3 Collapse Diagnosis Tool
Tests if the model has collapsed to 'Page Style' (all panels on a page are identical)
or if it has learned 'Panel Content' (panels distinct from each other).
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os
import csv
from pathlib import Path
from tqdm import tqdm
from stage3_panel_features_framework import PanelFeatureExtractor
from stage3_dataset import Stage3PanelDataset, collate_stage3

# Re-use normalization/bridging from training script
def normalize_key(cid):
    prefixes = ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/", "CalibreComics_extracted\\\\", "amazon/"]
    for p in prefixes:
        if cid.startswith(p):
            cid = cid.replace(p, "")
    res = cid.lower()
    for ext in ['.jpg', '.png', '.jpeg']:
        res = res.replace(ext, '')
    return res.replace('/', '_').replace('\\', '_').strip()

def build_json_map(s3_manifest_path):
    print("Building JSON ID Map...")
    suffix_map = {}
    with open(s3_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            if "__MACOSX" in cid: continue
            parts = cid.split('/')
            # Add all suffixes to map
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in suffix_map:
                    suffix_map[suffix] = cid
            filename = parts[-1]
            if filename not in suffix_map:
                suffix_map[filename] = cid
    return suffix_map

def test_collapse(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = PanelFeatureExtractor(
        visual_backbone=args.visual_backbone,
        visual_fusion=args.visual_fusion,
        feature_dim=args.feature_dim
    ).to(device)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', '?')} (Loss: {checkpoint.get('val_loss', '?')})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 2. Setup Dataset (Mini validation set)
    print("Setting up dataset...")
    
    # Load Image Map
    image_map = {}
    master_ids = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = row['canonical_id']
            image_map[mid] = row['absolute_image_path']
            master_ids.append(mid)

    # Build Bridge
    json_map = build_json_map(args.s3_manifest)
    bridged_map = {}
    for mid in master_ids:
        filename = Path(image_map[mid]).name
        calibre_id = json_map.get(filename) or json_map.get(mid)
        if calibre_id:
            bridged_map[normalize_key(mid)] = calibre_id

    dataset = Stage3PanelDataset(
        image_map=image_map,
        json_map=bridged_map,
        json_root=args.json_root,
        pss_labels_path=args.val_pss_labels,
        max_panels_per_page=8,
        limit=50  # Only need a few pages to test collapse
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_stage3)

    # 3. Run Tests
    print("\n" + "="*50)
    print("RUNNING COLLAPSE DIAGNOSTICS")
    print("="*50)

    sims_intra_page = []
    sims_inter_page = []
    text_retrieval_acc = []
    
    # Store first page embedding for inter-page comparison
    first_page_panels = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Testing")):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Flatten dimensions for model forward pass (B, N, ...) -> (B*N, ...)
            B, N = batch['panel_mask'].shape
            model_batch = {
                'images': batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
                'input_ids': batch['input_ids'].view(B*N, -1),
                'attention_mask': batch['attention_mask'].view(B*N, -1),
                'comp_feats': batch['comp_feats'].view(B*N, -1),
                'modality_mask': batch['modality_mask'].view(B*N, 3)
            }
            
            # Get Embeddings
            # We want FULL embeddings (Vision + Text + Comp)
            # Output will be (B*N, D)
            flat_embeddings = model(model_batch) 
            # Reshape back to (B, N, D)
            embeddings = flat_embeddings.view(B, N, -1)
            
            # Mask for valid panels
            B, N, D = embeddings.shape
            mask = batch['panel_mask'][0] # [N]
            valid_emb = embeddings[0][mask] # [num_valid, D]
            num_valid = valid_emb.shape[0]
            
            # --- DATA SANITY CHECK ---
            if i < 3: # Check first 3 batches
                print(f"\n--- Batch {i} Sanity Check ---")
                
                # Check 1: Image Variance
                valid_imgs = batch['images'].view(-1, 3, 224, 224)[mask]
                if valid_imgs.shape[0] > 1:
                    img_diff = (valid_imgs[0] - valid_imgs[1]).abs().mean().item()
                    print(f"Image Pairwise Diff: {img_diff:.6f} (Should be > 0.0)")
                
                # Check 2: Text Diversity & Content
                valid_ids = batch['input_ids'].view(-1, batch['input_ids'].shape[-1])[mask]
                if valid_ids.shape[0] > 0:
                    print("\n[Decoded Text Samples]")
                    for idx in range(min(3, valid_ids.shape[0])):
                        txt = dataset.tokenizer.decode(valid_ids[idx], skip_special_tokens=True)
                        print(f"  Panel {idx}: {txt[:100]}..." if len(txt) > 100 else f"  Panel {idx}: {txt}")
                
                if valid_ids.shape[0] > 1:
                    txt_diff = (valid_ids[0].float() - valid_ids[1].float()).abs().sum().item()
                    print(f"\nText ID Diff: {txt_diff} (Should be > 0)")
                    if txt_diff == 0:
                        print("CRITICAL WARNING: Text inputs are identical!")

            # --- Test 1: Intra-Page Similarity (Distinctness) ---
            # Measure avg cosine sim between all pairs on THIS page
            # Normalize first
            valid_emb_norm = F.normalize(valid_emb, dim=-1)
            sim_matrix = torch.mm(valid_emb_norm, valid_emb_norm.t()) # [V, V]
            
            # Get upper triangle (excluding self-similarity 1.0)
            triu_indices = torch.triu_indices(num_valid, num_valid, offset=1)
            if triu_indices.size(1) > 0:
                pairwise_sims = sim_matrix[triu_indices[0], triu_indices[1]]
                avg_sim = pairwise_sims.mean().item()
                sims_intra_page.append(avg_sim)
            
            # --- Test 2: Text-Image Retrieval (Alignment) ---
            # Can the vision encoder find the correct text for the panel?
            # Encode images independently
            imgs = batch['images'].view(-1, 3, 224, 224)[mask]
            vis_emb = F.normalize(model.encode_image_only(imgs), dim=-1)
            
            # Encode texts independently
            input_ids = batch['input_ids'].view(-1, batch['input_ids'].shape[-1])[mask]
            attn_mask = batch['attention_mask'].view(-1, batch['attention_mask'].shape[-1])[mask]
            txt_emb = F.normalize(model.encode_text_only(input_ids, attn_mask), dim=-1)
            
            # Similarity: Vision vs Text
            # Row i is Image i, Col j is Text j
            scores = torch.mm(vis_emb, txt_emb.t()) # [V, V]
            targets = torch.arange(num_valid).to(device)
            preds = scores.argmax(dim=-1)
            
            acc = (preds == targets).float().mean().item()
            text_retrieval_acc.append(acc)

            # --- Test 3: Inter-Page Similarity ---
            # Compare first panel of this page vs first panel of first page
            if first_page_panels is None:
                first_page_panels = valid_emb_norm[0].unsqueeze(0) # [1, D]
            else:
                current_panel = valid_emb_norm[0].unsqueeze(0)
                inter_sim = torch.mm(first_page_panels, current_panel.t()).item()
                sims_inter_page.append(inter_sim)

    # 4. Report Results
    avg_intra = sum(sims_intra_page) / len(sims_intra_page)
    avg_inter = sum(sims_inter_page) / len(sims_inter_page) if sims_inter_page else 0.0
    avg_acc = sum(text_retrieval_acc) / len(text_retrieval_acc)

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Num Pages Tested: {len(sims_intra_page)}")
    print(f"\n1. DISTINCTNESS (Lower is better)")
    print(f"   Avg Intra-Page Similarity: {avg_intra:.4f}")
    print("   (>0.90 = Collapse: All panels on page look identical)")
    print("   (<0.80 = Good: Panels are distinct)")
    
    print(f"\n2. DISCRIMINATION (Lower is better)")
    print(f"   Avg Inter-Page Similarity: {avg_inter:.4f}")
    print("   (Should be lower than Intra-Page, but not 1.0)")
    
    print(f"\n3. ALIGNMENT (Higher is better)")
    print(f"   Image-Text Retrieval Acc:  {avg_acc:.4f} ({avg_acc*100:.1f}%)")
    print("   (Random chance depends on panels per page, usually ~15-20%)")
    
    print("\nCONCLUSION:")
    if avg_intra > 0.95:
        print("❌ MODEL HAS COLLAPSED to Page Style.")
        print("   The embeddings represent the 'Page' but not individual 'Panels'.")
    elif avg_acc < 0.30:
        print("⚠️ WEAK ALIGNMENT.")
        print("   Model distinguishes panels but doesn't link image to text well.")
    else:
        print("✅ MODEL LOOKS HEALTHY.")
        print("   Panels are distinct and aligned with text.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--s3_manifest', required=True)
    parser.add_argument('--json_root', required=True)
    parser.add_argument('--val_pss_labels', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visual_backbone', type=str, default='both')
    parser.add_argument('--visual_fusion', type=str, default='attention')
    parser.add_argument('--feature_dim', type=int, default=512)
    
    args = parser.parse_args()
    test_collapse(args)
