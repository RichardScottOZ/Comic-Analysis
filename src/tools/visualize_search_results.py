#!/usr/bin/env python3
"""
Visualize Search Results
Runs a semantic search and visually crops/saves the top matching panels.
Also outputs a `search_results.json` so downstream scripts know the exact Zarr indices.
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
import csv
import sys

# Add src/version2 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'version2'))
from stage3_panel_features_framework import PanelFeatureExtractor
from search_stage5 import compute_similarity

def build_suffix_map(manifest_path):
    print("Indexing Master Manifest...")
    suffix_map = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cid = row['canonical_id']
            img_path = row['absolute_image_path']
            
            # Map by filename and exact ID
            suffix_map[Path(img_path).name] = img_path
            suffix_map[cid] = img_path
            suffix_map[Path(img_path).stem] = img_path
    return suffix_map

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine folder name
    if args.output_folder_name:
        folder_name = args.output_folder_name
    elif args.text_query:
        folder_name = args.text_query.replace(' ', '_')
    elif args.image_query:
        folder_name = os.path.splitext(os.path.basename(args.image_query))[0]
    elif args.query_canonical_id:
        folder_name = f"recursive_{Path(args.query_canonical_id).stem}_p{args.query_panel_idx}"
    else:
        folder_name = "search_results"
        
    out_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Database First
    print("Loading Zarr Database...")
    store = zarr.open(args.zarr, mode='r')
    raw_embs = store['panel_embeddings'][:]
    masks = store['panel_masks'][:]
    valid_embs = raw_embs[masks]

    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    flat_to_meta = []
    for i, meta in enumerate(metadata):
        for p_idx in range(int(masks[i].sum())):
            flat_to_meta.append((meta['canonical_id'], p_idx))

    # 2. Encode Query
    query_emb_np = None
    
    # A. Exact Zarr Match (First Principles)
    if args.query_canonical_id is not None and args.query_panel_idx is not None:
        print(f"Using exact Zarr embedding for {args.query_canonical_id} Panel {args.query_panel_idx}")
        for idx, (cid, p_idx) in enumerate(flat_to_meta):
            if cid == args.query_canonical_id and p_idx == args.query_panel_idx:
                query_emb_np = valid_embs[idx].reshape(1, -1)
                break
        if query_emb_np is None:
            print("Error: Could not find specified canonical_id and panel_idx in database.")
            return

    # B. Text or Raw Image Query
    if query_emb_np is None:
        print(f"Loading Model...")
        model = PanelFeatureExtractor(visual_backbone='both', visual_fusion='attention', feature_dim=512).to(device)
        checkpoint = torch.load(args.stage3_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()

        with torch.no_grad():
            if args.text_query:
                print(f"Encoding Text Query: '{args.text_query}'")
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                text_enc = tokenizer([args.text_query], return_tensors='pt', padding=True, truncation=True, max_length=128)
                query_emb = model.encode_text_only(text_enc['input_ids'].to(device), text_enc['attention_mask'].to(device))
                query_emb_np = query_emb.cpu().numpy()
            elif args.image_query:
                print(f"Encoding Image Query via Model: '{args.image_query}'")
                image = Image.open(args.image_query).convert('RGB')
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                text_enc = tokenizer([""], return_tensors='pt', padding='max_length', max_length=128)
                comp_feats = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5]]).to(device)
                modality_mask = torch.tensor([[1.0, 0.0, 1.0]]).to(device)
                
                batch = {
                    'images': img_tensor,
                    'input_ids': text_enc['input_ids'].to(device),
                    'attention_mask': text_enc['attention_mask'].to(device),
                    'comp_feats': comp_feats,
                    'modality_mask': modality_mask
                }
                query_emb = model(batch)
                query_emb_np = query_emb.cpu().numpy()

    if query_emb_np is None:
         print("Error: No valid query provided.")
         return

    # 3. Search Database
    print("Computing similarities...")
    similarities = compute_similarity(query_emb_np, valid_embs)
    top_indices = np.argsort(similarities)[::-1][:args.top_k]

    # 4. Resolve Paths, Crop, and Record Results
    print("Visualizing Results...")
    master_map = build_suffix_map(args.master_manifest)
    
    results_record = []

    for rank, idx in enumerate(top_indices):
        sim = similarities[idx]
        calibre_id, p_idx = flat_to_meta[idx]
        
        # Record data
        result_data = {
            "rank": rank + 1,
            "score": float(sim),
            "canonical_id": calibre_id,
            "panel_idx": int(p_idx),
            "image_file": None,
            "error": None
        }

        # Find Image Path
        parts = calibre_id.split('/')
        img_path = None
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            if suffix in master_map: img_path = master_map[suffix]; break
            if parts[-1] in master_map: img_path = master_map[parts[-1]]; break
            
        if not img_path or not os.path.exists(img_path):
            result_data["error"] = "Missing source image"
            results_record.append(result_data)
            continue

        # Print detailed info to console
        comic_name = Path(img_path).parent.name
        page_name = Path(img_path).stem
        print(f"Rank {rank+1} (Score: {sim:.3f}) | Comic: {comic_name} | Page: {page_name} | Panel: {p_idx}")

        # Find JSON Path for BBox
        json_path = os.path.join(args.json_root, calibre_id.replace('/', os.sep) + ".json")
        if not os.path.exists(json_path):
            result_data["error"] = "Missing JSON"
            results_record.append(result_data)
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f: jdata = json.load(f)
            panels = jdata.get('panels', [])
            if p_idx >= len(panels): 
                result_data["error"] = "Panel index out of bounds"
                results_record.append(result_data)
                continue
            
            bbox = panels[p_idx]['bbox']
            img = Image.open(img_path).convert('RGB')
            x, y, w, h = bbox
            crop = img.crop((x, y, x+w, y+h))
            
            # Construct informative filename
            safe_comic_name = comic_name.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
            safe_page_name = page_name.replace(' ', '_')[:30]
            
            filename = f"r{rank+1}_{sim:.3f}_{safe_comic_name}_{safe_page_name}_p{p_idx}.jpg"
            save_path = os.path.join(out_dir, filename)
            
            crop.save(save_path)
            result_data["image_file"] = filename
            
        except Exception as e:
            result_data["error"] = str(e)
            
        results_record.append(result_data)

    # Save absolute ground truth metadata for downstream scripts
    record_path = os.path.join(out_dir, "search_results.json")
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(results_record, f, indent=2)

    print(f"\nDone. Saved {len(results_record)} results to folder: {out_dir}")
    print(f"Metadata recorded in: {record_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Query types
    parser.add_argument('--text_query', type=str, help="Text query string")
    parser.add_argument('--image_query', type=str, help="Path to image to query")
    parser.add_argument('--query_canonical_id', type=str, help="Exact canonical_id from database")
    parser.add_argument('--query_panel_idx', type=int, help="Exact panel index from database")
    
    # Config
    parser.add_argument('--output_folder_name', type=str, help="Override default output folder name")
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--zarr', default="E:/Comic_Analysis_Results_v2/stage3_embeddings.zarr")
    parser.add_argument('--metadata', default="E:/Comic_Analysis_Results_v2/stage3_metadata.json")
    parser.add_argument('--stage3_checkpoint', default="checkpoints/stage3/best_model.pt")
    parser.add_argument('--master_manifest', default="manifests/master_manifest_20251229.csv")
    parser.add_argument('--json_root', default="E:/Comic_Analysis_Results_v2/stage3_json")
    parser.add_argument('--output_dir', default="local_test_output/search_results")
    args = parser.parse_args()
    
    if not any([args.text_query, args.image_query, (args.query_canonical_id and args.query_panel_idx is not None)]):
        parser.error("Must provide --text_query, --image_query, OR (--query_canonical_id AND --query_panel_idx)")
        
    main(args)