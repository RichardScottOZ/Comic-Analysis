#!/usr/bin/env python3
"""
PSS Classifier for Zarr Stores (High Performance Batched version)
Loads precomputed Zarr embeddings and saves predictions into the 'prediction' variable.
Uses Batched Sequence Inference to maximize GPU throughput.
"""

import os
import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import argparse
from transformers import BertConfig, BertModel
from tqdm import tqdm

# Class Mapping (From PSSDataset)
CLASS_NAMES = ["advertisement", "cover", "story", "textstory", "first-page", "credits", "art", "text", "back_cover"]

class BookBERT(nn.Module):
    def __init__(self, feature_dim, bert_input, num_hidden_layers, num_attention_heads, 
                 positional_embeddings, num_classes, hidden_dim, dropout_p):
        super().__init__()
        config = BertConfig(
            hidden_size=bert_input,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=bert_input * 4,
            hidden_dropout_prob=dropout_p,
            attention_probs_dropout_prob=dropout_p,
            max_position_embeddings=1024
        )
        self.bert_encoder = BertModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(bert_input, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 4, num_classes)
        )

class BookBERTMultimodal2(BookBERT):
    def __init__(self, textual_feature_dim, visual_feature_dim=1152, bert_input_dim=768,
                 projection_dim=1024, num_hidden_layers=4, num_attention_heads=4,
                 positional_embeddings='absolute', num_classes=9, hidden_dim=512, dropout_p=0.3):
        
        self.dropout_p = dropout_p
        self.textual_feature_dim = textual_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.bert_input_dim = bert_input_dim
        
        super(BookBERTMultimodal2, self).__init__(
            feature_dim=visual_feature_dim,
            bert_input=bert_input_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            positional_embeddings=positional_embeddings,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_p=dropout_p
        )

        sz1 = (self.visual_feature_dim + bert_input_dim) * 2
        sz2 = sz1 // 2
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, sz1), nn.LayerNorm(sz1), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz1, sz2), nn.LayerNorm(sz2), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz2, bert_input_dim)
        )
        
        sz1_t = (self.textual_feature_dim + bert_input_dim) * 2
        sz2_t = sz1_t // 2
        self.textual_projection = nn.Sequential(
            nn.Linear(textual_feature_dim, sz1_t), nn.LayerNorm(sz1_t), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz1_t, sz2_t), nn.LayerNorm(sz2_t), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz2_t, bert_input_dim)
        )
        self.norm = nn.LayerNorm(bert_input_dim)
        self.projection = nn.Linear(visual_feature_dim, bert_input_dim)

    def forward(self, textual_features, visual_features):
        batch_size, seq_len, _ = textual_features.shape
        mask = torch.ones((batch_size, seq_len), device=textual_features.device)
        
        t_proj = self.textual_projection(textual_features)
        v_proj = self.visual_projection(visual_features)
        
        t_norm = self.norm(t_proj)
        v_norm = self.norm(v_proj)
        stacked = torch.stack([t_norm, v_norm], dim=2)
        combined = stacked.view(batch_size, seq_len * 2, -1)
        
        exp_mask = mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, seq_len * 2)
        bert_out = self.bert_encoder(inputs_embeds=combined, attention_mask=exp_mask)
        reshaped = bert_out.last_hidden_state.view(batch_size, seq_len, 2, -1)
        logits = self.classifier(reshaped[:, :, -1, :])
        return logits

def classify_zarr(zarr_path, checkpoint_path, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    ds = xr.open_zarr(zarr_path)
    total_pages = len(ds.page_id)
    print(f"Loaded Zarr with {total_pages} pages.")
    
    # 2. Initialize Model
    model = BookBERTMultimodal2(textual_feature_dim=1024, visual_feature_dim=1152, num_classes=9, hidden_dim=512, dropout_p=0.0)
    print(f"Loading weights from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("✅ Model loaded.")

    # 3. Batched Inference
    # Model max is 1024 tokens = 512 pages.
    MAX_SEQ = 512
    # Batch size = how many 512-page sequences to run in parallel
    PAGES_PER_BATCH = batch_size * MAX_SEQ 
    
    print(f"\n--- Running Inference (Processing {PAGES_PER_BATCH} pages per pass) ---")
    all_preds_list = []
    
    for start_idx in tqdm(range(0, total_pages, PAGES_PER_BATCH)):
        end_idx = min(start_idx + PAGES_PER_BATCH, total_pages)
        chunk_len = end_idx - start_idx
        
        # Load data
        vis_raw = ds['visual'].isel(page_id=slice(start_idx, end_idx)).values
        txt_raw = ds['text'].isel(page_id=slice(start_idx, end_idx)).values
        
        num_sequences = (chunk_len + MAX_SEQ - 1) // MAX_SEQ
        total_padded = num_sequences * MAX_SEQ
        
        vis_padded = np.zeros((total_padded, 1152), dtype='float32')
        txt_padded = np.zeros((total_padded, 1024), dtype='float32')
        vis_padded[:chunk_len] = vis_raw
        txt_padded[:chunk_len] = txt_raw
        
        vis_tensor = torch.from_numpy(vis_padded).view(num_sequences, MAX_SEQ, 1152).to(device)
        txt_tensor = torch.from_numpy(txt_padded).view(num_sequences, MAX_SEQ, 1024).to(device)
        
        with torch.inference_mode():
            logits = model(txt_tensor, vis_tensor)
            preds = torch.argmax(logits, dim=-1).cpu().numpy().astype('int8').flatten()
            
        preds = preds[:chunk_len]
        all_preds_list.extend(preds)

        # Write back to Zarr ONLY if save is requested
        if args.save:
            ds_pred = xr.Dataset(
                data_vars={'prediction': (['page_id'], preds)},
                coords={'page_id': np.arange(start_idx, end_idx)}
            )
            ds_pred.to_zarr(zarr_path, region={'page_id': slice(start_idx, end_idx)})

    final_preds = np.array(all_preds_list)
    page_ids = ds['ids'].values

    # 4. Report
    print("\n--- Predictions (First 25) ---")
    print(f"{'Page ID':<60} | {'Label':<15}")
    print("-" * 80)
    for i in range(min(25, len(final_preds))):
        print(f"{str(page_ids[i])[:60]:<60} | {CLASS_NAMES[final_preds[i]]:<15}")

    from collections import Counter
    counts = Counter([CLASS_NAMES[p] for p in final_preds])
    print("\n--- Class Distribution ---")
    for label, count in sorted(counts.items()):
        print(f"  {label:<15}: {count}")

    if args.save:
        print(f"\n✅ All {total_pages} predictions saved to {zarr_path}")
    else:
        print(f"\nℹ️ Dry-run complete. Use --save to write results to Zarr.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save', action='store_true', help='Actually save results to the Zarr store')
    args = parser.parse_args()
    classify_zarr(args.zarr, args.checkpoint, args.batch_size)