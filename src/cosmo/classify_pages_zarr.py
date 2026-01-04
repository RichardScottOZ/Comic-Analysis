#!/usr/bin/env python3
"""
Acid Test: Classify Pages using CoSMo v4 (BookBERT)
Loads precomputed Zarr embeddings and runs inference to predict page types.
Uses the EXACT architecture recovered from the training source code.
"""

import os
import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import argparse
from transformers import BertConfig, BertModel

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
        
        # Deep Classifier Head: matches checkpoint keys 0, 1, 2, 5, 6, 9
        self.classifier = nn.Sequential(
            nn.Linear(bert_input, hidden_dim),       # 0: [512, 768]
            nn.Linear(hidden_dim, hidden_dim // 2),  # 1: [256, 512]
            nn.LayerNorm(hidden_dim // 2),           # 2: [256]
            nn.GELU(),                               # 3 (stateless)
            nn.Dropout(dropout_p),                   # 4 (stateless)
            nn.Linear(hidden_dim // 2, hidden_dim // 4), # 5: [128, 256]
            nn.LayerNorm(hidden_dim // 4),           # 6: [128]
            nn.GELU(),                               # 7 (stateless)
            nn.Dropout(dropout_p),                   # 8 (stateless)
            nn.Linear(hidden_dim // 4, num_classes)  # 9: [9, 128]
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

        # Visual Projection: intermidate_size1 = (1152 + 768) * 2 = 3840
        sz1 = (self.visual_feature_dim + bert_input_dim) * 2
        sz2 = sz1 // 2
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, sz1), nn.LayerNorm(sz1), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz1, sz2), nn.LayerNorm(sz2), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz2, bert_input_dim)
        )
        
        # Textual Projection: intermidate_size1 = (1024 + 768) * 2 = 3584
        sz1_t = (self.textual_feature_dim + bert_input_dim) * 2
        sz2_t = sz1_t // 2
        self.textual_projection = nn.Sequential(
            nn.Linear(textual_feature_dim, sz1_t), nn.LayerNorm(sz1_t), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz1_t, sz2_t), nn.LayerNorm(sz2_t), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(sz2_t, bert_input_dim)
        )
        
        self.norm = nn.LayerNorm(bert_input_dim)
        # Legacy checkpoint support
        self.projection = nn.Linear(visual_feature_dim, bert_input_dim)

    def forward(self, textual_features, visual_features):
        batch_size, seq_len, _ = textual_features.shape
        mask = torch.ones((batch_size, seq_len), device=textual_features.device)
        
        t_proj = self.textual_projection(textual_features)
        v_proj = self.visual_projection(visual_features)
        
        t_norm = self.norm(t_proj)
        v_norm = self.norm(v_proj)
        
        # [Batch, Seq, 2, Dim]
        stacked = torch.stack([t_norm, v_norm], dim=2)
        combined = stacked.view(batch_size, seq_len * 2, -1)
        
        exp_mask = mask.unsqueeze(2).expand(-1, -1, 2).reshape(batch_size, seq_len * 2)
        bert_out = self.bert_encoder(inputs_embeds=combined, attention_mask=exp_mask)
        
        # [Batch, Seq, 2, Dim]
        reshaped = bert_out.last_hidden_state.view(batch_size, seq_len, 2, -1)
        logits = self.classifier(reshaped[:, :, -1, :])
        return logits

def classify_zarr(zarr_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ds = xr.open_zarr(zarr_path)
    
    print("Initializing Model...")
    model = BookBERTMultimodal2(textual_feature_dim=1024, visual_feature_dim=1152, num_classes=9, hidden_dim=512, dropout_p=0.0)
    
    print(f"Loading weights from {checkpoint_path}...")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state: state = state['model_state_dict']
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    print("\n--- Running Inference ---")
    vis_full = torch.from_numpy(ds['visual'].values).to(device)
    txt_full = torch.from_numpy(ds['text'].values).to(device)
    ids_full = ds['ids'].values
    
    # Model max position embeddings is 1024.
    # Each page consumes 2 tokens (1 text + 1 visual).
    # Max pages per pass = 1024 / 2 = 512.
    # We use 500 for safety.
    CHUNK_SIZE = 500
    all_preds = []
    
    with torch.inference_mode():
        total_pages = len(ids_full)
        for start_idx in range(0, total_pages, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_pages)
            
            # Slice batch
            vis_chunk = vis_full[start_idx:end_idx].unsqueeze(0) # [1, N, 1152]
            txt_chunk = txt_full[start_idx:end_idx].unsqueeze(0) # [1, N, 1024]
            
            logits = model(txt_chunk, vis_chunk) # [1, N, 9]
            preds_chunk = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Handle scalar output if chunk size is 1
            if np.ndim(preds_chunk) == 0:
                preds_chunk = [preds_chunk]
                
            all_preds.extend(preds_chunk)
            print(f"Processed {end_idx}/{total_pages} pages...")

    preds = np.array(all_preds)

    # 4. Report
    print("\n--- Predictions (First 20) ---")
    for i in range(min(25, len(preds))):
        label_idx = preds[i]
        label = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Class {label_idx}"
        print(f"{str(ids_full[i])[:60]:<60} | {label:<15}")

    from collections import Counter
    counts = Counter([
        CLASS_NAMES[p] if p < len(CLASS_NAMES) else f"Class {p}" 
        for p in preds
    ])
    print("\n--- Class Distribution ---")
    for label, count in sorted(counts.items()):
        print(f"  {label:<15}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    classify_zarr(args.zarr, args.checkpoint)