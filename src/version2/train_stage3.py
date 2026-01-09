"Training Script for Stage 3: Domain-Adapted Panel Feature Extraction (Manifest-Driven)\n\nThis script trains the Stage 3 model with contrastive and reconstruction objectives.\nIt uses a manifest CSV to locate images across distributed storage.\n\nTraining objectives:\n1. Contrastive learning: Panels from same page should be similar\n2. Panel reconstruction: Predict masked panel features\n3. Modality alignment: Align visual, text, and compositional representations\n"
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
from pathlib import Path

from stage3_panel_features_framework import PanelFeatureExtractor
from stage3_dataset import Stage3PanelDataset, collate_stage3


# =============================================================================
# TRAINING OBJECTIVES
# ============================================================================

class Stage3TrainingObjectives(nn.Module):
    def __init__(self, feature_dim=512, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def contrastive_loss(self, panel_embeddings, panel_mask):
        B, N, D = panel_embeddings.shape
        embeddings_norm = F.normalize(panel_embeddings, dim=-1)
        losses = []
        for b in range(B):
            valid_mask = panel_mask[b]
            valid_panels = embeddings_norm[b][valid_mask]
            n_valid = valid_panels.shape[0]
            if n_valid < 2: continue
            
            sim_matrix = torch.mm(valid_panels, valid_panels.t()) / self.temperature
            pos_mask = ~torch.eye(n_valid, device=sim_matrix.device, dtype=torch.bool)
            pos_sims = sim_matrix[pos_mask].view(n_valid, n_valid - 1)
            loss = -pos_sims.mean()
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=panel_embeddings.device)
        return torch.stack(losses).mean()
    
    def reconstruction_loss(self, panel_embeddings, panel_mask):
        B, N, D = panel_embeddings.shape
        losses = []
        for b in range(B):
            valid_mask = panel_mask[b]
            valid_panels = panel_embeddings[b][valid_mask]
            n_valid = valid_panels.shape[0]
            if n_valid < 2: continue
            
            mask_idx = torch.randint(0, n_valid, (1,)).item()
            context_mask = torch.ones(n_valid, dtype=torch.bool, device=valid_panels.device)
            context_mask[mask_idx] = False
            context = valid_panels[context_mask].mean(dim=0)
            
            predicted = self.reconstruction_head(context)
            target = valid_panels[mask_idx]
            losses.append(F.mse_loss(predicted, target))
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=panel_embeddings.device)
        return torch.stack(losses).mean()
    
    def modality_alignment_loss(self, model, batch):
        B, N = batch['panel_mask'].shape
        device = batch['images'].device
        
        images = batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1])
        input_ids = batch['input_ids'].view(B*N, -1)
        attention_mask = batch['attention_mask'].view(B*N, -1)
        panel_mask_flat = batch['panel_mask'].view(B*N)
        modality_mask_flat = batch['modality_mask'].view(B*N, 3)
        
        # Only align if both image and text exist
        valid_mask = panel_mask_flat & (modality_mask_flat[:, 0] > 0) & (modality_mask_flat[:, 1] > 0)
        
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)
        
        vision_emb = F.normalize(model.encode_image_only(images[valid_mask]), dim=-1)
        text_emb = F.normalize(model.encode_text_only(input_ids[valid_mask], attention_mask[valid_mask]), dim=-1)
        
        logits = torch.mm(vision_emb, text_emb.t()) / self.temperature
        labels = torch.arange(vision_emb.shape[0], device=device)
        return F.cross_entropy(logits, labels)


# =============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, objectives, dataloader, optimizer, device, epoch):
    model.train()
    objectives.train()
    
    total_loss = 0
    total_contrastive = 0
    total_reconstruction = 0
    total_alignment = 0
    
    pbar = tqdm(dataloader, desc=f