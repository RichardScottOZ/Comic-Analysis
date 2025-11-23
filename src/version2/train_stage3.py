"""
Training Script for Stage 3: Domain-Adapted Panel Feature Extraction

This script trains the Stage 3 model with contrastive and reconstruction objectives
to learn rich panel representations suitable for Stage 4 sequence modeling.

Training objectives:
1. Contrastive learning: Panels from same page should be similar
2. Panel reconstruction: Predict masked panel features
3. Modality alignment: Align visual, text, and compositional representations
"""

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


# ============================================================================
# TRAINING OBJECTIVES
# ============================================================================

class Stage3TrainingObjectives(nn.Module):
    """
    Training objectives for Stage 3 panel feature learning.
    """
    
    def __init__(self, feature_dim=512, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Panel reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def contrastive_loss(self, panel_embeddings, panel_mask):
        """
        Contrastive loss: Panels from same page should be similar.
        
        Uses a proper contrastive formulation where all other panels on the same
        page are positive examples, and we maximize agreement among them.
        
        Args:
            panel_embeddings: (B, N, D) panel features
            panel_mask: (B, N) binary mask for valid panels
            
        Returns:
            Scalar contrastive loss
        """
        B, N, D = panel_embeddings.shape
        
        # Normalize embeddings
        embeddings_norm = F.normalize(panel_embeddings, dim=-1)
        
        losses = []
        for b in range(B):
            # Get valid panels for this page
            valid_mask = panel_mask[b]
            valid_panels = embeddings_norm[b][valid_mask]  # (n_valid, D)
            n_valid = valid_panels.shape[0]
            
            if n_valid < 2:
                continue  # Need at least 2 panels for contrastive loss
            
            # Compute similarity matrix
            sim_matrix = torch.mm(valid_panels, valid_panels.t())  # (n_valid, n_valid)
            sim_matrix = sim_matrix / self.temperature
            
            # For proper contrastive loss: maximize similarity to all other panels
            # on same page (all are positive examples)
            # Create mask for positive pairs (all pairs except self)
            pos_mask = ~torch.eye(n_valid, device=sim_matrix.device, dtype=torch.bool)
            
            # Extract positive similarities (all other panels on same page)
            pos_sims = sim_matrix[pos_mask].view(n_valid, n_valid - 1)
            
            # Contrastive loss: negative log of mean similarity to positives
            # We want to maximize similarity, so minimize negative similarity
            loss = -pos_sims.mean()
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=panel_embeddings.device)
        
        return torch.stack(losses).mean()
    
    def reconstruction_loss(self, panel_embeddings, panel_mask):
        """
        Panel reconstruction loss: Predict masked panel from context.
        
        Args:
            panel_embeddings: (B, N, D) panel features
            panel_mask: (B, N) binary mask for valid panels
            
        Returns:
            Scalar reconstruction loss
        """
        B, N, D = panel_embeddings.shape
        
        losses = []
        for b in range(B):
            # Get valid panels
            valid_mask = panel_mask[b]
            valid_panels = panel_embeddings[b][valid_mask]  # (n_valid, D)
            n_valid = valid_panels.shape[0]
            
            if n_valid < 2:
                continue  # Need at least 2 panels
            
            # Randomly mask one panel
            mask_idx = torch.randint(0, n_valid, (1,)).item()
            
            # Context: average of other panels
            context_mask = torch.ones(n_valid, dtype=torch.bool, device=valid_panels.device)
            context_mask[mask_idx] = False
            context = valid_panels[context_mask].mean(dim=0)  # (D,)
            
            # Predict masked panel from context
            predicted = self.reconstruction_head(context)
            target = valid_panels[mask_idx]
            
            # MSE loss
            loss = F.mse_loss(predicted, target)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=panel_embeddings.device)
        
        return torch.stack(losses).mean()
    
    def modality_alignment_loss(self, model, batch):
        """
        Modality alignment: Vision and text for same panel should be similar.
        
        Args:
            model: PanelFeatureExtractor model
            batch: Batch dictionary
            
        Returns:
            Scalar alignment loss
        """
        B, N = batch['panel_mask'].shape
        device = batch['images'].device
        
        # Flatten batch for processing
        images = batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1])
        input_ids = batch['input_ids'].view(B*N, -1)
        attention_mask = batch['attention_mask'].view(B*N, -1)
        panel_mask_flat = batch['panel_mask'].view(B*N)
        modality_mask_flat = batch['modality_mask'].view(B*N, 3)
        
        # Filter to valid panels with both image and text
        valid_mask = panel_mask_flat & (modality_mask_flat[:, 0] > 0) & (modality_mask_flat[:, 1] > 0)
        
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)
        
        # Get modality-specific embeddings
        vision_emb = model.encode_image_only(images[valid_mask])
        text_emb = model.encode_text_only(input_ids[valid_mask], attention_mask[valid_mask])
        
        # Normalize
        vision_emb = F.normalize(vision_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Contrastive loss between vision and text
        # Positive pairs: same panel
        logits = torch.mm(vision_emb, text_emb.t()) / self.temperature
        labels = torch.arange(vision_emb.shape[0], device=device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, objectives, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    objectives.train()
    
    total_loss = 0
    total_contrastive = 0
    total_reconstruction = 0
    total_alignment = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Flatten batch for model forward
        B, N = batch['panel_mask'].shape
        
        model_batch = {
            'images': batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B*N, -1),
            'attention_mask': batch['attention_mask'].view(B*N, -1),
            'comp_feats': batch['comp_feats'].view(B*N, -1),
            'modality_mask': batch['modality_mask'].view(B*N, 3)
        }
        
        # Forward pass
        panel_embeddings = model(model_batch)  # (B*N, D)
        panel_embeddings = panel_embeddings.view(B, N, -1)  # (B, N, D)
        
        # Compute losses
        loss_contrastive = objectives.contrastive_loss(panel_embeddings, batch['panel_mask'])
        loss_reconstruction = objectives.reconstruction_loss(panel_embeddings, batch['panel_mask'])
        loss_alignment = objectives.modality_alignment_loss(model, batch)
        
        # Combined loss (weights could be made configurable via args)
        # TODO: Add loss weight arguments for better experimentation
        loss = (1.0 * loss_contrastive + 
                0.5 * loss_reconstruction + 
                0.3 * loss_alignment)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_reconstruction += loss_reconstruction.item()
        total_alignment += loss_alignment.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'contr': f"{loss_contrastive.item():.4f}",
            'recon': f"{loss_reconstruction.item():.4f}",
            'align': f"{loss_alignment.item():.4f}"
        })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'contrastive': total_contrastive / n_batches,
        'reconstruction': total_reconstruction / n_batches,
        'alignment': total_alignment / n_batches
    }


@torch.no_grad()
def validate(model, objectives, dataloader, device):
    """
    Validate model.
    """
    model.eval()
    objectives.eval()
    
    total_loss = 0
    total_contrastive = 0
    total_reconstruction = 0
    total_alignment = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Flatten batch for model forward
        B, N = batch['panel_mask'].shape
        
        model_batch = {
            'images': batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B*N, -1),
            'attention_mask': batch['attention_mask'].view(B*N, -1),
            'comp_feats': batch['comp_feats'].view(B*N, -1),
            'modality_mask': batch['modality_mask'].view(B*N, 3)
        }
        
        # Forward pass
        panel_embeddings = model(model_batch)
        panel_embeddings = panel_embeddings.view(B, N, -1)
        
        # Compute losses
        loss_contrastive = objectives.contrastive_loss(panel_embeddings, batch['panel_mask'])
        loss_reconstruction = objectives.reconstruction_loss(panel_embeddings, batch['panel_mask'])
        loss_alignment = objectives.modality_alignment_loss(model, batch)
        
        loss = loss_contrastive + 0.5 * loss_reconstruction + 0.3 * loss_alignment
        
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_reconstruction += loss_reconstruction.item()
        total_alignment += loss_alignment.item()
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'contrastive': total_contrastive / n_batches,
        'reconstruction': total_reconstruction / n_batches,
        'alignment': total_alignment / n_batches
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="comic-analysis-stage3",
            name=args.run_name,
            config=vars(args)
        )
    
    # Create datasets
    train_dataset = Stage3PanelDataset(
        root_dir=args.data_root,
        pss_labels_path=args.train_pss_labels,
        max_text_length=args.max_text_length,
        image_size=args.image_size,
        only_narrative=True,
        max_panels_per_page=args.max_panels
    )
    
    val_dataset = Stage3PanelDataset(
        root_dir=args.data_root,
        pss_labels_path=args.val_pss_labels,
        max_text_length=args.max_text_length,
        image_size=args.image_size,
        only_narrative=True,
        max_panels_per_page=args.max_panels
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_stage3,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_stage3,
        pin_memory=True
    )
    
    # Create model
    model = PanelFeatureExtractor(
        visual_backbone=args.visual_backbone,
        visual_fusion=args.visual_fusion,
        feature_dim=args.feature_dim,
        freeze_backbones=args.freeze_backbones,
        use_modality_indicators=True
    ).to(device)
    
    # Create objectives
    objectives = Stage3TrainingObjectives(
        feature_dim=args.feature_dim,
        temperature=args.temperature
    ).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(objectives.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler (step per epoch)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, objectives, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, objectives, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Contrastive: {train_metrics['contrastive']:.4f}, "
              f"Reconstruction: {train_metrics['reconstruction']:.4f}, "
              f"Alignment: {train_metrics['alignment']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Contrastive: {val_metrics['contrastive']:.4f}, "
              f"Reconstruction: {val_metrics['reconstruction']:.4f}, "
              f"Alignment: {val_metrics['alignment']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/contrastive': train_metrics['contrastive'],
                'train/reconstruction': train_metrics['reconstruction'],
                'train/alignment': train_metrics['alignment'],
                'val/loss': val_metrics['loss'],
                'val/contrastive': val_metrics['contrastive'],
                'val/reconstruction': val_metrics['reconstruction'],
                'val/alignment': val_metrics['alignment'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save latest checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 3 Panel Feature Extractor")
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing book subdirectories')
    parser.add_argument('--train_pss_labels', type=str, required=True,
                       help='Path to training PSS labels JSON')
    parser.add_argument('--val_pss_labels', type=str, required=True,
                       help='Path to validation PSS labels JSON')
    
    # Model arguments
    parser.add_argument('--visual_backbone', type=str, default='both',
                       choices=['siglip', 'resnet', 'both'],
                       help='Visual backbone architecture')
    parser.add_argument('--visual_fusion', type=str, default='attention',
                       choices=['concat', 'attention', 'gate'],
                       help='Visual fusion strategy for multi-backbone')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension')
    parser.add_argument('--freeze_backbones', action='store_true',
                       help='Freeze pretrained backbones')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    
    # Data processing arguments
    parser.add_argument('--max_text_length', type=int, default=128,
                       help='Maximum text sequence length')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for panel crops')
    parser.add_argument('--max_panels', type=int, default=16,
                       help='Maximum panels per page')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage3',
                       help='Directory for saving checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--run_name', type=str, default='stage3_training',
                       help='Run name for wandb')
    
    args = parser.parse_args()
    main(args)
