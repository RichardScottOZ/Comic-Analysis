"""
Training Script for Stage 4: Semantic Sequence Modeling

Trains the Stage 4 model with multiple tasks:
- Panel picking (ComicsPAP)
- Character coherence
- Visual/Text closure
- Caption relevance
- Text-cloze
- Reading order

Uses multi-task learning with task-specific losses.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
from pathlib import Path
import numpy as np

from stage4_sequence_modeling_framework import Stage4SequenceModel
from stage4_dataset import Stage4SequenceDataset, collate_stage4


# ============================================================================
# TRAINING OBJECTIVES
# ============================================================================

def panel_picking_loss(predictions: torch.Tensor, 
                       labels: torch.Tensor) -> torch.Tensor:
    """
    Loss for panel picking task.
    
    Args:
        predictions: (B, K) logits for candidates
        labels: (B,) indices of correct panels
        
    Returns:
        Scalar cross-entropy loss
    """
    return F.cross_entropy(predictions, labels)


def closure_loss(predictions: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    """
    Loss for visual/text closure tasks.
    
    Args:
        predictions: (B, K) logits for candidates
        labels: (B,) indices of correct continuations
        
    Returns:
        Scalar cross-entropy loss
    """
    return F.cross_entropy(predictions, labels)


def reading_order_loss(order_matrix: torch.Tensor,
                      adjacency_matrix: torch.Tensor,
                      panel_mask: torch.Tensor) -> torch.Tensor:
    """
    Loss for reading order task.
    
    Args:
        order_matrix: (B, N, N) predicted pairwise orderings
        adjacency_matrix: (B, N, N) ground truth adjacencies
        panel_mask: (B, N) valid panel indicators
        
    Returns:
        Scalar binary cross-entropy loss
    """
    # Mask invalid positions
    B, N, _ = order_matrix.shape
    
    # Create mask for valid pairs
    mask_i = panel_mask.unsqueeze(2).expand(B, N, N)
    mask_j = panel_mask.unsqueeze(1).expand(B, N, N)
    valid_mask = (mask_i * mask_j).bool()
    
    # Compute loss only on valid pairs
    loss = F.binary_cross_entropy_with_logits(
        order_matrix[valid_mask],
        adjacency_matrix[valid_mask].float()
    )
    
    return loss


def contrastive_panel_loss(panel_embeddings: torch.Tensor,
                           panel_mask: torch.Tensor,
                           temperature: float = 0.07) -> torch.Tensor:
    """
    Contrastive loss for panel embeddings within a sequence.
    Encourages panels in same sequence to be similar.
    
    Args:
        panel_embeddings: (B, N, D) contextualized panel embeddings
        panel_mask: (B, N) valid panel indicators
        temperature: Temperature for contrastive loss
        
    Returns:
        Scalar contrastive loss
    """
    B, N, D = panel_embeddings.shape
    
    # Normalize embeddings
    embeddings_norm = F.normalize(panel_embeddings, dim=-1)
    
    losses = []
    for b in range(B):
        # Get valid panels
        valid_mask = panel_mask[b].bool()
        valid_panels = embeddings_norm[b][valid_mask]
        n_valid = valid_panels.shape[0]
        
        if n_valid < 2:
            continue
        
        # Compute similarity matrix
        sim_matrix = torch.mm(valid_panels, valid_panels.t()) / temperature
        
        # Mask out self-similarity
        mask = ~torch.eye(n_valid, device=sim_matrix.device, dtype=torch.bool)
        
        # Extract positive similarities
        pos_sims = sim_matrix[mask].view(n_valid, n_valid - 1)
        
        # Loss: negative mean similarity
        loss = -pos_sims.mean()
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=panel_embeddings.device)
    
    return torch.stack(losses).mean()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """
    Train for one epoch with multi-task learning.
    """
    model.train()
    
    total_loss = 0
    task_losses = {
        'panel_picking': 0,
        'visual_closure': 0,
        'text_closure': 0,
        'reading_order': 0,
        'contrastive': 0
    }
    task_counts = {k: 0 for k in task_losses.keys()}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        panel_embeddings = batch['panel_embeddings'].to(device)
        panel_mask = batch['panel_mask'].to(device)
        
        # Forward pass
        outputs = model(panel_embeddings, panel_mask)
        contextualized_panels = outputs['contextualized_panels']
        
        # Compute task-specific losses
        batch_loss = torch.tensor(0.0, device=device)
        
        # Process each sample's task
        for i, (task_type, task_data) in enumerate(zip(batch['task_types'], batch['task_data'])):
            
            if task_type == 'panel_picking':
                # Extract task data
                context_emb = torch.from_numpy(task_data['context_embedding']).float().to(device)
                candidates = torch.from_numpy(task_data['candidates']).float().to(device)
                correct_idx = torch.tensor([task_data['correct_idx']], device=device)
                
                # Predict
                scores = model.panel_picking_head(context_emb.unsqueeze(0), candidates.unsqueeze(0))
                loss = panel_picking_loss(scores, correct_idx)
                
                batch_loss = batch_loss + loss
                task_losses['panel_picking'] += loss.item()
                task_counts['panel_picking'] += 1
            
            elif task_type in ['visual_closure', 'text_closure']:
                # Extract task data
                preceding = torch.from_numpy(task_data['preceding_panels']).float().to(device)
                candidates = torch.from_numpy(task_data['candidates']).float().to(device)
                correct_idx = torch.tensor([task_data['correct_idx']], device=device)
                
                # Predict
                if task_type == 'visual_closure':
                    scores = []
                    for k in range(candidates.shape[0]):
                        score = model.visual_closure_head(
                            preceding.unsqueeze(0),
                            candidates[k].unsqueeze(0)
                        )
                        scores.append(score)
                    scores = torch.stack(scores).t()  # (1, K)
                    loss = closure_loss(scores, correct_idx)
                    task_losses['visual_closure'] += loss.item()
                    task_counts['visual_closure'] += 1
                else:
                    scores = []
                    for k in range(candidates.shape[0]):
                        score = model.text_closure_head(
                            preceding.unsqueeze(0),
                            candidates[k].unsqueeze(0)
                        )
                        scores.append(score)
                    scores = torch.stack(scores).t()  # (1, K)
                    loss = closure_loss(scores, correct_idx)
                    task_losses['text_closure'] += loss.item()
                    task_counts['text_closure'] += 1
                
                batch_loss = batch_loss + loss
            
            elif task_type == 'reading_order':
                # Extract task data
                shuffled_panels = torch.from_numpy(task_data['shuffled_panels']).float().to(device)
                adj_matrix = torch.from_numpy(task_data['adjacency_matrix']).long().to(device)
                
                # Predict
                order_matrix = model.reading_order_head(shuffled_panels.unsqueeze(0))
                
                # Create mask for this sequence
                seq_mask = torch.ones(shuffled_panels.shape[0], device=device)
                
                loss = reading_order_loss(
                    order_matrix,
                    adj_matrix.unsqueeze(0),
                    seq_mask.unsqueeze(0)
                )
                
                batch_loss = batch_loss + args.reading_order_weight * loss
                task_losses['reading_order'] += loss.item()
                task_counts['reading_order'] += 1
        
        # Add contrastive loss for all samples
        contrastive_loss_val = contrastive_panel_loss(
            contextualized_panels,
            panel_mask,
            temperature=args.temperature
        )
        batch_loss = batch_loss + args.contrastive_weight * contrastive_loss_val
        task_losses['contrastive'] += contrastive_loss_val.item()
        task_counts['contrastive'] += 1
        
        # Normalize by batch size
        batch_loss = batch_loss / len(batch['task_types'])
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # Update metrics
        total_loss += batch_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{batch_loss.item():.4f}",
            'pp': f"{task_losses['panel_picking'] / max(task_counts['panel_picking'], 1):.4f}",
            'ro': f"{task_losses['reading_order'] / max(task_counts['reading_order'], 1):.4f}"
        })
    
    # Compute average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    
    avg_task_losses = {
        k: v / max(task_counts[k], 1) 
        for k, v in task_losses.items()
    }
    
    return avg_loss, avg_task_losses, task_counts


@torch.no_grad()
def validate(model, dataloader, device, args):
    """
    Validate model.
    """
    model.eval()
    
    total_loss = 0
    task_losses = {
        'panel_picking': 0,
        'visual_closure': 0,
        'text_closure': 0,
        'reading_order': 0,
        'contrastive': 0
    }
    task_counts = {k: 0 for k in task_losses.keys()}
    
    for batch in tqdm(dataloader, desc="Validation"):
        panel_embeddings = batch['panel_embeddings'].to(device)
        panel_mask = batch['panel_mask'].to(device)
        
        outputs = model(panel_embeddings, panel_mask)
        contextualized_panels = outputs['contextualized_panels']
        
        batch_loss = torch.tensor(0.0, device=device)
        
        # Similar to training, compute losses for each task
        for i, (task_type, task_data) in enumerate(zip(batch['task_types'], batch['task_data'])):
            
            if task_type == 'panel_picking':
                context_emb = torch.from_numpy(task_data['context_embedding']).float().to(device)
                candidates = torch.from_numpy(task_data['candidates']).float().to(device)
                correct_idx = torch.tensor([task_data['correct_idx']], device=device)
                
                scores = model.panel_picking_head(context_emb.unsqueeze(0), candidates.unsqueeze(0))
                loss = panel_picking_loss(scores, correct_idx)
                
                batch_loss = batch_loss + loss
                task_losses['panel_picking'] += loss.item()
                task_counts['panel_picking'] += 1
            
            elif task_type in ['visual_closure', 'text_closure']:
                preceding = torch.from_numpy(task_data['preceding_panels']).float().to(device)
                candidates = torch.from_numpy(task_data['candidates']).float().to(device)
                correct_idx = torch.tensor([task_data['correct_idx']], device=device)
                
                if task_type == 'visual_closure':
                    scores = []
                    for k in range(candidates.shape[0]):
                        score = model.visual_closure_head(
                            preceding.unsqueeze(0),
                            candidates[k].unsqueeze(0)
                        )
                        scores.append(score)
                    scores = torch.stack(scores).t()
                    loss = closure_loss(scores, correct_idx)
                    task_losses['visual_closure'] += loss.item()
                    task_counts['visual_closure'] += 1
                else:
                    scores = []
                    for k in range(candidates.shape[0]):
                        score = model.text_closure_head(
                            preceding.unsqueeze(0),
                            candidates[k].unsqueeze(0)
                        )
                        scores.append(score)
                    scores = torch.stack(scores).t()
                    loss = closure_loss(scores, correct_idx)
                    task_losses['text_closure'] += loss.item()
                    task_counts['text_closure'] += 1
                
                batch_loss = batch_loss + loss
            
            elif task_type == 'reading_order':
                shuffled_panels = torch.from_numpy(task_data['shuffled_panels']).float().to(device)
                adj_matrix = torch.from_numpy(task_data['adjacency_matrix']).long().to(device)
                
                order_matrix = model.reading_order_head(shuffled_panels.unsqueeze(0))
                seq_mask = torch.ones(shuffled_panels.shape[0], device=device)
                
                loss = reading_order_loss(
                    order_matrix,
                    adj_matrix.unsqueeze(0),
                    seq_mask.unsqueeze(0)
                )
                
                batch_loss = batch_loss + args.reading_order_weight * loss
                task_losses['reading_order'] += loss.item()
                task_counts['reading_order'] += 1
        
        contrastive_loss_val = contrastive_panel_loss(
            contextualized_panels,
            panel_mask,
            temperature=args.temperature
        )
        batch_loss = batch_loss + args.contrastive_weight * contrastive_loss_val
        task_losses['contrastive'] += contrastive_loss_val.item()
        task_counts['contrastive'] += 1
        
        batch_loss = batch_loss / len(batch['task_types'])
        total_loss += batch_loss.item()
    
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    
    avg_task_losses = {
        k: v / max(task_counts[k], 1)
        for k, v in task_losses.items()
    }
    
    return avg_loss, avg_task_losses, task_counts


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
            project="comic-analysis-stage4",
            name=args.run_name,
            config=vars(args)
        )
    
    # Create datasets
    train_dataset = Stage4SequenceDataset(
        embeddings_path=args.train_embeddings,
        metadata_path=args.train_metadata,
        min_panels=args.min_panels,
        max_panels=args.max_panels,
        num_candidates=args.num_candidates
    )
    
    val_dataset = Stage4SequenceDataset(
        embeddings_path=args.val_embeddings,
        metadata_path=args.val_metadata,
        min_panels=args.min_panels,
        max_panels=args.max_panels,
        num_candidates=args.num_candidates
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_stage4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_stage4,
        pin_memory=True
    )
    
    # Create model
    model = Stage4SequenceModel(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_panels=args.max_panels
    ).to(device)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs,
        T_mult=1,
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
        train_loss, train_task_losses, train_counts = train_epoch(
            model, train_loader, optimizer, device, epoch, args
        )
        
        # Validate
        val_loss, val_task_losses, val_counts = validate(
            model, val_loader, device, args
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Task losses: {train_task_losses}")
        print(f"  Task counts: {train_counts}")
        
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"  Task losses: {val_task_losses}")
        print(f"  Task counts: {val_counts}")
        
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            # Add task losses
            for task, loss in train_task_losses.items():
                log_dict[f'train/{task}'] = loss
            for task, loss in val_task_losses.items():
                log_dict[f'val/{task}'] = loss
            
            wandb.log(log_dict)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
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
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 4 Sequence Model")
    
    # Data arguments
    parser.add_argument('--train_embeddings', type=str, required=True,
                       help='Path to training embeddings.zarr')
    parser.add_argument('--train_metadata', type=str, required=True,
                       help='Path to training metadata.json')
    parser.add_argument('--val_embeddings', type=str, required=True,
                       help='Path to validation embeddings.zarr')
    parser.add_argument('--val_metadata', type=str, required=True,
                       help='Path to validation metadata.json')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    
    # Task weights
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                       help='Weight for contrastive loss')
    parser.add_argument('--reading_order_weight', type=float, default=0.3,
                       help='Weight for reading order loss')
    
    # Data processing arguments
    parser.add_argument('--min_panels', type=int, default=3,
                       help='Minimum panels per sequence')
    parser.add_argument('--max_panels', type=int, default=16,
                       help='Maximum panels per sequence')
    parser.add_argument('--num_candidates', type=int, default=5,
                       help='Number of candidates for discriminative tasks')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage4',
                       help='Directory for saving checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--run_name', type=str, default='stage4_training',
                       help='Run name for wandb')
    
    args = parser.parse_args()
    main(args)
