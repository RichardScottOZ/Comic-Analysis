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
    NT-Xent contrastive loss for panel embeddings.

    For each panel, panels from the SAME page are positives and panels
    from ALL OTHER pages in the batch are negatives.  This prevents the
    trivial collapsed solution (all embeddings identical) that the old
    pull-only formulation produced.

    Args:
        panel_embeddings: (B, N, D) contextualized panel embeddings
        panel_mask: (B, N) valid panel indicators (1 = valid)
        temperature: softmax temperature

    Returns:
        Scalar NT-Xent loss
    """
    B, N, D = panel_embeddings.shape

    # --- 1. Build a flat list of valid panel embeddings + their page index ---
    embs, page_ids = [], []
    for b in range(B):
        valid = panel_mask[b].bool()
        panels = F.normalize(panel_embeddings[b][valid], dim=-1)  # (n_valid, D)
        embs.append(panels)
        page_ids.extend([b] * panels.shape[0])

    if len(page_ids) < 2:
        return torch.tensor(0.0, device=panel_embeddings.device)

    all_embs = torch.cat(embs, dim=0)          # (M, D)
    page_ids = torch.tensor(page_ids, device=panel_embeddings.device)  # (M,)

    # --- 2. Full pairwise similarity matrix ---
    sim = torch.mm(all_embs, all_embs.t()) / temperature  # (M, M)

    # --- 3. Masks ---
    M = all_embs.shape[0]
    same_page = page_ids.unsqueeze(1) == page_ids.unsqueeze(0)  # (M, M) bool
    eye = torch.eye(M, device=sim.device, dtype=torch.bool)
    # Positives: same page, not self
    pos_mask = same_page & ~eye
    # Self-similarity is excluded from denominator too
    neg_inf = torch.finfo(sim.dtype).min

    # --- 4. NT-Xent: for each anchor, log-softmax over all non-self, pick positives ---
    sim_no_self = sim.masked_fill(eye, neg_inf)          # (M, M)
    log_probs = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)

    # Average log-prob of all positive pairs per anchor
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
    loss_per_anchor = -(log_probs * pos_mask).sum(dim=1) / n_pos

    # Only compute loss for anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=panel_embeddings.device)

    return loss_per_anchor[has_pos].mean()


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def _find_latest_checkpoint(checkpoint_dir: Path):
    """Return (path, epoch, batch) for the most recent checkpoint, or None."""
    best = None
    for p in checkpoint_dir.glob('checkpoint_stage4_epoch_*.pt'):
        name = p.stem  # e.g. checkpoint_stage4_epoch_3_batch_5000 or checkpoint_stage4_epoch_3
        parts = name.split('_')
        try:
            ep_idx = parts.index('epoch') + 1
            ep = int(parts[ep_idx])
            if 'batch' in parts:
                ba_idx = parts.index('batch') + 1
                ba = int(parts[ba_idx])
            else:
                ba = 0
            if best is None or (ep, ba) > (best[1], best[2]):
                best = (p, ep, ba)
        except (ValueError, IndexError):
            continue
    return best


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, args, start_batch=0, checkpoint_dir=None):
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
        if batch_idx < start_batch:
            continue
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
                    scores = model.visual_closure_head(preceding.unsqueeze(0), candidates.unsqueeze(0))  # (1, K)
                    loss = closure_loss(scores, correct_idx)
                    task_losses['visual_closure'] += loss.item()
                    task_counts['visual_closure'] += 1
                else:
                    scores = model.text_closure_head(preceding.unsqueeze(0), candidates.unsqueeze(0))  # (1, K)
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
        
        if checkpoint_dir is not None and args.save_every_n_batches > 0 and (batch_idx + 1) % args.save_every_n_batches == 0:
            mid_path = checkpoint_dir / f'checkpoint_stage4_epoch_{epoch}_batch_{batch_idx + 1}.pt'
            torch.save({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': None,
            }, mid_path)
        
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
                    scores = model.visual_closure_head(preceding.unsqueeze(0), candidates.unsqueeze(0))  # (1, K)
                    loss = closure_loss(scores, correct_idx)
                    task_losses['visual_closure'] += loss.item()
                    task_counts['visual_closure'] += 1
                else:
                    scores = model.text_closure_head(preceding.unsqueeze(0), candidates.unsqueeze(0))  # (1, K)
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
    
    # Resume logic
    start_epoch = 1
    start_batch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        found = _find_latest_checkpoint(checkpoint_dir)
        if found:
            ckpt_path, start_epoch, start_batch = found
            print(f"▶️  Resuming from {ckpt_path} (epoch {start_epoch}, batch {start_batch})")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if ckpt.get('val_loss'):
                best_val_loss = ckpt['val_loss']
            # If batch > 0, we're mid-epoch; if batch == 0, start_epoch is the next epoch
            if start_batch == 0:
                start_epoch += 1  # checkpoint was end-of-epoch, start next
        else:
            print("No checkpoint found, starting from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_task_losses, train_counts = train_epoch(
            model, train_loader, optimizer, device, epoch, args,
            start_batch=start_batch if epoch == start_epoch else 0,
            checkpoint_dir=checkpoint_dir
        )
        start_batch = 0  # only skip batches for the resumed epoch
        
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
        
        # Always save end-of-epoch checkpoint for resume support
        epoch_ckpt_path = checkpoint_dir / f'checkpoint_stage4_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'batch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(args)
        }, epoch_ckpt_path)
    
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
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--save_every_n_batches', type=int, default=2000,
                       help='Save mid-epoch checkpoint every N batches')
    
    args = parser.parse_args()
    main(args)
