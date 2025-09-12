"""
Training script for CLOSURE-Lite framework
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb

from closure_lite_framework import ClosureLite
from closure_lite_dataset import create_dataloader

def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {'L_mpm': 0, 'L_pop': 0, 'L_rpp': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            loss, components = model(batch)
        
        # Check for NaN loss with detailed debugging
        if torch.isnan(loss):
            print(f"\nNaN loss detected at batch {batch_idx}!")
            print(f"  Loss components: {components}")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for NaN in model parameters
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
            
            if nan_params:
                print(f"  NaN parameters found: {nan_params}")
                print("Resetting model parameters...")
                # Reset the model to a previous checkpoint if available
                if hasattr(model, '_last_checkpoint'):
                    model.load_state_dict(model._last_checkpoint)
                else:
                    print("No checkpoint available, reinitializing...")
                    # Reinitialize the problematic layers
                    for name, param in model.named_parameters():
                        if name in nan_params:
                            torch.nn.init.xavier_uniform_(param)
            
            print("Skipping this batch and reducing learning rate...")
            
            # Reduce learning rate when NaN occurs
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            continue
            
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent NaN - more aggressive
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # If gradients are still too large, reduce learning rate
        if grad_norm > 0.3:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'MPM': f"{components['L_mpm']:.4f}",
            'POP': f"{components['L_pop']:.4f}",
            'RPP': f"{components['L_rpp']:.4f}"
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'train/loss': loss.item(),
                'train/mpm_loss': components['L_mpm'],
                'train/pop_loss': components['L_pop'],
                'train/rpp_loss': components['L_rpp'],
                'epoch': epoch,
                'batch': batch_idx
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def main():
    parser = argparse.ArgumentParser(description='Train CLOSURE-Lite model')
    parser.add_argument('--json_dir', type=str, required=True, 
                       help='Directory containing DataSpec JSON files')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Root directory for comic images')
    parser.add_argument('--output_dir', type=str, default='./closure_lite_output',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (pages)')
    parser.add_argument('--grad_accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--max_panels', type=int, default=12,
                       help='Maximum panels per page')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of pages to sample (None = use all 800K pages)')
    parser.add_argument('--rtl', action='store_true',
                       help='Right-to-left reading order (for manga)')
    parser.add_argument('--wandb_project', type=str, default='closure-lite',
                       help='Wandb project name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"closure_lite_{Path(args.json_dir).name}"
    )
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(
        args.json_dir, 
        args.image_root, 
        batch_size=args.batch_size,
        max_panels=args.max_panels,
        rtl=args.rtl,
        num_workers=2,
        max_samples=args.max_samples
    )
    print(f"Dataset size: {len(dataloader.dataset)} pages")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    print("Creating model...")
    model = ClosureLite(d=384).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.05
    )
    
    # Setup scheduler - more conservative
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader),
        eta_min=args.lr * 0.1  # Less aggressive decay
    )
    
    # Setup mixed precision
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Save model state before epoch for recovery
        model._last_checkpoint = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Train
        avg_loss, avg_components = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch+1
        )
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        print(f"  MPM: {avg_components['L_mpm']:.4f}")
        print(f"  POP: {avg_components['L_pop']:.4f}")
        print(f"  RPP: {avg_components['L_rpp']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'config': vars(args)
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
            print(f"New best model saved! Loss: {best_loss:.4f}")
        
        # Log epoch metrics
        if wandb.run is not None:
            wandb.log({
                'epoch/avg_loss': avg_loss,
                'epoch/avg_mpm': avg_components['L_mpm'],
                'epoch/avg_pop': avg_components['L_pop'],
                'epoch/avg_rpp': avg_components['L_rpp'],
                'epoch/lr': scheduler.get_last_lr()[0]
            })
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    wandb.finish()

if __name__ == "__main__":
    main()
