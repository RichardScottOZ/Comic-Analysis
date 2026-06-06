"""
Stage 3 Training Script — VLM-Backed

Identical to train_stage3.py but uses Stage3PanelDatasetVLM (VLM JSON output)
instead of Stage3PanelDataset (old OCR panel JSONs).

Key difference: --vlm_cache_dir replaces --s3_manifest + --json_root.
The VLM cache is populated by sync_vlm_cache.py before running this script.

Usage:
    python train_stage3_vlm.py \
        --manifest manifests/master_manifest_20251229.csv \
        --vlm_cache_dir /data/vlm_cache \
        --train_pss_labels train_pss.json \
        --val_pss_labels val_pss.json \
        --checkpoint_dir checkpoints/stage3_vlm \
        --freeze_backbones \
        --epochs 20 \
        --batch_size 8
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
import csv

from stage3_panel_features_framework import PanelFeatureExtractor
from stage3_dataset_vlm import Stage3PanelDatasetVLM, collate_stage3


# ============================================================================
# TRAINING OBJECTIVES  (unchanged from train_stage3.py)
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
        flat_embeddings = panel_embeddings.view(B * N, D)
        flat_mask = panel_mask.view(B * N)

        valid_embeddings = flat_embeddings[flat_mask]
        valid_embeddings = F.normalize(valid_embeddings, dim=-1)

        if valid_embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=panel_embeddings.device)

        page_ids = torch.arange(B, device=panel_embeddings.device).unsqueeze(1).expand(B, N)
        valid_page_ids = page_ids.reshape(-1)[flat_mask]

        sim_matrix = torch.mm(valid_embeddings, valid_embeddings.t()) / self.temperature

        pos_mask = valid_page_ids.unsqueeze(0) == valid_page_ids.unsqueeze(1)
        eye = torch.eye(valid_embeddings.shape[0], device=panel_embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=panel_embeddings.device)

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        exp_sim = torch.exp(sim_matrix) * (~eye).float()
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-6)
        return -mean_log_prob_pos.mean()

    def reconstruction_loss(self, panel_embeddings, panel_mask):
        B, N, D = panel_embeddings.shape
        losses = []
        for b in range(B):
            valid_panels = panel_embeddings[b][panel_mask[b]]
            if valid_panels.shape[0] < 2:
                continue
            mask_idx = torch.randint(0, valid_panels.shape[0], (1,)).item()
            context_mask = torch.ones(valid_panels.shape[0], dtype=torch.bool, device=valid_panels.device)
            context_mask[mask_idx] = False
            predicted = self.reconstruction_head(valid_panels[context_mask].mean(dim=0))
            losses.append(F.mse_loss(predicted, valid_panels[mask_idx]))
        if not losses:
            return torch.tensor(0.0, device=panel_embeddings.device)
        return torch.stack(losses).mean()

    def modality_alignment_loss(self, model, batch):
        B, N = batch['panel_mask'].shape
        device = batch['images'].device

        images = batch['images'].view(B * N, 3, batch['images'].shape[-2], batch['images'].shape[-1])
        input_ids = batch['input_ids'].view(B * N, -1)
        attention_mask = batch['attention_mask'].view(B * N, -1)
        panel_mask_flat = batch['panel_mask'].view(B * N)
        modality_mask_flat = batch['modality_mask'].view(B * N, 3)

        valid_mask = panel_mask_flat & (modality_mask_flat[:, 0] > 0) & (modality_mask_flat[:, 1] > 0)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        vision_emb = F.normalize(model.encode_image_only(images[valid_mask]), dim=-1)
        text_emb = F.normalize(model.encode_text_only(input_ids[valid_mask], attention_mask[valid_mask]), dim=-1)

        logits = torch.mm(vision_emb, text_emb.t()) / self.temperature
        labels = torch.arange(vision_emb.shape[0], device=device)
        return F.cross_entropy(logits, labels)


# ============================================================================
# TRAINING LOOP  (unchanged from train_stage3.py)
# ============================================================================

def train_epoch(model, objectives, dataloader, optimizer, device, epoch):
    model.train()
    objectives.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        B, N = batch['panel_mask'].shape

        model_batch = {
            'images': batch['images'].view(B * N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B * N, -1),
            'attention_mask': batch['attention_mask'].view(B * N, -1),
            'comp_feats': batch['comp_feats'].view(B * N, -1),
            'modality_mask': batch['modality_mask'].view(B * N, 3)
        }

        panel_embeddings = model(model_batch).view(B, N, -1)
        loss = (1.0 * objectives.contrastive_loss(panel_embeddings, batch['panel_mask']) +
                0.5 * objectives.reconstruction_loss(panel_embeddings, batch['panel_mask']) +
                0.3 * objectives.modality_alignment_loss(model, batch))

        # Skip degenerate batches where all panels are masked (loss has no grad_fn)
        if loss.grad_fn is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return {'loss': total_loss / len(dataloader) if dataloader else 0}


@torch.no_grad()
def validate(model, objectives, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Validation"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        B, N = batch['panel_mask'].shape
        model_batch = {
            'images': batch['images'].view(B * N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B * N, -1),
            'attention_mask': batch['attention_mask'].view(B * N, -1),
            'comp_feats': batch['comp_feats'].view(B * N, -1),
            'modality_mask': batch['modality_mask'].view(B * N, 3)
        }
        panel_embeddings = model(model_batch).view(B, N, -1)
        loss = (objectives.contrastive_loss(panel_embeddings, batch['panel_mask']) +
                0.5 * objectives.reconstruction_loss(panel_embeddings, batch['panel_mask']) +
                0.3 * objectives.modality_alignment_loss(model, batch))
        if loss.grad_fn is not None or loss.item() > 0:
            total_loss += loss.item()
    n = len(dataloader)
    return {'loss': total_loss / n if n > 0 else 0}


# ============================================================================
# MAIN
# ============================================================================

def _find_latest_checkpoint(checkpoint_dir: str):
    """Return the checkpoint file with the highest epoch number, or None."""
    import glob
    pattern = str(Path(checkpoint_dir) / 'checkpoint_vlm_epoch_*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract epoch number from filename and return highest
    def epoch_num(p):
        try:
            return int(Path(p).stem.split('_')[-1])
        except ValueError:
            return -1
    return max(files, key=epoch_num)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.use_wandb:
        wandb.init(project="comic-analysis-stage3-vlm", name=args.run_name, config=vars(args))

    # Build image_map from master manifest (canonical_id → absolute_image_path)
    print(f"Loading master manifest: {args.manifest}")
    image_map = {}
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            image_map[row['canonical_id']] = row['absolute_image_path']
    print(f"  {len(image_map):,} image paths loaded")

    train_dataset = Stage3PanelDatasetVLM(
        image_map=image_map,
        vlm_cache_dir=args.vlm_cache_dir,
        pss_labels_path=args.train_pss_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        limit=args.limit,
        shuffle_seed=args.shuffle_seed
    )

    val_labels = args.val_pss_labels if args.val_pss_labels else args.train_pss_labels
    val_limit = args.limit // 5 if args.limit else 500

    val_dataset = Stage3PanelDatasetVLM(
        image_map=image_map,
        vlm_cache_dir=args.vlm_cache_dir,
        pss_labels_path=val_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        limit=val_limit
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_stage3, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_stage3, pin_memory=True
    )

    model = PanelFeatureExtractor(
        visual_backbone=args.visual_backbone,
        visual_fusion=args.visual_fusion,
        feature_dim=args.feature_dim,
        freeze_backbones=args.freeze_backbones
    ).to(device)

    objectives = Stage3TrainingObjectives(feature_dim=args.feature_dim).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(objectives.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume from latest checkpoint if --resume is set
    if args.resume:
        ckpt_path = _find_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            print(f"Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('val_loss', float('inf'))
            print(f"  Resumed at epoch {ckpt['epoch']} | Val Loss was {ckpt['val_loss']:.4f}")
            print(f"  Starting from epoch {start_epoch}/{args.epochs}")
        else:
            print("No checkpoint found in checkpoint_dir — starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, objectives, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, objectives, val_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")

        if args.use_wandb:
            wandb.log({'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss'], 'epoch': epoch})

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), Path(args.checkpoint_dir) / 'best_model_vlm.pt')
            print("  ✅ Saved best model.")

        # Save every epoch so a crash never loses more than one epoch of work
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
        }, Path(args.checkpoint_dir) / f'checkpoint_vlm_epoch_{epoch}.pt')
        # Keep only last 2 epoch checkpoints to save disk space
        prev = Path(args.checkpoint_dir) / f'checkpoint_vlm_epoch_{epoch - 2}.pt'
        if prev.exists():
            prev.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Training — VLM-Backed")

    # Data
    parser.add_argument('--manifest', type=str, required=True,
                        help="Master manifest CSV (has canonical_id + absolute_image_path columns)")
    parser.add_argument('--vlm_cache_dir', type=str, required=True,
                        help="Local root dir of VLM JSONs from sync_vlm_cache.py")
    parser.add_argument('--train_pss_labels', type=str, required=True,
                        help="PSS labels JSON for training pages")
    parser.add_argument('--val_pss_labels', type=str,
                        help="PSS labels JSON for validation pages (optional; uses train if omitted)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit pages loaded — shuffled first so sample is representative")
    parser.add_argument('--shuffle_seed', type=int, default=42,
                        help="Random seed for index shuffle before limit (default: 42)")

    # Model
    parser.add_argument('--visual_backbone', type=str, default='both',
                        choices=['siglip', 'resnet', 'both'])
    parser.add_argument('--visual_fusion', type=str, default='attention')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_panels', type=int, default=16)

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage3_vlm')
    parser.add_argument('--resume', action='store_true',
                        help="Resume from latest checkpoint in --checkpoint_dir")
    parser.add_argument('--run_name', type=str, default='stage3_vlm_run')

    args = parser.parse_args()
    main(args)
