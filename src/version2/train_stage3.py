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
from stage3_dataset import Stage3PanelDataset, collate_stage3

# ============================================================================ 
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
        # Flatten to (TotalPanels, D)
        flat_embeddings = panel_embeddings.view(B*N, D)
        flat_mask = panel_mask.view(B*N)
        
        # Filter invalid
        valid_embeddings = flat_embeddings[flat_mask]
        valid_embeddings = F.normalize(valid_embeddings, dim=-1)
        
        if valid_embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=panel_embeddings.device)
            
        # Create Page IDs for each panel: [0,0,0, 1,1, 2,2,2...]
        page_ids = torch.arange(B, device=panel_embeddings.device).unsqueeze(1).expand(B, N)
        valid_page_ids = page_ids.reshape(-1)[flat_mask]
        
        # Similarity Matrix
        sim_matrix = torch.mm(valid_embeddings, valid_embeddings.t()) / self.temperature
        
        # Positive Mask: Same Page
        pos_mask = valid_page_ids.unsqueeze(0) == valid_page_ids.unsqueeze(1)
        # Remove diagonal (self-contrast)
        eye = torch.eye(valid_embeddings.shape[0], device=panel_embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye
        
        # Check if we have valid positives
        if pos_mask.sum() == 0:
             return torch.tensor(0.0, device=panel_embeddings.device)

        # InfoNCE: SupCon formulation
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        exp_sim = torch.exp(sim_matrix) * (~eye).float()
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        
        # Mean log_prob over positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        return loss
    
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
        
        if not losses: return torch.tensor(0.0, device=panel_embeddings.device)
        return torch.stack(losses).mean()
    
    def modality_alignment_loss(self, model, batch):
        B, N = batch['panel_mask'].shape
        device = batch['images'].device
        
        images = batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1])
        input_ids = batch['input_ids'].view(B*N, -1)
        attention_mask = batch['attention_mask'].view(B*N, -1)
        panel_mask_flat = batch['panel_mask'].view(B*N)
        modality_mask_flat = batch['modality_mask'].view(B*N, 3)
        
        valid_mask = panel_mask_flat & (modality_mask_flat[:, 0] > 0) & (modality_mask_flat[:, 1] > 0)
        if valid_mask.sum() < 2: return torch.tensor(0.0, device=device)
        
        vision_emb = F.normalize(model.encode_image_only(images[valid_mask]), dim=-1)
        text_emb = F.normalize(model.encode_text_only(input_ids[valid_mask], attention_mask[valid_mask]), dim=-1)
        
        logits = torch.mm(vision_emb, text_emb.t()) / self.temperature
        labels = torch.arange(vision_emb.shape[0], device=device)
        return F.cross_entropy(logits, labels)


# ============================================================================ 
# TRAINING LOOP
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
            'images': batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B*N, -1),
            'attention_mask': batch['attention_mask'].view(B*N, -1),
            'comp_feats': batch['comp_feats'].view(B*N, -1),
            'modality_mask': batch['modality_mask'].view(B*N, 3)
        }
        
        panel_embeddings = model(model_batch).view(B, N, -1)
        loss = 1.0 * objectives.contrastive_loss(panel_embeddings, batch['panel_mask']) + \
               0.5 * objectives.reconstruction_loss(panel_embeddings, batch['panel_mask']) + \
               0.3 * objectives.modality_alignment_loss(model, batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return {'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0}

@torch.no_grad()
def validate(model, objectives, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Validation"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        B, N = batch['panel_mask'].shape
        model_batch = {
            'images': batch['images'].view(B*N, 3, batch['images'].shape[-2], batch['images'].shape[-1]),
            'input_ids': batch['input_ids'].view(B*N, -1),
            'attention_mask': batch['attention_mask'].view(B*N, -1),
            'comp_feats': batch['comp_feats'].view(B*N, -1),
            'modality_mask': batch['modality_mask'].view(B*N, 3)
        }
        panel_embeddings = model(model_batch).view(B, N, -1)
        loss = objectives.contrastive_loss(panel_embeddings, batch['panel_mask']) + \
               0.5 * objectives.reconstruction_loss(panel_embeddings, batch['panel_mask']) + \
               0.3 * objectives.modality_alignment_loss(model, batch)
        total_loss += loss.item()
    n = len(dataloader)
    return {'loss': total_loss / n if n > 0 else 0}

# --- Manifest Bridging ---

def normalize_key(cid):
    prefixes = ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/", "CalibreComics_extracted\\", "amazon/"]
    for p in prefixes:
        if cid.startswith(p):
            cid = cid.replace(p, "")
    res = cid.lower()
    for ext in ['.jpg', '.png', '.jpeg']:
        res = res.replace(ext, '')
    return res.replace('/', '_').replace('\\', '_').strip()

def build_json_map(s3_manifest_path):
    print("Building JSON ID Map (Suffix Strategy)...")
    suffix_map = {}
    with open(s3_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            if "__MACOSX" in cid: continue
            
            parts = cid.split('/')
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in suffix_map:
                    suffix_map[suffix] = cid
            
            filename = parts[-1]
            if filename not in suffix_map:
                suffix_map[filename] = cid
    
    print(f"Suffix Map Size: {len(suffix_map)}")
    return suffix_map

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.use_wandb:
        wandb.init(project="comic-analysis-stage3", name=args.run_name, config=vars(args))
    
    # Build Map (Master Key -> Calibre ID)
    json_map = build_json_map(args.s3_manifest)
    
    # Load Master Manifest for Image Paths
    print(f"Loading Master Manifest: {args.manifest}")
    image_map = {}
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            image_map[row['canonical_id']] = row['absolute_image_path']
    
    train_dataset = Stage3PanelDataset(
        image_map=image_map,
        json_map=json_map,
        json_root=args.json_root,
        pss_labels_path=args.train_pss_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        limit=args.limit
    )
    
    val_labels = args.val_pss_labels if args.val_pss_labels else args.train_pss_labels
    val_limit = args.limit // 5 if args.limit else 500
    
    val_dataset = Stage3PanelDataset(
        image_map=image_map,
        json_map=json_map,
        json_root=args.json_root,
        pss_labels_path=val_labels,
        image_size=args.image_size,
        max_panels_per_page=args.max_panels,
        limit=val_limit
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_stage3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_stage3, pin_memory=True)
    
    model = PanelFeatureExtractor(
        visual_backbone=args.visual_backbone,
        visual_fusion=args.visual_fusion,
        feature_dim=args.feature_dim,
        freeze_backbones=args.freeze_backbones
    ).to(device)
    
    objectives = Stage3TrainingObjectives(feature_dim=args.feature_dim).to(device)
    optimizer = AdamW(list(model.parameters()) + list(objectives.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, objectives, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, objectives, val_loader, device)
        scheduler.step()
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        
        if args.use_wandb:
            wandb.log({'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss'], 'epoch': epoch})
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), Path(args.checkpoint_dir) / 'best_model.pt')
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help="Master Manifest (Local Paths)")
    parser.add_argument('--s3_manifest', type=str, required=True, help="Calibre Manifest (IDs)")
    parser.add_argument('--json_root', type=str, required=True)
    parser.add_argument('--train_pss_labels', type=str, required=True)
    parser.add_argument('--val_pss_labels', type=str)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of pages for training')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--visual_backbone', type=str, default='both', choices=['siglip', 'resnet', 'both'])
    parser.add_argument('--visual_fusion', type=str, default='attention')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--freeze_backbones', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/stage3')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='stage3_run')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--max_panels', type=int, default=16)
    
    args = parser.parse_args()
    main(args)