"""
CLOSURE-Lite Framework for Comic Understanding
Extracted from second-notes.py

A 16GB-ready model for comic understanding using:
- ViT-S + RoBERTa-base fusion
- Reading order + adjacency inference
- Self-supervised learning (MPM, POP-lite, RPP-lite)
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
import timm

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def norm_boxes(panels, page_w, page_h):
    boxes = []
    for p in panels:
        x,y,w,h = p['panel_coords']
        boxes.append([x/page_w, y/page_h, w/page_w, h/page_h])
    return boxes  # list of [x,y,w,h] in [0,1]

def _vertical_overlap(a, b):
    ay1, ay2 = a[1], a[1] + a[3]
    by1, by2 = b[1], b[1] + b[3]
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / max(1e-6, min(a[3], b[3]))  # overlap relative to smaller height

def _horizontal_overlap(a, b):
    ax1, ax2 = a[0], a[0] + a[2]
    bx1, bx2 = b[0], b[0] + b[2]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return inter / max(1e-6, min(a[2], b[2]))

def _centers(b):
    return (b[0] + b[2]/2, b[1] + b[3]/2)

def compute_reading_order(panels: List[Dict], page_w: int, page_h: int, rtl: bool=False) -> List[int]:
    """
    Returns an ordered list of panel indices following a Z-path reading order
    with row grouping by vertical overlap.
    """
    boxes = norm_boxes(panels, page_w, page_h)
    idxs = list(range(len(panels)))
    # Sort by top y
    idxs.sort(key=lambda i: boxes[i][1])

    rows = []
    current = [idxs[0]] if idxs else []
    for i in idxs[1:]:
        prev_box = boxes[current[-1]]
        box = boxes[i]
        if _vertical_overlap(prev_box, box) > 0.25:
            current.append(i)
        else:
            rows.append(current)
            current = [i]
    if current: rows.append(current)

    # Within each row, sort left->right (or right->left for RTL)
    order = []
    for row in rows:
        row.sort(key=lambda i: boxes[i][0], reverse=rtl)
        order.extend(row)
    return order

def build_adjacency_and_next(panels: List[Dict], page_w: int, page_h: int, rtl: bool=False, k_fallback:int=2):
    """
    Builds:
      - adj_mask: NxN binary adjacency (left/right/above/below + next-in-order + kNN fallback)
      - next_idx: length-N array: next panel index in reading order (or -100 for last)
    """
    N = len(panels)
    boxes = norm_boxes(panels, page_w, page_h)
    centers = [_centers(b) for b in boxes]
    order = compute_reading_order(panels, page_w, page_h, rtl=rtl)

    adj = [[0]*N for _ in range(N)]
    next_idx = [-100]*N
    # next in order
    for pos, i in enumerate(order):
        if pos < len(order) - 1:
            j = order[pos+1]
            adj[i][j] = 1
            next_idx[i] = j

    # directional neighbors with overlap constraints
    for i in range(N):
        bi = boxes[i]; ci = centers[i]
        left, right, above, below = None, None, None, None
        left_dx, right_dx, up_dy, down_dy = 1e9, 1e9, 1e9, 1e9
        for j in range(N):
            if i == j: continue
            bj = boxes[j]; cj = centers[j]
            dx, dy = cj[0] - ci[0], cj[1] - ci[1]
            # left neighbor (j left of i) with vertical overlap
            if dx < 0 and _vertical_overlap(bi, bj) > 0.2 and abs(dx) < left_dx:
                left, left_dx = j, abs(dx)
            # right neighbor
            if dx > 0 and _vertical_overlap(bi, bj) > 0.2 and dx < right_dx:
                right, right_dx = j, dx
            # above neighbor
            if dy < 0 and _horizontal_overlap(bi, bj) > 0.2 and abs(dy) < up_dy:
                above, up_dy = j, abs(dy)
            # below neighbor
            if dy > 0 and _horizontal_overlap(bi, bj) > 0.2 and dy < down_dy:
                below, down_dy = j, dy
        for j in [left, right, above, below]:
            if j is not None:
                adj[i][j] = 1

    # kNN fallback to avoid isolates
    C = np.array(centers)
    for i in range(N):
        if sum(adj[i]) == 0:
            d = np.sqrt(((C - C[i])**2).sum(axis=1))
            nn = np.argsort(d)[1:k_fallback+1]
            for j in nn: adj[i][int(j)] = 1

    adj_mask = np.array(adj, dtype=np.int64)
    next_idx = np.array(next_idx, dtype=np.int64)
    return adj_mask, next_idx, order

def comp_features_for_panel(panel: Dict, page_w: int, page_h: int) -> np.ndarray:
    x,y,w,h = panel['panel_coords']
    aspect = (w+1e-6)/(h+1e-6)
    size_ratio = (w*h) / (page_w*page_h + 1e-6)
    chars = panel.get('character_coords', []) or []
    char_count = len(chars)
    # Shot scale proxy: mean bbox area ratio
    if char_count > 0:
        ratios = [ (cw*ch)/(w*h+1e-6) for (cx,cy,cw,ch) in chars ]
        shot_mean = float(np.mean(ratios))
        shot_max = float(np.max(ratios))
    else:
        shot_mean = 0.0; shot_max = 0.0
    # Character positions: mean normalized center (coarse)
    if char_count > 0:
        centers = [ ((cx+cw/2 - x)/(w+1e-6), (cy+ch/2 - y)/(h+1e-6)) for (cx,cy,cw,ch) in chars ]
        mean_cx = float(np.mean([c[0] for c in centers]))
        mean_cy = float(np.mean([c[1] for c in centers]))
    else:
        mean_cx = 0.5; mean_cy = 0.5
    return np.array([aspect, size_ratio, char_count, shot_mean, shot_max, mean_cx, mean_cy], dtype=np.float32)

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", out_dim=384, freeze=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)  # returns pooled
        feat_dim = self.vit.num_features
        self.proj = nn.Linear(feat_dim, out_dim)
        if freeze:
            for p in self.vit.parameters(): p.requires_grad = False

    def forward(self, images):  # images: (B,3,224,224)
        v = self.vit(images)        # (B, feat)
        return self.proj(v)         # (B, D)

class RobertaEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", out_dim=384, freeze=True):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        hid = self.lm.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)
        if freeze:
            for p in self.lm.parameters(): p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]  # CLS token
        return self.proj(pooled)             # (B, D)

class CompositionEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.mlp(x)

class GatedFusion(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.gate = nn.Linear(3*d, 3)
    def forward(self, v, t, c):
        h = torch.cat([v,t,c], dim=-1)
        alpha = torch.softmax(self.gate(h), dim=-1)  # (B,3)
        return alpha[:,0:1]*v + alpha[:,1:2]*t + alpha[:,2:3]*c

class PanelAtomizerLite(nn.Module):
    def __init__(self, comp_in_dim=7, d=384):
        super().__init__()
        self.vision = ViTEncoder(out_dim=d, freeze=True)
        self.text = RobertaEncoder(out_dim=d, freeze=True)
        self.comp = CompositionEncoder(comp_in_dim, out_dim=d)
        self.fuse = GatedFusion(d=d)

    def forward(self, images, input_ids, attention_mask, comp_feats):
        V = self.vision(images)
        T = self.text(input_ids, attention_mask)
        C = self.comp(comp_feats)
        P = self.fuse(V, T, C)  # (B,D)
        return P

class GutterSeqLite(nn.Module):
    def __init__(self, d=384, nhead=6, layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4*d, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, 64, d))  # support up to 64 panels/segment

    def forward(self, P_seq, attn_mask=None):
        # P_seq: (B,N,D), attn_mask: (B,N) with 1 for valid, 0 for pad
        B,N,D = P_seq.shape
        x = P_seq + self.pos[:,:N,:]
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask.bool()  # True where padding
        S = self.enc(x, src_key_padding_mask=key_padding_mask)
        return S

class StoryHANLite(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.page_rnn = nn.GRU(d, d//2, batch_first=True, bidirectional=True)
        self.page_q = nn.Parameter(torch.randn(1,1,d))
        self.page_proj = nn.Linear(d, d)

    def panels_to_page(self, S_seq, mask):
        # S_seq: (B,N,D), mask: (B,N)
        h,_ = self.page_rnn(S_seq)  # (B,N,D)
        q = self.page_q.expand(h.size(0), -1, -1)           # (B,1,D)
        scores = torch.matmul(self.page_proj(h), q.transpose(-1,-2)).squeeze(-1)  # (B,N)
        scores = scores.masked_fill(~mask.bool(), float('-inf'))
        w = torch.softmax(scores, dim=-1)                   # (B,N)
        E_page = (w.unsqueeze(-1) * h).sum(dim=1)           # (B,D)
        return E_page, w

class NextPanelHead(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.scorer = nn.Bilinear(d, d, 1)
    def forward(self, S_seq):  # (B,N,D)
        # Pairwise scores (B,N,N) via bilinear
        B,N,D = S_seq.shape
        Si = S_seq.unsqueeze(2).expand(B,N,N,D)
        Sj = S_seq.unsqueeze(1).expand(B,N,N,D)
        logits = self.scorer(Si, Sj).squeeze(-1)
        return logits  # (B,N,N)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def loss_mpm(S_seq, P_seq, masked_idxs, temperature=0.07):
    # InfoNCE: predict original panel embedding from masked S representation
    B,N,D = S_seq.shape
    q = []
    pos = []
    for b in range(B):
        mi = masked_idxs[b]
        q.append(S_seq[b, mi])
        pos.append(P_seq[b, mi].detach())
    q = F.normalize(torch.stack(q), dim=-1)    # (B,D)
    pos = F.normalize(torch.stack(pos), dim=-1)
    # negatives: all P in batch except positives
    P_all = F.normalize(P_seq.reshape(B*N, D), dim=-1)  # includes positives, fine for simplicity
    logits = q @ P_all.t() / temperature                # (B, B*N)
    labels = torch.arange(B, device=q.device) * N + torch.tensor(masked_idxs, device=q.device)
    return F.cross_entropy(logits, labels)

def loss_pop(E_page_first, E_page_second, labels):
    # labels: 0=forward, 1=swapped, 2=unrelated
    x = torch.cat([E_page_first, E_page_second, torch.abs(E_page_first - E_page_second)], dim=-1)
    clf = nn.Sequential(nn.Linear(x.size(-1), 256), nn.ReLU(), nn.Linear(256, 3)).to(x.device)
    return F.cross_entropy(clf(x), labels)

def loss_rpp(logits_neighbors, next_idx, adj_mask):
    # logits_neighbors: (B,N,N); next_idx: (B,N); adj_mask: (B,N,N)
    logits = logits_neighbors.masked_fill(adj_mask==0, float('-inf'))
    B,N,_ = logits.shape
    losses = []
    for b in range(B):
        # per-node CE over neighbors; ignore if next_idx=-100
        valid_nodes = (next_idx[b] != -100).nonzero().flatten()
        if len(valid_nodes)==0: continue
        l = F.cross_entropy(logits[b][valid_nodes], next_idx[b][valid_nodes])
        losses.append(l)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)

# ============================================================================
# MAIN MODEL
# ============================================================================

class ClosureLite(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.atom = PanelAtomizerLite(comp_in_dim=7, d=d)
        self.seq = GutterSeqLite(d=d, nhead=6, layers=2)
        self.han = StoryHANLite(d=d)
        self.next_head = NextPanelHead(d=d)

    def forward(self, batch):
        B,N,_,_,_ = batch['images'].shape
        images = batch['images'].flatten(0,1)                 # (B*N,3,224,224)
        input_ids = batch['input_ids'].flatten(0,1)           # (B*N,L)
        attention_mask = batch['attention_mask'].flatten(0,1)
        comp_feats = batch['comp_feats'].flatten(0,1)         # (B*N,7)
        # Panel embeddings
        P_flat = self.atom(images, input_ids, attention_mask, comp_feats)  # (B*N,D)
        P = P_flat.view(B, N, -1)
        # Mask one panel per page for MPM
        masked_idxs = [int(torch.randint(0, batch['panel_mask'][b].sum(), (1,))) for b in range(B)]
        # For simplicity, no token-level masking; use S prediction to retrieve P_true in InfoNCE
        S = self.seq(P, attn_mask=batch['panel_mask'])
        E_page, _ = self.han.panels_to_page(S, batch['panel_mask'])
        logits_neighbors = self.next_head(S)

        # Losses
        L_mpm = loss_mpm(S, P, masked_idxs)
        # POP-lite: sample positives/negatives by shuffling E_page within batch
        E2 = E_page[torch.randperm(B)]
        labels = torch.zeros(B, dtype=torch.long, device=E_page.device)
        labels[E2.eq(E_page).all(dim=-1)] = 2  # degenerate identical → mark as unrelated
        L_pop = loss_pop(E_page, E2, labels)
        L_rpp = loss_rpp(logits_neighbors, batch['next_idx'], batch['adj_mask'])

        L = L_mpm + 0.3*L_pop + 0.5*L_rpp
        return L, {'L_mpm': L_mpm.item(), 'L_pop': L_pop.item(), 'L_rpp': L_rpp.item()}

if __name__ == "__main__":
    print("CLOSURE-Lite Framework loaded successfully!")
    print("Ready for training on DataSpec v0.3 format.")
