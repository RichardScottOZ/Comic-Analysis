Perfect—that’s enough to get a lean build running locally. Below is a “CLOSURE-Lite” variant tailored for a single 16GB A5000, using ViT+RoBERTa and inferring adjacency/reading order directly from panel boxes.

What you’ll get here
- A 16GB-ready model config (ViT-S + RoBERTa-base, D=384)
- Reading-order + adjacency inference (no adjacency in JSON)
- Composition feature extractor
- Minimal modules: PanelAtomizer (gated fusion), GUTTER-Seq (tiny Transformer), Story-HAN (attn pooling)
- Self-supervised losses (MPM, POP-lite, RPP-lite)
- Training recipe and memory tips
- When to consider AWS

CLOSURE-Lite (16GB) configuration
- Vision: ViT-S/16 (timm: vit_small_patch16_224, pretrained ImageNet)
- Text: RoBERTa-base (HuggingFace), frozen initially
- Dimensionality: D=384
- Fusion: Gated fusion on pooled features (memory-cheap)
- Seq encoder: 2-layer Transformer encoder (nhead=6)
- Aggregation: GRU + attention pooling
- Max panels per page (train-time cap): 12
- Batch size: 4 pages, grad_accum=4 (effective 16)
- Mixed precision: on
- Backbones: frozen in v0 (optionally unfreeze later or LoRA)

1) Reading order + adjacency inference (no adjacency provided)
- Reading order: standard LTR Z-path with robust row grouping by vertical overlap. Optional RTL switch for manga.
- Adjacency (for RPP): nearest valid neighbor to the left/right/above/below (overlap-constrained) + next-in-reading-order. Fallback: kNN in 2D center space to avoid isolates.

Code — reading order + adjacency
```python
import math
from typing import List, Dict, Tuple

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
    import numpy as np
    C = np.array(centers)
    for i in range(N):
        if sum(adj[i]) == 0:
            d = np.sqrt(((C - C[i])**2).sum(axis=1))
            nn = np.argsort(d)[1:k_fallback+1]
            for j in nn: adj[i][int(j)] = 1

    import numpy as np
    adj_mask = np.array(adj, dtype=np.int64)
    next_idx = np.array(next_idx, dtype=np.int64)
    return adj_mask, next_idx, order
```

2) Composition features (structured Vcomp)
```python
import numpy as np

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
```

3) Model modules (lite)
```python
# pip install torch timm transformers einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel

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
```

RPP-lite head and losses (memory-cheap)
```python
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
```

4) Minimal Dataset and collate
```python
from PIL import Image
import json, torch
import torchvision.transforms as T
from transformers import AutoTokenizer

class ComicsPageDataset(torch.utils.data.Dataset):
    def __init__(self, json_paths, image_root, max_panels=12, rtl=False, text_model='roberta-base'):
        self.pages = []
        for jp in json_paths:
            with open(jp, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): data = [data]
                self.pages.extend(data)
        self.image_root = image_root
        self.max_panels = max_panels
        self.rtl = rtl
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])

    def __len__(self): return len(self.pages)

    def __getitem__(self, idx):
        page = self.pages[idx]
        img = Image.open(page['page_image_path']).convert('RGB')
        W, H = img.size
        panels = page['panels']
        # crop panels and build per-panel data
        crops, texts, comps, boxes = [], [], [], []
        for p in panels[:self.max_panels]:
            x,y,w,h = p['panel_coords']
            crop = img.crop((x, y, x+w, y+h))
            crops.append(self.tf(crop))
            # aggregate text
            tdict = p.get('text', {}) or {}
            parts = []
            for k in ('dialogue','narration','sfx'):
                vals = tdict.get(k) or []
                parts.extend([v for v in vals if v])
            text = ' | '.join(parts) if parts else ''
            texts.append(text)
            comps.append(comp_features_for_panel(p, W, H))
            boxes.append([x/W, y/H, w/W, h/H])

        # reading order + adjacency
        adj_mask, next_idx, order = build_adjacency_and_next(panels[:len(crops)], W, H, rtl=self.rtl)
        # pad to max_panels
        N = len(crops)
        padN = self.max_panels - N
        if padN > 0:
            pad_img = torch.zeros(3,224,224)
            crops += [pad_img]*padN
            texts += ['']*padN
            comps += [np.zeros(7, dtype=np.float32)]*padN
            boxes += [[0,0,0,0]]*padN
            import numpy as np
            adj_pad = np.zeros((self.max_panels, self.max_panels), dtype=np.int64)
            adj_pad[:N,:N] = adj_mask
            adj_mask = adj_pad
            next_pad = np.full((self.max_panels,), -100, dtype=np.int64)
            next_pad[:N] = next_idx
            next_idx = next_pad

        # tokenize text
        tok = self.tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        batch = {
            'images': torch.stack(crops),           # (N,3,224,224)
            'input_ids': tok['input_ids'],          # (N,L)
            'attention_mask': tok['attention_mask'],
            'comp_feats': torch.tensor(np.stack(comps), dtype=torch.float32),  # (N,7)
            'boxes': torch.tensor(boxes, dtype=torch.float32),                 # (N,4)
            'panel_mask': torch.zeros(self.max_panels, dtype=torch.bool).index_fill_(0, torch.arange(N), True),
            'adj_mask': torch.tensor(adj_mask, dtype=torch.long),              # (N,N)
            'next_idx': torch.tensor(next_idx, dtype=torch.long),              # (N,)
        }
        return batch

def collate_pages(batch_list):
    # each item is a page dict
    keys = batch_list[0].keys()
    out = {}
    for k in keys:
        if k in ('panel_mask', 'next_idx'):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N) or (B,N)
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N,...) or (B,N,N)
    return out
```

5) Training step (v0)
```python
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
```

Training recipe for 16GB A5000
- Environment:
  - pip install torch torchvision timm transformers einops faiss-gpu
- Flags:
  - Enable AMP (torch.cuda.amp.autocast)
  - torch.backends.cudnn.benchmark = True
  - Use channels_last memory format for images
- Hyperparams:
  - lr: 3e-4 (AdamW, weight_decay 0.05), cosine decay, 2k warmup
  - batch_size: 4 pages, grad_accum 4
  - epochs: start with 3–5 epochs on a few thousand pages
- Freezing:
  - Keep ViT and RoBERTa frozen for v0; train atomizer MLPs, fusion, seq, HAN, heads
  - Optionally unfreeze last ViT block later (small lr 1e-5) if you move to AWS

Memory tips
- Keep panel cap to 12 at train time; chunk longer pages
- Mixed precision and gradient checkpointing on GutterSeq if needed
- Cache tokenization and panel crops to disk to speed up I/O
- Precompute/freeze P on a large corpus and train only seq layers as a second stage to save memory

When to consider AWS
- If you want:
  - Unfreeze ViT and RoBERTa fully
  - Larger D (768), longer sequences (>16 panels), larger batches
- A cost-effective step-up: AWS g5.xlarge (A10 24GB) or g6.xlarge (L4 24GB)
- Scale runs (multi-GPU): g5.12xlarge (4x A10 24GB) or p4d.24xlarge (8x A100 40GB)

What I’d do next (quick wins)
- Share 2–3 sample JSON pages + images; I’ll tailor the dataset/reading-order thresholds
- Run a smoke test: forward pass on 8–16 pages end-to-end, verify loss values decrease
- Build a tiny retrieval demo: index S_i embeddings with FAISS and try narrative-aware queries via text encoder

Optional upgrades after v0
- 2D layout bias in attention (learned bias from pairwise box deltas)
- Learned composition captions → feed through RoBERTa as Vcomp
- PATH-GNN (graph head) if pages with non-Z layouts are common
- POP with curated positives/negatives (true consecutive vs. swapped vs. unrelated segments)

If you drop a couple of page JSONs and their images, I can wire a ready-to-run training script that fits the A5000 on first try.