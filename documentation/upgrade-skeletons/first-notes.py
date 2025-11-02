Awesome. Let’s turn the report into concrete, buildable frameworks. Below is a practical, implementation-ready “family” that follows your blueprint, with code skeletons, training objectives, and a data pipeline you can plug into your existing JSON outputs.

Deliverables in this response
- CLOSURE family (four interoperable frameworks, each standalone-capable)
- End-to-end pipeline spec (data schema, preprocessing, batching)
- PyTorch module skeletons for all core components
- Self-supervised training objectives (MPM, POP, RPP) and losses
- Inference and retrieval APIs (panel/page/comic)
- Evaluation harness outline and config templates
- Roadmap and compute estimates

Part I — The CLOSURE Family (buildable frameworks)
1) CLOSURE-Panel: Atomic panel embeddings with tri-modal deconstruction + cross-modal fusion
- Purpose: Produce a rich per-panel vector that combines visual, text (OCR), and composition/layout semantics.
- Input: panel image (RGB), aggregated panel text, structured composition features.
- Output: panel embedding P_i ∈ R^D
- Highlights:
  - Visual encoder: ViT or CLIP/EVA-CLIP vision encoder
  - Text encoder: BERT/RoBERTa or CLIP text encoder
  - Composition encoder: structured MLP or learned “composition captioner”
  - TriFuse: cross-modal attention + gated fusion

2) GUTTER-Seq: 2D-layout-aware sequential encoder for narrative context
- Purpose: Convert a sequence of panel embeddings into context-aware sequential embeddings S_i that encode narrative flow and inferred temporality.
- Input: sequence of panel embeddings [P_1..P_N], 2D panel positions and adjacency
- Output: [S_1..S_N]
- Highlights:
  - Bidirectional Transformer encoder
  - Relative positional bias that mixes (1D order + 2D layout proximity)
  - Self-supervised objectives: MPM (masked panel), POP (panel order), RPP (reading-path) with adjacency supervision

3) STORY-HAN: Hierarchical aggregation to pages and books
- Purpose: Aggregate panels → page embeddings E_page, and pages → comic embedding E_comic with attention.
- Input: [S_i] per page → E_page; [E_page] per comic → E_comic
- Output: E_page, E_comic
- Highlights:
  - Page encoder with attention pooling over panels
  - Comic encoder with attention pooling over pages
  - Interpretable attention weights (narrative “highlights”)

4) PATH-GNN: Graph-style reading path predictor
- Purpose: Explicitly learn the reading flow on complex layouts (parallel to GUTTER-Seq’s RPP head, but graph-native).
- Input: Per-page graph G=(V=panels, E=adjacency/overlaps), node features = P_i
- Output: probability distribution over next-panel edges; optional full path decoding
- Highlights:
  - Graph Transformer or GAT with edge features (spatial relations)
  - Train jointly with GUTTER-Seq or standalone as an auxiliary task

Part II — End-to-End Architecture Diagram (conceptual)
Panel image + OCR + layout JSON
  → CLOSURE-Panel (P_i)
  → GUTTER-Seq (S_i)
  → STORY-HAN Page (E_page)
  → STORY-HAN Comic (E_comic)
  → Retrieval, similarity search, classification, change-point detection, etc.

Part III — Data Schema and Pipeline
Assumed JSON keys per page (compatible with batch_comic_analysis_multi.py):
- page_id: str
- page_image_path: str
- panels: list of
  - panel_id: str
  - panel_coords: [x, y, w, h] (absolute pixels)
  - text: {dialogue: [str], narration: [str], sfx: [str]}
  - character_coords: list of [x, y, w, h]
  - neighbors: dict {left: [panel_id], right: [...], above: [...], below: [...]}
  - order_index: int (if available)

Preprocessing steps
- Crop panel images via panel_coords
- Aggregate text per panel: text_str = " | ".join(dialogue + narration + sfx)
- Structured composition features (per panel):
  - aspect_ratio = w/h
  - character_count
  - shot_scale stats: mean(area(character)/area(panel))
  - positions normalized per panel: xi_rel = (xi - x0)/w, yi_rel = (yi - y0)/h, wi_rel, hi_rel
  - panel size ratio on page, relative z-order if present
- Assemble sequences per page (or per scene) using order_index or your reading order solver
- For RPP, build adjacency list for each panel using neighbors

Part IV — Core Modules (PyTorch skeletons)
Dependencies
- torch, torchvision, timm (for ViT), transformers (HF), faiss (for retrieval), pytorch-lightning (optional)

1) Encoders and Atomization

```python
# pip install torch timm transformers einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CompositionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, comp_feats):
        # comp_feats: (B, F) structured features
        return self.mlp(comp_feats)  # (B, Cc)

class GatedFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dv, dt, dc, dout = dims
        self.proj_v = nn.Linear(dv, dout)
        self.proj_t = nn.Linear(dt, dout)
        self.proj_c = nn.Linear(dc, dout)
        self.gate = nn.Linear(3 * dout, 3)  # learn weights per modality

    def forward(self, v, t, c):
        v, t, c = self.proj_v(v), self.proj_t(t), self.proj_c(c)
        alpha = torch.softmax(self.gate(torch.cat([v, t, c], dim=-1)), dim=-1)  # (B,3)
        out = alpha[:,0:1]*v + alpha[:,1:2]*t + alpha[:,2:3]*c
        return out  # (B, D)

class CrossModalTriFuse(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.xattn = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))

    def forward(self, vis_tokens, txt_tokens, comp_tokens):
        # vis_tokens: (B, Tv, D)  text_tokens: (B, Tt, D)  comp_tokens: (B, Tc, D)
        B = vis_tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, vis_tokens, txt_tokens, comp_tokens], dim=1)
        h = self.xattn(seq)  # (B, 1+Tv+Tt+Tc, D)
        return h[:,0]  # fused representation (B, D)

class PanelAtomizer(nn.Module):
    def __init__(self, vision_backbone, text_backbone, comp_in_dim, out_dim=768, fuse='cross'):
        super().__init__()
        self.visual = vision_backbone  # returns token-level and/or pooled
        self.text = text_backbone      # returns token-level and/or pooled
        self.comp_enc = CompositionEncoder(comp_in_dim, out_dim)
        self.fuse = fuse
        if fuse == 'gated':
            self.fuser = GatedFusion((out_dim, out_dim, out_dim, out_dim))
        else:
            self.fuser = CrossModalTriFuse(d_model=out_dim)

    def forward(self, panel_img, panel_text_ids, comp_feats, attn_masks=None):
        # visual tokens
        v_tokens, v_pooled = self.visual(panel_img)         # (B, Tv, D), (B, D)
        t_tokens, t_pooled = self.text(panel_text_ids)      # (B, Tt, D), (B, D)
        c_vec = self.comp_enc(comp_feats)                   # (B, D)
        if self.fuse == 'gated':
            return self.fuser(v_pooled, t_pooled, c_vec)    # (B, D)
        else:
            c_tokens = c_vec.unsqueeze(1)                   # (B,1,D) as a single token
            return self.fuser(v_tokens, t_tokens, c_tokens) # (B, D)
```

Notes
- vision_backbone: use a ViT that returns both patch tokens and pooled CLS (e.g., timm.create_model("vit_base_patch16_224", pretrained=True), modify forward to return tokens).
- text_backbone: any HF encoder returning token and pooled outputs (e.g., RobertaModel). If using CLIP, adapt shapes.
- comp_in_dim: depends on your structured feature vector length.

2) GUTTER-Seq: Layout-aware Transformer

```python
class LayoutBias(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # Map relative spatial relations into additive attention bias
        self.mlp = nn.Sequential(
            nn.Linear(6, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, boxes):  # boxes: (B, N, 4) in normalized page coords
        # Compute pairwise deltas: dx, dy, dw, dh, center_dist, IoU or overlap flags
        B, N, _ = boxes.shape
        centers = torch.stack([boxes[...,0]+boxes[...,2]/2, boxes[...,1]+boxes[...,3]/2], dim=-1)  # (B,N,2)
        dx = centers[:,:,None,0] - centers[:,None,:,0]
        dy = centers[:,:,None,1] - centers[:,None,:,1]
        dw = torch.log((boxes[:,:,None,2]+1e-6)/(boxes[:,None,:,2]+1e-6))
        dh = torch.log((boxes[:,:,None,3]+1e-6)/(boxes[:,None,:,3]+1e-6))
        dist = torch.sqrt(dx**2 + dy**2)
        feats = torch.stack([dx, dy, dw, dh, dist, (dist<0.1).float()], dim=-1)  # (B,N,N,6)
        bias = self.mlp(feats).squeeze(-1)  # (B,N,N)
        return bias

class GutterSeq(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layout_bias = LayoutBias()
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model))  # 1D for long sequences

    def forward(self, P_seq, panel_boxes):
        # P_seq: (B, N, D), panel_boxes: (B, N, 4) normalized [x,y,w,h]
        B, N, D = P_seq.shape
        bias = self.layout_bias(panel_boxes)  # (B,N,N)
        h = P_seq + self.pos_emb[:,:N,:]
        # Apply attention with additive bias
        # nn.Transformer doesn't expose bias directly; implement custom or simulate via attention mask
        # For skeleton: encode without bias, then optionally mix bias through residual MLP
        S = self.encoder(h)  # (B,N,D)
        return S  # context-aware panel embeddings
```

3) STORY-HAN: Hierarchical aggregators

```python
class AttnPool(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1,1,d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, seq):
        # seq: (B, T, D)
        q = self.q.expand(seq.size(0), -1, -1)   # (B,1,D)
        scores = torch.matmul(self.proj(seq), q.transpose(-1,-2)).squeeze(-1)  # (B,T)
        w = torch.softmax(scores, dim=-1)
        pooled = (w.unsqueeze(-1) * seq).sum(dim=1)  # (B,D)
        return pooled, w  # weights for interpretability

class StoryHAN(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.page_encoder = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.page_pool = AttnPool(d_model)
        self.comic_encoder = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.comic_pool = AttnPool(d_model)

    def panels_to_page(self, S_seq):
        h,_ = self.page_encoder(S_seq)  # (B,N,D)
        E_page, alpha_panels = self.page_pool(h)
        return E_page, alpha_panels

    def pages_to_comic(self, E_pages_seq):
        h,_ = self.comic_encoder(E_pages_seq)  # (B,M,D)
        E_comic, alpha_pages = self.comic_pool(h)
        return E_comic, alpha_pages
```

4) PATH-GNN: Reading path predictor

```python
class PathGNN(nn.Module):
    def __init__(self, d_model=768, num_layers=2, heads=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, heads, batch_first=True)
                                     for _ in range(num_layers)])
        self.edge_mlp = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, P_nodes, adj_mask):
        # P_nodes: (B, N, D); adj_mask: (B, N, N) {1 if adjacent}
        h = P_nodes
        for layer in self.layers:
            h = layer(h)
        # Next-panel scores for each node among its neighbors
        H_i = h.unsqueeze(2).expand(-1,-1,h.size(1),-1)
        H_j = h.unsqueeze(1).expand(-1,h.size(1),-1,-1)
        pair = torch.cat([H_i, H_j], dim=-1)  # (B,N,N,2D)
        logits = self.edge_mlp(pair).squeeze(-1)  # (B,N,N)
        logits = logits.masked_fill(adj_mask==0, float('-inf'))
        return logits  # softmax over neighbors per node
```

5) Objectives and Losses

```python
class Objectives(nn.Module):
    def __init__(self, d_model=768, queue_size=4096, temp=0.07):
        super().__init__()
        self.temp = temp
        self.register_buffer('queue', torch.randn(queue_size, d_model))
        self.queue = F.normalize(self.queue, dim=-1)
        self.q_ptr = 0

    def _enqueue(self, z):
        k = z.detach()
        bs = k.size(0)
        end = (self.q_ptr + bs) % self.queue.size(0)
        if self.q_ptr + bs <= self.queue.size(0):
            self.queue[self.q_ptr:self.q_ptr+bs] = k
        else:
            first = self.queue.size(0) - self.q_ptr
            self.queue[self.q_ptr:] = k[:first]
            self.queue[:end] = k[first:]
        self.q_ptr = end

    def masked_panel_modeling(self, S_seq, masked_idx, target_vec):
        # InfoNCE: query S_seq[:,masked_idx], positives=target_vec, negatives=queue
        q = F.normalize(S_seq[torch.arange(S_seq.size(0)), masked_idx], dim=-1)  # (B,D)
        pos = F.normalize(target_vec, dim=-1)                                    # (B,D)
        queue = self.queue                                                       # (K,D)
        logits_pos = (q * pos).sum(-1) / self.temp
        logits_neg = q @ queue.t() / self.temp
        logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        self._enqueue(pos)
        return loss

    def panel_order_prediction(self, E_first, E_second):
        # classify relation: forward, swapped, unrelated
        logits = torch.cat([E_first, E_second, torch.abs(E_first-E_second)], dim=-1)
        logits = nn.Sequential(nn.Linear(logits.size(-1), 512), nn.ReLU(), nn.Linear(512, 3)).to(E_first.device)(logits)
        labels = ...  # supply
        return F.cross_entropy(logits, labels)

    def reading_path_prediction(self, logits_neighbors, gt_next_idx):
        # logits_neighbors: (B,N,N), gt_next_idx: (B,N) next neighbor index
        losses = []
        for b in range(logits_neighbors.size(0)):
            loss = F.cross_entropy(logits_neighbors[b], gt_next_idx[b])  # per node softmax over neighbors
            losses.append(loss)
        return torch.stack(losses).mean()
```

Part V — Training Loop (pseudo)
- Inputs per batch:
  - B pages (or short story segments)
  - For each page: N panels with images, text tokens, comp features, layout boxes, adjacency

Forward
1. P_i = CLOSURE-Panel(panel_img, panel_text, comp_feats)
2. S_i = GUTTER-Seq(P_seq, panel_boxes)
3. E_page = STORY-HAN.panels_to_page(S_seq)
4. Optional multi-page: E_comic = STORY-HAN.pages_to_comic(E_page_seq)
5. PATH-GNN logits = PathGNN(P_nodes, adj_mask)

Losses
- L_mpm = Objectives.masked_panel_modeling(S_seq, masked_idx, target_vec=P_masked_original)
- L_pop = Objectives.panel_order_prediction(E_page_first, E_page_second)
- L_rpp = Objectives.reading_path_prediction(path_logits, gt_next_neighbors)
- Total: L = w1*L_mpm + w2*L_pop + w3*L_rpp (use uncertainty weighting or GradNorm for stability)

Optimization
- AdamW, lr warmup + cosine decay
- Mixed precision recommended
- Gradient checkpointing on Transformers

Part VI — Inference and APIs
- embed_panel(image, text, comp): returns P
- embed_page(page_json): returns [S_i], E_page, attention over panels
- embed_comic(comic_json): returns E_comic, attention over pages
- search(query):
  - If text: encode via text backbone to query vector; retrieve panels/pages/comics via cosine similarity
  - If panel image: encode to P or S and retrieve
- Indexing: FAISS (IVFPQ or HNSW), multi-scale indices for panels, pages, comics
- Narrative-aware search: Use S_i or E_page/E_comic embeddings; optionally re-rank with sequential coherence scoring

Part VII — Evaluation Harness
Intrinsic
- UMAP/t-SNE of S_i and E_page/E_comic
- Cluster quality (silhouette, DB-index)
- Vector analogies on small curated sets

Extrinsic
- Narrative similarity search with human judgments → nDCG, mAP
- Genre/artist/era classification from E_comic (SVM or MLP)
- Change-point detection on S_i trajectories → precision/recall vs. annotated beats
- Character clustering and sentiment/style analysis across panels with the same entity

Part VIII — Config Template (YAML)
```yaml
model:
  d_model: 768
  fusion: cross
  seq_layers: 6
  heads: 12
  page_encoder: gru
  comic_encoder: gru

backbones:
  vision: vit_base_patch16_224
  text: roberta-base

loss_weights:
  mpm: 1.0
  pop: 0.5
  rpp: 0.5

training:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.05
  warmup_steps: 2000
  max_steps: 200000
  batch_size_panels: 64
  amp: true
  grad_clip: 1.0

data:
  json_root: /data/comics/json
  image_root: /data/comics/images
  max_panels_per_page: 12
  text_max_tokens: 128
```

Part IX — Minimal Compute Plans
- M0 (feasibility): ViT-S/16 + DistilRoBERTa, 2 seq layers, batch size 32 pages → 1x A100 40GB or 2x 24GB GPUs
- M1 (research): ViT-B/16 + RoBERTa-base, 6 seq layers, batch size 64 pages → 4x 40GB GPUs
- M2 (scaling): EVA-CLIP-L + DeBERTa-v3-large, 12 seq layers, 2D bias, larger negatives queue → 8x 80GB GPUs

Part X — What You Can Run First (2–3 weeks)
Week 1
- Implement PanelAtomizer with structured CompositionEncoder and GatedFusion
- Train image-text-comp panel embedding with a simple contrastive objective (align text↔visual, auxiliary for comp)
- Build dataset + dataloader that reads your JSON, crops panels, assembles sequences

Week 2
- Add GUTTER-Seq Transformer, train masked panel modeling (MPM) with in-batch negatives
- Add POP (two mini-batches with swapped order) and RPP on easy layouts

Week 3
- STORY-HAN page/comic aggregators + attention visualizations
- Build FAISS indices for multi-scale search
- Run intrinsic evals and one extrinsic pilot (genre/era classification)

Optional future add-ons
- Learned composition captions: use a captioner fine-tuned to generate composition descriptions; feed through text encoder as V_comp
- PATH-GNN joint training for complex layouts
- Cross-lingual OCR normalization + multilingual text encoder

Open Questions for You
- Which base backbones do you prefer (CLIP vs. ViT+RoBERTa)? Any licensing constraints?
- Do you already have adjacency (neighbors) in your JSON, or should we infer via geometric heuristics?
- What compute do you have on hand? We’ll tune the model size accordingly.
- Do you want us to prioritize narrative retrieval or reading-path accuracy first?

If you share a small sample JSON and a handful of page images, I can tailor the data loader and give you a runnable v0 training script next.

# pip install torch timm transformers einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CompositionEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, comp_feats):
        # comp_feats: (B, F) structured features
        return self.mlp(comp_feats)  # (B, Cc)

class GatedFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        dv, dt, dc, dout = dims
        self.proj_v = nn.Linear(dv, dout)
        self.proj_t = nn.Linear(dt, dout)
        self.proj_c = nn.Linear(dc, dout)
        self.gate = nn.Linear(3 * dout, 3)  # learn weights per modality

    def forward(self, v, t, c):
        v, t, c = self.proj_v(v), self.proj_t(t), self.proj_c(c)
        alpha = torch.softmax(self.gate(torch.cat([v, t, c], dim=-1)), dim=-1)  # (B,3)
        out = alpha[:,0:1]*v + alpha[:,1:2]*t + alpha[:,2:3]*c
        return out  # (B, D)

class CrossModalTriFuse(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.xattn = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))

    def forward(self, vis_tokens, txt_tokens, comp_tokens):
        # vis_tokens: (B, Tv, D)  text_tokens: (B, Tt, D)  comp_tokens: (B, Tc, D)
        B = vis_tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, vis_tokens, txt_tokens, comp_tokens], dim=1)
        h = self.xattn(seq)  # (B, 1+Tv+Tt+Tc, D)
        return h[:,0]  # fused representation (B, D)

class PanelAtomizer(nn.Module):
    def __init__(self, vision_backbone, text_backbone, comp_in_dim, out_dim=768, fuse='cross'):
        super().__init__()
        self.visual = vision_backbone  # returns token-level and/or pooled
        self.text = text_backbone      # returns token-level and/or pooled
        self.comp_enc = CompositionEncoder(comp_in_dim, out_dim)
        self.fuse = fuse
        if fuse == 'gated':
            self.fuser = GatedFusion((out_dim, out_dim, out_dim, out_dim))
        else:
            self.fuser = CrossModalTriFuse(d_model=out_dim)

    def forward(self, panel_img, panel_text_ids, comp_feats, attn_masks=None):
        # visual tokens
        v_tokens, v_pooled = self.visual(panel_img)         # (B, Tv, D), (B, D)
        t_tokens, t_pooled = self.text(panel_text_ids)      # (B, Tt, D), (B, D)
        c_vec = self.comp_enc(comp_feats)                   # (B, D)
        if self.fuse == 'gated':
            return self.fuser(v_pooled, t_pooled, c_vec)    # (B, D)
        else:
            c_tokens = c_vec.unsqueeze(1)                   # (B,1,D) as a single token
            return self.fuser(v_tokens, t_tokens, c_tokens) # (B, D)



class LayoutBias(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        # Map relative spatial relations into additive attention bias
        self.mlp = nn.Sequential(
            nn.Linear(6, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, boxes):  # boxes: (B, N, 4) in normalized page coords
        # Compute pairwise deltas: dx, dy, dw, dh, center_dist, IoU or overlap flags
        B, N, _ = boxes.shape
        centers = torch.stack([boxes[...,0]+boxes[...,2]/2, boxes[...,1]+boxes[...,3]/2], dim=-1)  # (B,N,2)
        dx = centers[:,:,None,0] - centers[:,None,:,0]
        dy = centers[:,:,None,1] - centers[:,None,:,1]
        dw = torch.log((boxes[:,:,None,2]+1e-6)/(boxes[:,None,:,2]+1e-6))
        dh = torch.log((boxes[:,:,None,3]+1e-6)/(boxes[:,None,:,3]+1e-6))
        dist = torch.sqrt(dx**2 + dy**2)
        feats = torch.stack([dx, dy, dw, dh, dist, (dist<0.1).float()], dim=-1)  # (B,N,N,6)
        bias = self.mlp(feats).squeeze(-1)  # (B,N,N)
        return bias

class GutterSeq(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layout_bias = LayoutBias()
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model))  # 1D for long sequences

    def forward(self, P_seq, panel_boxes):
        # P_seq: (B, N, D), panel_boxes: (B, N, 4) normalized [x,y,w,h]
        B, N, D = P_seq.shape
        bias = self.layout_bias(panel_boxes)  # (B,N,N)
        h = P_seq + self.pos_emb[:,:N,:]
        # Apply attention with additive bias
        # nn.Transformer doesn't expose bias directly; implement custom or simulate via attention mask
        # For skeleton: encode without bias, then optionally mix bias through residual MLP
        S = self.encoder(h)  # (B,N,D)
        return S  # context-aware panel embeddings


model:
  d_model: 768
  fusion: cross
  seq_layers: 6
  heads: 12
  page_encoder: gru
  comic_encoder: gru

backbones:
  vision: vit_base_patch16_224
  text: roberta-base

loss_weights:
  mpm: 1.0
  pop: 0.5
  rpp: 0.5

training:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.05
  warmup_steps: 2000
  max_steps: 200000
  batch_size_panels: 64
  amp: true
  grad_clip: 1.0

data:
  json_root: /data/comics/json
  image_root: /data/comics/images
  max_panels_per_page: 12
  text_max_tokens: 128        