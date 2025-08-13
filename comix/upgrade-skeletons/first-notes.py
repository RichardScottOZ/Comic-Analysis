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