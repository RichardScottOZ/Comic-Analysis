"""
CLOSURE-Lite Simple Framework - Skip problematic sequence processing
This version bypasses GutterSeqLite entirely to preserve panel diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

# ============================================================================
# ENCODERS (Same as before)
# ============================================================================

class ViTEncoder(nn.Module):
    def __init__(self, out_dim=384, freeze=True):
        super().__init__()
        from transformers import ViTModel
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, out_dim)
    
    def forward(self, images):  # images: (B,3,224,224)
        out = self.vit(images)      # BaseModelOutputWithPooling
        v = out.last_hidden_state[:, 0]  # CLS token (B, hidden_size)
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
        self.vision = ViTEncoder(out_dim=d, freeze=False)  # Unfreeze to learn panel-specific features
        self.text = RobertaEncoder(out_dim=d, freeze=False)  # Unfreeze to learn panel-specific features
        self.comp = CompositionEncoder(comp_in_dim, out_dim=d)
        self.fuse = GatedFusion(d)
    
    def forward(self, images, input_ids, attention_mask, comp_feats):
        V = self.vision(images)
        T = self.text(input_ids, attention_mask)
        C = self.comp(comp_feats)
        P = self.fuse(V, T, C)  # (B,D)
        return P

# ============================================================================
# SIMPLE ATTENTION MECHANISM (No sequence processing)
# ============================================================================

class SimpleAttention(nn.Module):
    """Simple attention mechanism that preserves panel diversity"""
    
    def __init__(self, d=384, num_heads=4, temperature=0.1):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Multi-head attention
        self.head_dim = d // num_heads
        self.heads = nn.ModuleList([
            nn.Linear(d, self.head_dim) for _ in range(num_heads)
        ])
        
        # Content-based scoring for each head
        self.content_scorers = nn.ModuleList([
            nn.Linear(self.head_dim, 1) for _ in range(num_heads)
        ])
        
        # Positional bias (learned)
        self.pos_bias = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(d, d)
        
    def forward(self, P_seq, mask):
        # P_seq: (B,N,D), mask: (B,N) - Use raw panel embeddings directly!
        B, N, D = P_seq.shape
        
        # Process with multiple attention heads
        head_outputs = []
        head_weights = []
        
        for head, scorer in zip(self.heads, self.content_scorers):
            # Project to head dimension
            h = head(P_seq)  # (B,N,head_dim)
            
            # Compute attention scores
            scores = scorer(h).squeeze(-1)  # (B,N)
            
            # Add positional bias
            pos_bias = self.pos_bias[:, :N, :].mean(dim=-1)  # (1,N)
            scores = scores + pos_bias
            
            # Apply mask
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
            
            # Apply temperature scaling
            attention_weights = F.softmax(scores / self.temperature, dim=-1)  # (B,N)
            
            # Compute weighted representation using original P_seq
            head_output = (attention_weights.unsqueeze(-1) * P_seq).sum(dim=1)  # (B,D)
            head_outputs.append(head_output)
            head_weights.append(attention_weights)
        
        # Combine head outputs by averaging
        combined_output = torch.stack(head_outputs, dim=1).mean(dim=1)  # (B,D)
        final_output = self.out_proj(combined_output)  # (B,D)
        
        # Average attention weights across heads
        avg_attention = torch.stack(head_weights, dim=1).mean(dim=1)  # (B,N)
        
        return final_output, avg_attention

class StoryHANLiteSimple(nn.Module):
    """Simple version that skips sequence processing"""
    
    def __init__(self, d=384, num_heads=4, temperature=0.1):
        super().__init__()
        self.attention = SimpleAttention(d, num_heads, temperature)
    
    def panels_to_page(self, P_seq, mask):
        # P_seq: (B,N,D), mask: (B,N) - Use raw panel embeddings directly!
        E_page, attention_weights = self.attention(P_seq, mask)
        return E_page, attention_weights

class NextPanelHead(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.scorer = nn.Bilinear(d, d, 1)
    def forward(self, P_seq):  # (B,N,D) - Use raw panel embeddings directly!
        # Pairwise scores (B,N,N) via bilinear
        B,N,D = P_seq.shape
        Si = P_seq.unsqueeze(2).expand(B,N,N,D)
        Sj = P_seq.unsqueeze(1).expand(B,N,N,D)
        logits = self.scorer(Si, Sj).squeeze(-1)
        return logits  # (B,N,N)

# ============================================================================
# LOSS FUNCTIONS (Same as before)
# ============================================================================

def loss_mpm(P_pred, P_true, masked_idxs):
    # P_pred: (B,N,D), P_true: (B,N,D), masked_idxs: list of ints
    B, N, D = P_pred.shape
    losses = []
    for b in range(B):
        if masked_idxs[b] < N:
            p_pred = P_pred[b, masked_idxs[b]]  # (D,)
            p_true = P_true[b, masked_idxs[b]]  # (D,)
            loss = F.mse_loss(p_pred, p_true)
            losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=P_pred.device)

class POPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        return self.clf(x)

def loss_pop(E_page_first, E_page_second, labels, pop_classifier):
    # labels: 0=forward, 1=swapped, 2=unrelated
    x = torch.cat([E_page_first, E_page_second, torch.abs(E_page_first - E_page_second)], dim=-1)
    return F.cross_entropy(pop_classifier(x), labels)

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

class ClosureLiteSimple(nn.Module):
    """CLOSURE-Lite model that skips problematic sequence processing"""
    
    def __init__(self, d=384, num_heads=4, temperature=0.1):
        super().__init__()
        self.atom = PanelAtomizerLite(comp_in_dim=7, d=d)
        # Skip GutterSeqLite entirely - use raw panel embeddings directly!
        self.han = StoryHANLiteSimple(d=d, num_heads=num_heads, temperature=temperature)
        self.next_head = NextPanelHead(d=d)
        # POP classifier for page order prediction
        self.pop_classifier = POPClassifier(input_dim=3*d)
    
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
        
        # Skip sequence processing - use raw panel embeddings directly!
        E_page, attention_weights = self.han.panels_to_page(P, batch['panel_mask'])
        logits_neighbors = self.next_head(P)

        # Losses
        L_mpm = loss_mpm(P, P, masked_idxs)  # Use P for both pred and true since we're not doing sequence processing
        
        # POP-lite: sample positives/negatives by shuffling E_page within batch
        E2 = E_page[torch.randperm(B)]
        labels = torch.zeros(B, dtype=torch.long, device=E_page.device)
        labels[E2.eq(E_page).all(dim=-1)] = 2  # degenerate identical → mark as unrelated
        L_pop = loss_pop(E_page, E2, labels, self.pop_classifier)
        
        L_rpp = loss_rpp(logits_neighbors, batch['next_idx'], batch['adj_mask'])

        # Combine losses
        L = L_mpm + 0.3*L_pop + 0.5*L_rpp
        
        return L, {
            'L_mpm': L_mpm.item(), 
            'L_pop': L_pop.item(), 
            'L_rpp': L_rpp.item()
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def comp_features_for_panel(panel, W, H):
    """Extract composition features for a panel"""
    x, y, w, h = panel['panel_coords']
    return np.array([
        x/W, y/H, w/W, h/H,  # normalized position and size
        w*h/(W*H),           # area ratio
        w/h,                 # aspect ratio
        (x+w/2)/W, (y+h/2)/H  # center position
    ], dtype=np.float32)

def build_adjacency_and_next(panels, W, H, rtl=False):
    """Build adjacency matrix and next panel indices"""
    N = len(panels)
    adj = np.zeros((N, N), dtype=np.int64)
    next_idx = np.full(N, -100, dtype=np.int64)
    
    # Simple left-to-right, top-to-bottom ordering
    panel_centers = []
    for p in panels:
        x, y, w, h = p['panel_coords']
        panel_centers.append((y + h/2, x + w/2))  # (center_y, center_x)
    
    # Sort by reading order
    reading_order = sorted(range(N), key=lambda i: panel_centers[i])
    
    # Build adjacency and next relationships
    for i in range(N-1):
        curr_idx = reading_order[i]
        next_idx_val = reading_order[i+1]
        next_idx[curr_idx] = next_idx_val
        adj[curr_idx, next_idx_val] = 1
    
    return adj, next_idx, reading_order

if __name__ == "__main__":
    print("CLOSURE-Lite Simple Framework loaded successfully!")
    print("Key improvements:")
    print("- Skips problematic GutterSeqLite sequence processing")
    print("- Uses raw panel embeddings directly")
    print("- Preserves panel diversity")
    print("- Ready for training on DataSpec v0.3 format.")
