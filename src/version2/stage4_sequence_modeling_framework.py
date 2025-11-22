"""
Stage 4: Semantic Sequence Modeling (ComicsPAP & Text-Cloze Inspired)

This module implements the Stage 4 pipeline from Version 2.0 framework:
- Transformer encoder for sequential panel understanding
- ComicsPAP-inspired tasks (panel picking, character coherence, closure)
- Text-Cloze task for contextual text prediction
- Reading order refinement
- Generates contextualized panel embeddings and semantic strip embeddings

Key Features:
1. Processes sequences of panel embeddings from Stage 3
2. Models narrative flow and panel dependencies
3. Supports multiple training tasks simultaneously
4. Outputs both panel-level and strip-level representations

References:
- ComicsPAP: https://arxiv.org/abs/2503.08561
- Text-Cloze: https://arxiv.org/abs/2403.03719
- CoMix: https://github.com/emanuelevivoli/CoMix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
import numpy as np


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for panel sequences.
    Uses sinusoidal encoding to preserve sequential information.
    """
    
    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) panel embeddings
            
        Returns:
            (B, N, D) panel embeddings with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# PANEL SEQUENCE TRANSFORMER
# ============================================================================

class PanelSequenceTransformer(nn.Module):
    """
    Transformer encoder for modeling panel sequences.
    
    Similar to BERT but adapted for comic panel sequences:
    - Processes sequences of panel embeddings from Stage 3
    - Captures narrative flow and dependencies between panels
    - Outputs contextualized panel representations
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_panels: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.max_panels = max_panels
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_panels, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Strip-level aggregation (for semantic strip embedding)
        self.strip_aggregator = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for strip aggregation
        self.strip_query = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, 
                panel_embeddings: torch.Tensor,
                panel_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            panel_embeddings: (B, N, D) panel embeddings from Stage 3
            panel_mask: (B, N) binary mask for valid panels (1 = valid, 0 = padding)
            
        Returns:
            contextualized_panels: (B, N, D) contextualized panel embeddings
            strip_embedding: (B, D) semantic strip embedding
        """
        B, N, D = panel_embeddings.shape
        
        # Add positional encoding
        x = self.pos_encoder(panel_embeddings)  # (B, N, D)
        
        # Create attention mask for transformer
        # Shape: (B, N) where True = masked (padding), False = not masked (valid)
        if panel_mask is not None:
            # Convert binary mask (1=valid) to transformer mask (True=padding)
            src_key_padding_mask = ~panel_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Apply transformer encoder
        contextualized_panels = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, N, D)
        
        # Aggregate to strip-level embedding using attention
        strip_query = self.strip_query.expand(B, -1, -1)  # (B, 1, D)
        
        strip_embedding, _ = self.strip_aggregator(
            query=strip_query,
            key=contextualized_panels,
            value=contextualized_panels,
            key_padding_mask=src_key_padding_mask
        )  # (B, 1, D)
        
        strip_embedding = strip_embedding.squeeze(1)  # (B, D)
        
        return contextualized_panels, strip_embedding


# ============================================================================
# COMICSPAP-INSPIRED TASK HEADS
# ============================================================================

class PanelPickingHead(nn.Module):
    """
    Head for panel picking task (ComicsPAP sequence filling).
    
    Given a sequence with one masked panel, select the correct panel
    from a set of candidates.
    """
    
    def __init__(self, d_model: int = 512, num_candidates: int = 5):
        super().__init__()
        
        self.num_candidates = num_candidates
        
        # Score computation for panel-candidate compatibility
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, 
                context_embedding: torch.Tensor,
                candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_embedding: (B, D) contextual embedding around masked position
            candidate_embeddings: (B, K, D) candidate panel embeddings
            
        Returns:
            scores: (B, K) compatibility scores for each candidate
        """
        B, K, D = candidate_embeddings.shape
        
        # Expand context to match candidates
        context_expanded = context_embedding.unsqueeze(1).expand(B, K, D)  # (B, K, D)
        
        # Concatenate context and candidates
        combined = torch.cat([context_expanded, candidate_embeddings], dim=-1)  # (B, K, 2D)
        
        # Score each candidate
        scores = self.scorer(combined).squeeze(-1)  # (B, K)
        
        return scores


class CharacterCoherenceHead(nn.Module):
    """
    Head for character coherence task.
    
    Ensures selected panels maintain consistent character appearances.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # Character consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, 
                panel_sequence: torch.Tensor,
                candidate_panel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_sequence: (B, N, D) existing panel sequence
            candidate_panel: (B, D) candidate panel to evaluate
            
        Returns:
            coherence_scores: (B,) character coherence score
        """
        B, N, D = panel_sequence.shape
        
        # Average existing panels as context
        context = panel_sequence.mean(dim=1)  # (B, D)
        
        # Compute coherence
        combined = torch.cat([context, candidate_panel], dim=-1)  # (B, 2D)
        coherence_score = self.consistency_scorer(combined).squeeze(-1)  # (B,)
        
        return coherence_score


class ClosureHead(nn.Module):
    """
    Head for visual and text closure tasks.
    
    Predicts the correct continuation/conclusion of visual actions or text.
    """
    
    def __init__(self, d_model: int = 512, closure_type: str = 'visual'):
        super().__init__()
        
        self.closure_type = closure_type
        
        # Closure prediction network
        self.closure_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, 
                preceding_panels: torch.Tensor,
                candidate_panel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preceding_panels: (B, N, D) panels before the gap
            candidate_panel: (B, D) candidate panel for closure
            
        Returns:
            closure_scores: (B,) closure plausibility scores
        """
        # Aggregate preceding context
        context = preceding_panels[:, -1, :]  # Use last panel (B, D)
        
        # Concatenate and score
        combined = torch.cat([context, candidate_panel], dim=-1)  # (B, 2D)
        closure_score = self.closure_net(combined).squeeze(-1)  # (B,)
        
        return closure_score


class CaptionRelevanceHead(nn.Module):
    """
    Head for caption relevance task.
    
    Aligns panels with appropriate captions (textual narrative integration).
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # Relevance scorer (bilinear for efficiency)
        self.relevance_scorer = nn.Bilinear(d_model, d_model, 1)
    
    def forward(self, 
                panel_embedding: torch.Tensor,
                caption_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_embedding: (B, D) panel visual+text embedding
            caption_embedding: (B, D) caption text embedding
            
        Returns:
            relevance_scores: (B,) relevance scores
        """
        relevance_score = self.relevance_scorer(
            panel_embedding,
            caption_embedding
        ).squeeze(-1)  # (B,)
        
        return relevance_score


# ============================================================================
# TEXT-CLOZE HEAD
# ============================================================================

class TextClozeHead(nn.Module):
    """
    Head for text-cloze task.
    
    Predicts missing text within a panel based on visual and textual context.
    Uses a discriminative approach: select correct text from candidates.
    """
    
    def __init__(self, d_model: int = 512, num_candidates: int = 4):
        super().__init__()
        
        self.num_candidates = num_candidates
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Text compatibility scorer
        self.text_scorer = nn.Bilinear(d_model, d_model, 1)
    
    def forward(self, 
                visual_context: torch.Tensor,
                surrounding_text: torch.Tensor,
                candidate_texts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_context: (B, D) visual context of the panel
            surrounding_text: (B, D) text from surrounding panels
            candidate_texts: (B, K, D) candidate text embeddings
            
        Returns:
            scores: (B, K) compatibility scores for each candidate
        """
        B, K, D = candidate_texts.shape
        
        # Encode context
        context = torch.cat([visual_context, surrounding_text], dim=-1)  # (B, 2D)
        context_encoded = self.context_encoder(context)  # (B, D)
        
        # Score each candidate text
        scores = []
        for k in range(K):
            score = self.text_scorer(
                context_encoded,
                candidate_texts[:, k, :]
            )  # (B, 1)
            scores.append(score)
        
        scores = torch.cat(scores, dim=-1)  # (B, K)
        
        return scores


# ============================================================================
# READING ORDER HEAD
# ============================================================================

class ReadingOrderHead(nn.Module):
    """
    Head for reading order prediction/refinement.
    
    Predicts the correct sequential order of panels.
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # Pairwise ordering scorer
        self.order_scorer = nn.Bilinear(d_model, d_model, 1)
    
    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_embeddings: (B, N, D) panel embeddings
            
        Returns:
            order_matrix: (B, N, N) pairwise ordering scores
                         order_matrix[b, i, j] > 0 means panel i comes before j
        """
        B, N, D = panel_embeddings.shape
        
        # Compute all pairwise orderings
        order_matrix = torch.zeros(B, N, N, device=panel_embeddings.device)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    score = self.order_scorer(
                        panel_embeddings[:, i, :],
                        panel_embeddings[:, j, :]
                    )  # (B, 1)
                    order_matrix[:, i, j] = score.squeeze(-1)
        
        return order_matrix


# ============================================================================
# MAIN STAGE 4 MODEL
# ============================================================================

class Stage4SequenceModel(nn.Module):
    """
    Complete Stage 4 semantic sequence modeling system.
    
    Combines:
    1. Panel sequence transformer for contextualization
    2. Multiple task heads for ComicsPAP and Text-Cloze
    3. Reading order refinement
    
    Inputs: Panel embeddings from Stage 3
    Outputs: Contextualized panel embeddings + semantic strip embeddings
    """
    
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_panels: int = 32,
                 enable_all_tasks: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.enable_all_tasks = enable_all_tasks
        
        # Core sequence transformer
        self.sequence_transformer = PanelSequenceTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_panels=max_panels
        )
        
        # Task heads
        self.panel_picking_head = PanelPickingHead(d_model)
        self.character_coherence_head = CharacterCoherenceHead(d_model)
        self.visual_closure_head = ClosureHead(d_model, closure_type='visual')
        self.text_closure_head = ClosureHead(d_model, closure_type='text')
        self.caption_relevance_head = CaptionRelevanceHead(d_model)
        self.text_cloze_head = TextClozeHead(d_model)
        self.reading_order_head = ReadingOrderHead(d_model)
    
    def forward(self, 
                panel_embeddings: torch.Tensor,
                panel_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            panel_embeddings: (B, N, D) panel embeddings from Stage 3
            panel_mask: (B, N) binary mask for valid panels
            
        Returns:
            Dictionary containing:
            - contextualized_panels: (B, N, D)
            - strip_embedding: (B, D)
        """
        # Process sequence through transformer
        contextualized_panels, strip_embedding = self.sequence_transformer(
            panel_embeddings,
            panel_mask
        )
        
        return {
            'contextualized_panels': contextualized_panels,
            'strip_embedding': strip_embedding,
            'panel_mask': panel_mask
        }
    
    def compute_task_predictions(self,
                                 contextualized_panels: torch.Tensor,
                                 task_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute predictions for various tasks.
        
        Args:
            contextualized_panels: (B, N, D) from forward pass
            task_data: Dictionary containing task-specific inputs
            
        Returns:
            Dictionary of task predictions
        """
        predictions = {}
        
        # Panel picking task
        if 'panel_picking_context' in task_data:
            predictions['panel_picking'] = self.panel_picking_head(
                task_data['panel_picking_context'],
                task_data['panel_candidates']
            )
        
        # Character coherence task
        if 'character_sequence' in task_data:
            predictions['character_coherence'] = self.character_coherence_head(
                task_data['character_sequence'],
                task_data['character_candidate']
            )
        
        # Visual closure task
        if 'visual_preceding' in task_data:
            predictions['visual_closure'] = self.visual_closure_head(
                task_data['visual_preceding'],
                task_data['visual_candidate']
            )
        
        # Text closure task
        if 'text_preceding' in task_data:
            predictions['text_closure'] = self.text_closure_head(
                task_data['text_preceding'],
                task_data['text_candidate']
            )
        
        # Caption relevance task
        if 'panel_for_caption' in task_data:
            predictions['caption_relevance'] = self.caption_relevance_head(
                task_data['panel_for_caption'],
                task_data['caption_embedding']
            )
        
        # Text-cloze task
        if 'cloze_visual' in task_data:
            predictions['text_cloze'] = self.text_cloze_head(
                task_data['cloze_visual'],
                task_data['cloze_surrounding'],
                task_data['cloze_candidates']
            )
        
        # Reading order task
        if 'reading_order_panels' in task_data:
            predictions['reading_order'] = self.reading_order_head(
                task_data['reading_order_panels']
            )
        
        return predictions
    
    @torch.no_grad()
    def encode_strip(self, 
                     panel_embeddings: torch.Tensor,
                     panel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a comic strip/page into a semantic embedding.
        
        Args:
            panel_embeddings: (B, N, D) panel embeddings from Stage 3
            panel_mask: (B, N) binary mask
            
        Returns:
            strip_embedding: (B, D) semantic strip embedding
        """
        outputs = self.forward(panel_embeddings, panel_mask)
        return outputs['strip_embedding']


if __name__ == "__main__":
    print("Stage 4 Semantic Sequence Modeling Framework")
    print("=" * 60)
    print("\nBased on:")
    print("- ComicsPAP: Panel picking, character coherence, closure tasks")
    print("- Text-Cloze: Contextual text prediction")
    print("- Transformer encoder for sequential understanding")
    print("\nFeatures:")
    print("1. Panel sequence transformer with positional encoding")
    print("2. Strip-level aggregation for semantic embeddings")
    print("3. Multiple task heads for ComicsPAP tasks")
    print("4. Text-cloze head for text prediction")
    print("5. Reading order head for sequence refinement")
    print("\nReady for integration with Stage 3 and Stage 5")
