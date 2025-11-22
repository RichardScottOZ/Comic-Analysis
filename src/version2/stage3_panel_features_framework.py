"""
Stage 3: Domain-Adapted Multimodal Panel Feature Generation

This module implements the Stage 3 pipeline from Version 2.0 framework:
- Domain-adapted visual backbone(s) fine-tuned on comic book images
- Improved multi-modal fusion that addresses v1 GatedFusion issues
- Rich panel-level feature generation for downstream sequence modeling

Key Improvements over v1 (ClosureLiteSimple):
1. Modality-Independent Encoders: Each encoder can operate independently
2. Flexible Fusion: Support for late fusion, avoiding dummy input dominance
3. Multiple Visual Backbones: SigLIP + ResNet fusion for richer features
4. Domain Adaptation: Fine-tuning layers specifically for comic aesthetics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, SiglipVisionModel
import numpy as np
from typing import Dict, Optional, Tuple


# ============================================================================
# DOMAIN-ADAPTED VISUAL ENCODERS
# ============================================================================

class DomainAdaptedSigLIP(nn.Module):
    """
    SigLIP vision encoder with domain adaptation layers for comic images.
    
    Addresses v1 issues:
    - Fine-tunable for comic-specific visual features
    - Can be used independently (no fusion dominance)
    - Generates discriminative embeddings even without text/comp features
    """
    
    def __init__(self, model_name="google/siglip-base-patch16-224", 
                 out_dim=512, freeze_backbone=True, adaptation_layers=2):
        super().__init__()
        
        # Load SigLIP vision model
        self.siglip = SiglipVisionModel.from_pretrained(model_name)
        self.hidden_size = self.siglip.config.hidden_size
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.siglip.parameters():
                param.requires_grad = False
        
        # Domain adaptation layers for comic-specific features
        adaptation_modules = []
        current_dim = self.hidden_size
        
        for i in range(adaptation_layers):
            next_dim = out_dim if i == adaptation_layers - 1 else current_dim // 2
            adaptation_modules.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # Remove last activation and dropout
        adaptation_modules = adaptation_modules[:-2]
        self.adaptation = nn.Sequential(*adaptation_modules)
        
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) tensor of panel images
            
        Returns:
            (B, out_dim) panel visual features
        """
        # Extract features from SigLIP
        outputs = self.siglip(images)
        # Use pooled output (global image representation)
        pooled = outputs.pooler_output  # (B, hidden_size)
        
        # Apply domain adaptation
        features = self.adaptation(pooled)  # (B, out_dim)
        
        return features


class DomainAdaptedResNet(nn.Module):
    """
    ResNet-based encoder with domain adaptation for comic panel analysis.
    
    Provides complementary features to SigLIP:
    - Better at low-level visual details
    - Captures texture and fine-grained patterns
    - Trained on different visual priors
    """
    
    def __init__(self, out_dim=512, freeze_backbone=True, adaptation_layers=2):
        super().__init__()
        
        # Use timm for efficient ResNet loading
        import timm
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.feature_dim = self.resnet.num_features  # 2048 for ResNet50
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Domain adaptation layers
        adaptation_modules = []
        current_dim = self.feature_dim
        
        for i in range(adaptation_layers):
            next_dim = out_dim if i == adaptation_layers - 1 else current_dim // 2
            adaptation_modules.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
        
        # Remove last activation and dropout
        adaptation_modules = adaptation_modules[:-2]
        self.adaptation = nn.Sequential(*adaptation_modules)
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) tensor of panel images
            
        Returns:
            (B, out_dim) panel visual features
        """
        # Extract features from ResNet
        features = self.resnet(images)  # (B, feature_dim)
        
        # Apply domain adaptation
        adapted = self.adaptation(features)  # (B, out_dim)
        
        return adapted


class MultiBackboneVisualEncoder(nn.Module):
    """
    Combines multiple visual backbones for richer panel representations.
    
    Fusion strategies:
    - concat: Concatenate features from both backbones
    - attention: Learned attention-weighted combination
    - gate: Learned gating mechanism (similar to v1 but improved)
    """
    
    def __init__(self, backbone_dim=512, fusion_type='attention', use_resnet=True):
        super().__init__()
        
        self.use_resnet = use_resnet
        self.fusion_type = fusion_type
        
        # Initialize backbones
        self.siglip = DomainAdaptedSigLIP(out_dim=backbone_dim)
        
        if use_resnet:
            self.resnet = DomainAdaptedResNet(out_dim=backbone_dim)
            
            # Fusion mechanism
            if fusion_type == 'concat':
                self.out_dim = backbone_dim * 2
                self.fusion = nn.Identity()
            elif fusion_type == 'attention':
                self.out_dim = backbone_dim
                self.attention = nn.Sequential(
                    nn.Linear(backbone_dim * 2, 2),
                    nn.Softmax(dim=-1)
                )
            elif fusion_type == 'gate':
                self.out_dim = backbone_dim
                # Improved gating: uses both features to compute gate
                self.gate = nn.Sequential(
                    nn.Linear(backbone_dim * 2, backbone_dim),
                    nn.GELU(),
                    nn.Linear(backbone_dim, 2),
                    nn.Softmax(dim=-1)
                )
        else:
            self.out_dim = backbone_dim
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) tensor of panel images
            
        Returns:
            (B, out_dim) fused visual features
        """
        # Get SigLIP features
        f_siglip = self.siglip(images)  # (B, backbone_dim)
        
        if not self.use_resnet:
            return f_siglip
        
        # Get ResNet features
        f_resnet = self.resnet(images)  # (B, backbone_dim)
        
        # Fuse features
        if self.fusion_type == 'concat':
            return torch.cat([f_siglip, f_resnet], dim=-1)
        
        elif self.fusion_type == 'attention':
            # Compute attention weights
            combined = torch.cat([f_siglip, f_resnet], dim=-1)
            weights = self.attention(combined)  # (B, 2)
            
            # Weighted combination
            fused = weights[:, 0:1] * f_siglip + weights[:, 1:2] * f_resnet
            return fused
        
        elif self.fusion_type == 'gate':
            # Compute gating weights
            combined = torch.cat([f_siglip, f_resnet], dim=-1)
            weights = self.gate(combined)  # (B, 2)
            
            # Gated combination
            fused = weights[:, 0:1] * f_siglip + weights[:, 1:2] * f_resnet
            return fused


# ============================================================================
# TEXT ENCODER (Improved from v1)
# ============================================================================

class TextEncoder(nn.Module):
    """
    Text encoder with optional output for independent operation.
    
    Improvements over v1:
    - Can handle empty text gracefully
    - Returns valid embeddings even with minimal input
    - Supports masking for missing text
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 out_dim=512, freeze=True):
        super().__init__()
        
        # Load sentence transformer model
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection layer
        self.proj = nn.Linear(self.hidden_size, out_dim)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, seq_len) token ids
            attention_mask: (B, seq_len) attention mask
            
        Returns:
            (B, out_dim) text features
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Project to output dimension
        features = self.proj(pooled)
        
        return features


# ============================================================================
# COMPOSITIONAL ENCODER (Enhanced from v1)
# ============================================================================

class CompositionEncoder(nn.Module):
    """
    Encoder for panel compositional features.
    
    Improvements over v1:
    - Deeper network for better feature learning
    - Batch normalization for stability
    - Optional dropout for regularization
    """
    
    def __init__(self, in_dim=7, out_dim=512, hidden_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, comp_feats):
        """
        Args:
            comp_feats: (B, in_dim) compositional features
            
        Returns:
            (B, out_dim) composition features
        """
        return self.mlp(comp_feats)


# ============================================================================
# IMPROVED MULTI-MODAL FUSION
# ============================================================================

class AdaptiveFusion(nn.Module):
    """
    Improved fusion mechanism that addresses v1 GatedFusion issues.
    
    Key improvements:
    1. Modality presence indicators: Knows which modalities are actually present
    2. Dynamic gating: Gate weights depend on modality quality, not just features
    3. Residual connections: Prevents complete override of any modality
    4. Normalization: Each modality normalized before fusion
    """
    
    def __init__(self, dim=512, use_modality_indicators=True):
        super().__init__()
        
        self.dim = dim
        self.use_modality_indicators = use_modality_indicators
        
        # Modality-specific normalization
        self.norm_vision = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_comp = nn.LayerNorm(dim)
        
        # Gate computation
        if use_modality_indicators:
            # Gate takes features + presence indicators
            self.gate = nn.Sequential(
                nn.Linear(3 * dim + 3, dim),  # +3 for presence flags
                nn.GELU(),
                nn.Linear(dim, 3)
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(3 * dim, dim),
                nn.GELU(),
                nn.Linear(dim, 3)
            )
        
        # Residual projection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, vision_feats, text_feats, comp_feats, 
                modality_mask=None):
        """
        Args:
            vision_feats: (B, dim) visual features
            text_feats: (B, dim) text features
            comp_feats: (B, dim) compositional features
            modality_mask: (B, 3) binary mask indicating presence of [vision, text, comp]
                          If None, assumes all modalities present
            
        Returns:
            (B, dim) fused features
        """
        B = vision_feats.shape[0]
        
        # Default mask: all modalities present
        if modality_mask is None:
            modality_mask = torch.ones(B, 3, device=vision_feats.device)
        
        # Normalize each modality
        v_norm = self.norm_vision(vision_feats)
        t_norm = self.norm_text(text_feats)
        c_norm = self.norm_comp(comp_feats)
        
        # Compute gate weights
        if self.use_modality_indicators:
            gate_input = torch.cat([v_norm, t_norm, c_norm, modality_mask], dim=-1)
        else:
            gate_input = torch.cat([v_norm, t_norm, c_norm], dim=-1)
        
        # Raw gate logits
        gate_logits = self.gate(gate_input)  # (B, 3)
        
        # Mask out unavailable modalities (set to very negative value)
        gate_logits = gate_logits + (1 - modality_mask) * (-1e9)
        
        # Compute gate weights
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, 3)
        
        # Fused representation (weighted sum)
        fused = (gate_weights[:, 0:1] * v_norm + 
                gate_weights[:, 1:2] * t_norm + 
                gate_weights[:, 2:3] * c_norm)
        
        # Add residual connection to preserve some of each modality
        # This prevents complete override
        residual = (v_norm + t_norm + c_norm) / 3.0
        fused = fused + self.residual_weight * residual
        
        return fused


# ============================================================================
# MAIN PANEL FEATURE EXTRACTOR
# ============================================================================

class PanelFeatureExtractor(nn.Module):
    """
    Complete Stage 3 panel feature extraction model.
    
    Inputs (for narrative pages from Stage 2):
    - Panel image crops
    - Panel text (from OCR/VLM)
    - Panel compositional features (bounding boxes, positions, etc.)
    
    Outputs:
    - Rich multimodal panel embeddings for Stage 4
    
    Key design principles:
    1. Each encoder can operate independently
    2. Fusion is adaptive and respects modality availability
    3. Outputs are suitable for sequence modeling in Stage 4
    """
    
    def __init__(self, 
                 visual_backbone='siglip',  # 'siglip', 'resnet', or 'both'
                 visual_fusion='attention',  # for multi-backbone
                 feature_dim=512,
                 freeze_backbones=True,
                 use_modality_indicators=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Visual encoder
        if visual_backbone == 'siglip':
            self.vision = DomainAdaptedSigLIP(out_dim=feature_dim, 
                                             freeze_backbone=freeze_backbones)
        elif visual_backbone == 'resnet':
            self.vision = DomainAdaptedResNet(out_dim=feature_dim,
                                             freeze_backbone=freeze_backbones)
        elif visual_backbone == 'both':
            self.vision = MultiBackboneVisualEncoder(backbone_dim=feature_dim,
                                                     fusion_type=visual_fusion,
                                                     use_resnet=True)
        else:
            raise ValueError(f"Unknown visual_backbone: {visual_backbone}")
        
        # Text encoder
        self.text = TextEncoder(out_dim=feature_dim, freeze=freeze_backbones)
        
        # Composition encoder
        self.comp = CompositionEncoder(in_dim=7, out_dim=feature_dim)
        
        # Fusion
        self.fusion = AdaptiveFusion(dim=feature_dim, 
                                    use_modality_indicators=use_modality_indicators)
    
    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                - images: (B, 3, H, W) panel images
                - input_ids: (B, seq_len) text token ids
                - attention_mask: (B, seq_len) text attention mask
                - comp_feats: (B, 7) compositional features
                - modality_mask: (B, 3) optional modality presence indicators
                
        Returns:
            (B, feature_dim) multimodal panel embeddings
        """
        # Extract features from each modality
        v_feats = self.vision(batch['images'])
        t_feats = self.text(batch['input_ids'], batch['attention_mask'])
        c_feats = self.comp(batch['comp_feats'])
        
        # Fuse modalities
        modality_mask = batch.get('modality_mask', None)
        panel_embedding = self.fusion(v_feats, t_feats, c_feats, modality_mask)
        
        return panel_embedding
    
    @torch.no_grad()
    def encode_image_only(self, images):
        """
        Encode panels using only visual features.
        Useful for queries or when other modalities unavailable.
        
        Args:
            images: (B, 3, H, W) panel images
            
        Returns:
            (B, feature_dim) visual panel embeddings
        """
        return self.vision(images)
    
    @torch.no_grad()
    def encode_text_only(self, input_ids, attention_mask):
        """
        Encode panels using only text features.
        
        Args:
            input_ids: (B, seq_len) text token ids
            attention_mask: (B, seq_len) text attention mask
            
        Returns:
            (B, feature_dim) text panel embeddings
        """
        return self.text(input_ids, attention_mask)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_modality_mask(has_image=True, has_text=True, has_comp=True, 
                        device='cpu', batch_size=1):
    """
    Create modality presence mask.
    
    Args:
        has_image: Whether image modality is present
        has_text: Whether text modality is present
        has_comp: Whether compositional modality is present
        device: Device for tensor
        batch_size: Batch size
        
    Returns:
        (batch_size, 3) binary mask tensor
    """
    mask = torch.tensor(
        [[float(has_image), float(has_text), float(has_comp)]], 
        device=device
    ).repeat(batch_size, 1)
    return mask


def comp_features_for_panel(panel_bbox, page_width, page_height):
    """
    Extract compositional features for a panel.
    
    Args:
        panel_bbox: [x, y, w, h] bounding box
        page_width: Page width in pixels
        page_height: Page height in pixels
        
    Returns:
        numpy array of 7 compositional features
    """
    x, y, w, h = panel_bbox
    
    return np.array([
        w / h,              # aspect ratio
        (w * h) / (page_width * page_height),  # size ratio
        0,                  # character count (placeholder)
        0,                  # shot mean (placeholder)
        0,                  # shot max (placeholder)
        (x + w/2) / page_width,   # center x (normalized)
        (y + h/2) / page_height   # center y (normalized)
    ], dtype=np.float32)


if __name__ == "__main__":
    print("Stage 3 Panel Feature Extraction Framework")
    print("=" * 60)
    print("\nKey improvements over v1:")
    print("1. Independent modality encoders (no fusion dominance)")
    print("2. Domain-adapted visual backbones for comics")
    print("3. Adaptive fusion with modality awareness")
    print("4. Support for single-modality queries")
    print("5. Multiple visual backbone fusion options")
    print("\nReady for integration with Stage 2 (PSS) and Stage 4 (Sequence Modeling)")
