import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BookBERTMultimodal(nn.Module):
    """BookBERT multimodal model (version 1)."""
    
    def __init__(self, textual_feature_dim, visual_feature_dim, num_classes, 
                 hidden_dim=256, num_attention_heads=4, bert_input_dim=768,
                 projection_dim=1024, num_hidden_layers=4, dropout_p=0.4, 
                 positional_embeddings='absolute'):
        super().__init__()
        
        self.textual_feature_dim = textual_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Project both features to a common dimension
        total_input_dim = textual_feature_dim + visual_feature_dim
        self.input_projection = nn.Linear(total_input_dim, bert_input_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=bert_input_dim,
            nhead=num_attention_heads,
            dim_feedforward=projection_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # Classification head
        self.classifier_head = nn.Linear(bert_input_dim, num_classes)
        
    def forward(self, textual_features, visual_features, attention_mask=None):
        """Forward pass with batched sequences."""
        # Concatenate features
        fused_features = torch.cat([textual_features, visual_features], dim=-1)
        
        # Project to BERT input dimension
        x = self.input_projection(fused_features)
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # Classification
        logits = self.classifier_head(x)
        
        return logits
    
    def forward_sequence(self, fused_features):
        """Forward pass for a single sequence (L, D_total) -> (L, num_classes)."""
        x = self.input_projection(fused_features)
        x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        logits = self.classifier_head(x)
        return logits


class BookBERTMultimodal2(nn.Module):
    """BookBERT multimodal model (version 2) with enhanced capabilities."""
    
    def __init__(self, textual_feature_dim, visual_feature_dim, num_classes, 
                 hidden_dim=256, num_attention_heads=4, bert_input_dim=768,
                 projection_dim=1024, num_hidden_layers=4, dropout_p=0.4, 
                 positional_embeddings='absolute'):
        super().__init__()
        
        self.textual_feature_dim = textual_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.positional_embeddings = positional_embeddings
        
        # Project both features to a common dimension
        total_input_dim = textual_feature_dim + visual_feature_dim
        self.input_projection = nn.Linear(total_input_dim, bert_input_dim)
        
        # Positional embeddings
        if positional_embeddings == 'absolute':
            self.pos_embedding = nn.Embedding(512, bert_input_dim)
        elif positional_embeddings == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, 512, bert_input_dim))
        else:
            self.pos_embedding = None
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=bert_input_dim,
            nhead=num_attention_heads,
            dim_feedforward=projection_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # Classification head
        self.classifier_head = nn.Linear(bert_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, textual_features, visual_features, attention_mask=None):
        """Forward pass with batched sequences."""
        # Concatenate features
        fused_features = torch.cat([textual_features, visual_features], dim=-1)
        
        # Project to BERT input dimension
        x = self.input_projection(fused_features)
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            batch_size, seq_len, _ = x.shape
            if self.positional_embeddings == 'absolute':
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                pos_emb = self.pos_embedding(positions)
            else:  # learned
                pos_emb = self.pos_embedding[:, :seq_len, :]
            x = x + pos_emb
        
        # Pass through transformer
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # Apply dropout and classification
        x = self.dropout(x)
        logits = self.classifier_head(x)
        
        return logits
    
    def forward_sequence(self, fused_features):
        """Forward pass for a single sequence (L, D_total) -> (L, num_classes)."""
        # Add batch dimension
        x = self.input_projection(fused_features)
        x = x.unsqueeze(0)  # (1, L, D)
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            seq_len = x.size(1)
            if self.positional_embeddings == 'absolute':
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                pos_emb = self.pos_embedding(positions)
            else:  # learned
                pos_emb = self.pos_embedding[:, :seq_len, :]
            x = x + pos_emb
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Remove batch dimension and apply classification
        x = x.squeeze(0)  # (L, D)
        x = self.dropout(x)
        logits = self.classifier_head(x)
        
        return logits
