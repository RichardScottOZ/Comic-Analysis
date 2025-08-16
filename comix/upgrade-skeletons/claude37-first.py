# Comic Book Narrative Embedding Framework Implementation

I'll create a practical implementation framework for the multi-modal, sequential, and compositional comic book embeddings described in the report. This implementation focuses on the core architecture components and how they fit together.

## Stage 1: Atomic Panel Representation Module

```python
class AtomicPanelRepresentation:
    def __init__(self, vision_model, text_model, composition_encoder):
        self.vision_model = vision_model  # Pre-trained VLM vision component
        self.text_model = text_model      # Pre-trained VLM text component
        self.composition_encoder = composition_encoder  # MLP for compositional features
        
    def extract_visual_vector(self, panel_image):
        # Process panel image through vision model
        return self.vision_model(panel_image)
    
    def extract_textual_vector(self, panel_text):
        # Combine all text (dialogue, narration, sound effects)
        return self.text_model(panel_text)
    
    def extract_compositional_vector(self, panel_metadata):
        # Extract compositional features like:
        # - Panel aspect ratio
        # - Character count and positions
        # - Shot scale estimation (character area / panel area)
        features = self._process_compositional_features(panel_metadata)
        return self.composition_encoder(features)
    
    def process_panel(self, panel_image, panel_text, panel_metadata):
        v_vis = self.extract_visual_vector(panel_image)
        v_txt = self.extract_textual_vector(panel_text)
        v_comp = self.extract_compositional_vector(panel_metadata)
        
        return v_vis, v_txt, v_comp
```

## Stage 2: Intra-Panel Fusion Module

```python
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, modal_dim, fusion_dim, num_heads=4):
        super().__init__()
        self.visual_proj = nn.Linear(modal_dim, fusion_dim)
        self.text_proj = nn.Linear(modal_dim, fusion_dim)
        self.comp_proj = nn.Linear(modal_dim, fusion_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, v_vis, v_txt, v_comp):
        # Project each modality to common dimension
        vis_proj = self.visual_proj(v_vis)
        txt_proj = self.text_proj(v_txt)
        comp_proj = self.comp_proj(v_comp)
        
        # Use visual features as query, text+comp as key/value
        context = torch.cat([txt_proj, comp_proj], dim=1)
        attn_out, _ = self.cross_attention(
            query=vis_proj.unsqueeze(1),
            key=context,
            value=context
        )
        
        # Concatenate and fuse all modalities
        fused = torch.cat([vis_proj, txt_proj, comp_proj], dim=-1)
        panel_embedding = self.fusion_mlp(fused)
        
        return panel_embedding
```

## Stage 3: Inter-Panel Sequential Encoding

```python
class SequentialPanelEncoder(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 max_sequence_length=32):
        super().__init__()
        
        # Position encoding for sequential information
        self.pos_encoding = nn.Parameter(
            torch.zeros(max_sequence_length, embedding_dim)
        )
        
        # Transformer encoder for panel sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, panel_embeddings):
        # Add positional encoding to capture sequence information
        seq_length = panel_embeddings.shape[1]
        panel_embeddings = panel_embeddings + self.pos_encoding[:seq_length]
        
        # Process through transformer to get context-aware panel embeddings
        sequential_embeddings = self.transformer(panel_embeddings)
        
        return sequential_embeddings
```

## Hierarchical Aggregation Module

```python
class HierarchicalAttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        # Page-level attention aggregation
        self.page_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Comic-level attention aggregation
        self.comic_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Trainable query vectors
        self.page_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.comic_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
    def panels_to_page(self, panel_embeddings, panel_mask=None):
        # Use learned query to attend over all panels
        batch_size = panel_embeddings.shape[0]
        query = self.page_query.expand(batch_size, -1, -1)
        
        # Attend over panels to create page embedding
        page_embedding, attention_weights = self.page_attention(
            query=query,
            key=panel_embeddings,
            value=panel_embeddings,
            key_padding_mask=panel_mask
        )
        
        return page_embedding.squeeze(1), attention_weights
    
    def pages_to_comic(self, page_embeddings, page_mask=None):
        # Use learned query to attend over all pages
        batch_size = page_embeddings.shape[0]
        query = self.comic_query.expand(batch_size, -1, -1)
        
        # Attend over pages to create comic embedding
        comic_embedding, attention_weights = self.comic_attention(
            query=query,
            key=page_embeddings,
            value=page_embeddings,
            key_padding_mask=page_mask
        )
        
        return comic_embedding.squeeze(1), attention_weights
```

## Complete Framework Integration

```python
class ComicBookEmbeddingFramework:
    def __init__(self, config):
        # Initialize all components
        self.panel_representation = AtomicPanelRepresentation(
            vision_model=load_vision_model(config),
            text_model=load_text_model(config),
            composition_encoder=build_composition_encoder(config)
        )
        
        self.modal_fusion = CrossModalAttentionFusion(
            modal_dim=config.modal_dim,
            fusion_dim=config.fusion_dim
        )
        
        self.sequential_encoder = SequentialPanelEncoder(
            embedding_dim=config.fusion_dim,
            num_layers=config.num_transformer_layers
        )
        
        self.hierarchical_aggregator = HierarchicalAttentionAggregator(
            embedding_dim=config.fusion_dim
        )
        
    def process_comic(self, comic_data):
        # Process each panel in each page
        all_page_embeddings = []
        
        for page in comic_data:
            panel_vectors = []
            
            for panel in page['panels']:
                # Stage 1: Extract atomic representations
                v_vis, v_txt, v_comp = self.panel_representation.process_panel(
                    panel['image'], 
                    panel['text'], 
                    panel['metadata']
                )
                
                # Stage 2: Fuse modalities
                panel_embedding = self.modal_fusion(v_vis, v_txt, v_comp)
                panel_vectors.append(panel_embedding)
            
            # Convert list to tensor
            panel_embeddings = torch.stack(panel_vectors)
            
            # Stage 3: Encode sequential relationships
            sequential_panel_embeddings = self.sequential_encoder(panel_embeddings)
            
            # Create page embedding
            page_embedding, panel_attention = self.hierarchical_aggregator.panels_to_page(
                sequential_panel_embeddings
            )
            all_page_embeddings.append(page_embedding)
        
        # Create final comic embedding
        page_embeddings = torch.stack(all_page_embeddings)
        comic_embedding, page_attention = self.hierarchical_aggregator.pages_to_comic(
            page_embeddings
        )
        
        return {
            'panel_embeddings': sequential_panel_embeddings,
            'page_embedding': page_embeddings,
            'comic_embedding': comic_embedding,
            'panel_attention': panel_attention,
            'page_attention': page_attention
        }
```

## Training Objectives Implementation

```python
class ComicEmbeddingTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def masked_panel_modeling_loss(self, panel_embeddings, masked_indices):
        # Implementation of MPM objective:
        # Given sequence with masked panels, predict original panel embeddings
        pass
        
    def panel_order_prediction_loss(self, seq1, seq2, labels):
        # Implementation of POP objective:
        # Predict if second sequence correctly follows first
        pass
        
    def reading_path_prediction_loss(self, panel_embedding, neighbor_embeddings, correct_idx):
        # Implementation of RPP objective:
        # Predict which neighbor is the next panel in reading order
        pass
        
    def train_step(self, batch):
        # Combined training step with all objectives
        outputs = self.model.process_comic(batch)
        
        # Calculate losses from different objectives
        mpm_loss = self.masked_panel_modeling_loss(
            outputs['panel_embeddings'], 
            batch['masked_indices']
        )
        
        pop_loss = self.panel_order_prediction_loss(
            batch['seq1'], 
            batch['seq2'], 
            batch['order_labels']
        )
        
        rpp_loss = self.reading_path_prediction_loss(
            outputs['panel_embeddings'],
            batch['neighbor_embeddings'],
            batch['next_panel_idx']
        )
        
        # Combine losses with appropriate weights
        total_loss = (
            self.config.mpm_weight * mpm_loss +
            self.config.pop_weight * pop_loss +
            self.config.rpp_weight * rpp_loss
        )
        
        return total_loss
```

Would you like me to explain or break down any part of this implementation?