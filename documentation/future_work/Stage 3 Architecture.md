# Stage 3: Domain-Adapted Multimodal Panel Feature Generation

## Overview

Stage 3 is the third component of the Version 2.0 framework, responsible for generating rich multimodal panel embeddings from narrative pages identified by Stage 2 (PSS/CoSMo). These embeddings serve as input to Stage 4's sequence modeling.

## Position in Version 2.0 Pipeline

```
Stage 1: Raw Data Ingestion
    ↓
Stage 2: Page Stream Segmentation (CoSMo) → Identifies narrative pages
    ↓
Stage 3: Panel Feature Generation ← YOU ARE HERE
    ↓
Stage 4: Semantic Sequence Modeling
    ↓
Stage 5: Queryable Embeddings
```

## Key Improvements Over Version 1 (ClosureLiteSimple)

### Problems Identified in v1 (from Model_debugging.md)

1. **Fusion Dominance Issue**: GatedFusion was dominated by dummy inputs when doing single-modality queries
   - Vision features were "drowned out" by constant dummy text/comp features
   - Made image-only queries non-discriminative

2. **Compositional Dominance**: Vision-only embeddings showed layout/structure dominance over semantic content
   - Clustering based on panel count, aspect ratios, darkness
   - Lacked semantic understanding of actual content

3. **Multi-Modal Mismatch**: Single-modality queries against multi-modal embeddings yielded poor results
   - Pure vision queries vs. fused targets lived in different spaces
   - Model couldn't effectively handle missing modalities

### Stage 3 Solutions

#### 1. Modality-Independent Encoders

Each encoder (vision, text, composition) can operate **independently**:

```python
# Vision encoder can work alone
visual_features = model.encode_image_only(images)

# Text encoder can work alone  
text_features = model.encode_text_only(input_ids, attention_mask)

# No forced fusion with dummy inputs
```

**Benefits**:
- Single-modality queries produce discriminative embeddings
- No dummy input contamination
- Each modality has its own learned representation space

#### 2. Domain-Adapted Visual Backbones

Multiple visual encoders fine-tuned for comic aesthetics:

**SigLIP Encoder**:
- Pre-trained on image-text pairs
- Better semantic understanding
- Captures high-level content

**ResNet Encoder**:
- Stronger low-level features
- Better texture/pattern recognition
- Complementary to SigLIP

**Multi-Backbone Fusion**:
```python
visual_features = α * SigLIP(image) + β * ResNet(image)
```

Where α and β are learned attention weights, preventing any single backbone from dominating.

**Benefits**:
- Richer visual representations than v1's single ViT
- Better comic-specific feature learning
- Reduced compositional dominance through semantic features

#### 3. Adaptive Fusion Mechanism

New `AdaptiveFusion` module addresses v1's GatedFusion issues:

**Modality Presence Indicators**:
```python
modality_mask = [has_vision, has_text, has_comp]  # Binary flags
```

**Dynamic Gating**:
- Gate weights informed by which modalities are actually present
- Missing modalities get zero weight (not dummy values)
- Prevents override by constant features

**Residual Connections**:
```python
fused = weighted_sum(modalities) + residual_weight * mean(modalities)
```
- Ensures each modality contributes something
- Prevents complete override

**Benefits**:
- Graceful handling of missing modalities
- No dummy input contamination
- Better multi-modal alignment

## Architecture Details

### Visual Encoders

#### DomainAdaptedSigLIP
```
Input: (B, 3, 224, 224) panel images
    ↓
SigLIP Vision Model (frozen/unfrozen)
    ↓
Hidden Features (B, 768)
    ↓
Domain Adaptation Layers:
  - Linear(768 → 512)
  - LayerNorm
  - GELU
    ↓
Output: (B, 512) visual features
```

#### DomainAdaptedResNet
```
Input: (B, 3, 224, 224) panel images
    ↓
ResNet50 (frozen/unfrozen)
    ↓
Hidden Features (B, 2048)
    ↓
Domain Adaptation Layers:
  - Linear(2048 → 1024)
  - LayerNorm
  - GELU
  - Linear(1024 → 512)
  - LayerNorm
  - GELU
    ↓
Output: (B, 512) visual features
```

#### MultiBackboneVisualEncoder
```
Input: (B, 3, 224, 224) panel images
    ↓
    ├─→ SigLIP → f_siglip (B, 512)
    └─→ ResNet → f_resnet (B, 512)
    ↓
Fusion (attention/gate/concat)
    ↓
Output: (B, 512) or (B, 1024) fused visual features
```

### Text Encoder

```
Input: (B, seq_len) token IDs + attention mask
    ↓
SentenceTransformer Model (frozen)
    ↓
Mean Pooling (with attention mask)
    ↓
Projection Layer
    ↓
Output: (B, 512) text features
```

**Improvements over v1**:
- Better handling of empty text
- Mean pooling instead of just CLS token
- Returns valid embeddings even with minimal input

### Compositional Encoder

```
Input: (B, 7) compositional features
  [aspect_ratio, size_ratio, char_count, shot_mean, shot_max, center_x, center_y]
    ↓
MLP:
  - Linear(7 → 256)
  - BatchNorm + ReLU + Dropout
  - Linear(256 → 256)
  - BatchNorm + ReLU + Dropout
  - Linear(256 → 512)
    ↓
Output: (B, 512) composition features
```

**Improvements over v1**:
- Deeper network (3 layers vs 2)
- Batch normalization for stability
- Dropout for regularization

### AdaptiveFusion

```
Inputs:
  - vision_feats (B, 512)
  - text_feats (B, 512)
  - comp_feats (B, 512)
  - modality_mask (B, 3)
    ↓
Normalize each modality (LayerNorm)
    ↓
Concatenate: [v_norm, t_norm, c_norm, modality_mask]
    ↓
Gate Network:
  - Linear(512*3 + 3 → 512)
  - GELU
  - Linear(512 → 3)
    ↓
Mask unavailable modalities
    ↓
Softmax → gate_weights (B, 3)
    ↓
fused = Σ (gate_weight_i * modality_i)
    ↓
Add residual: fused + λ * mean(all_modalities)
    ↓
Output: (B, 512) fused panel embedding
```

## Training Objectives

### 1. Contrastive Learning

**Goal**: Panels from the same page should have similar embeddings.

```
For each page with panels [p1, p2, p3, ..., pn]:
  - Compute panel embeddings: [e1, e2, e3, ..., en]
  - Similarity matrix: S = normalize(E) @ normalize(E).T
  - Loss: Cross-entropy encouraging high similarity within page
```

**Why this works**:
- Panels on same page share visual style, characters, narrative context
- Encourages semantic coherence in embedding space
- Prepares embeddings for Stage 4's sequence modeling

### 2. Panel Reconstruction

**Goal**: Predict a masked panel's features from context of other panels.

```
For each page:
  - Randomly mask one panel: p_masked
  - Context: mean of other panel embeddings
  - Predict: reconstruction_head(context) → p_predicted
  - Loss: MSE(p_predicted, p_masked)
```

**Why this works**:
- Forces model to learn contextual relationships
- Builds understanding of narrative flow
- Tests if embeddings capture enough information

### 3. Modality Alignment

**Goal**: Vision and text features for same panel should be similar.

```
For each panel with both image and text:
  - v = vision_encoder(image)
  - t = text_encoder(text)
  - Contrastive loss: align v and t for same panel
```

**Why this works**:
- Ensures multi-modal features are in aligned space
- Allows flexible queries (image-only, text-only, or both)
- Prevents modality-specific clustering

### Combined Training Loss

```
L_total = L_contrastive + 0.5 * L_reconstruction + 0.3 * L_alignment
```

## Data Flow

### Input (from Stage 2)

```json
{
  "book_id/page_001.json": {
    "image_width": 1988,
    "image_height": 3057,
    "pss_label": "narrative",
    "panels": [
      {
        "bbox": [100, 200, 500, 400],
        "text": "Panel dialogue here",
        "confidence": 0.95
      },
      ...
    ]
  }
}
```

### Processing

1. **Load page data**: Read JSON + image
2. **Crop panels**: Extract panel images from bounding boxes
3. **Compute features**:
   - Vision: Transform + encode panel crops
   - Text: Tokenize + encode panel text
   - Composition: Calculate spatial features from bbox
4. **Generate embeddings**: Fuse modalities
5. **Create batch**: Stack panels from multiple pages

### Output

```python
{
  'panel_embeddings': (B, N, 512),  # Rich multimodal features
  'panel_mask': (B, N),              # Valid panel indicators
  'metadata': [...]                  # Book IDs, page names, etc.
}
```

These embeddings feed into Stage 4 for sequence modeling.

## Usage Examples

### Training

```bash
python train_stage3.py \
  --data_root /path/to/comics \
  --train_pss_labels /path/to/train_pss.json \
  --val_pss_labels /path/to/val_pss.json \
  --visual_backbone both \
  --visual_fusion attention \
  --feature_dim 512 \
  --batch_size 4 \
  --epochs 20 \
  --lr 1e-4 \
  --use_wandb \
  --run_name stage3_multibackbone
```

### Inference (Image-Only Query)

```python
from stage3_panel_features_framework import PanelFeatureExtractor
import torch
from PIL import Image

# Load model
model = PanelFeatureExtractor(visual_backbone='both', feature_dim=512)
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
model.eval()

# Query with image only
image = Image.open('panel.jpg')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    query_embedding = model.encode_image_only(image_tensor)

# Search similar panels
similarities = cosine_similarity(query_embedding, panel_database)
```

### Inference (Multi-Modal Query)

```python
# Query with image + text
batch = {
    'images': image_tensor,
    'input_ids': tokenizer(text, return_tensors='pt')['input_ids'],
    'attention_mask': tokenizer(text, return_tensors='pt')['attention_mask'],
    'comp_feats': torch.tensor([[1.0, 0.5, 0, 0, 0, 0.5, 0.5]]),
    'modality_mask': torch.tensor([[1.0, 1.0, 1.0]])  # All modalities present
}

with torch.no_grad():
    query_embedding = model(batch)
```

## Comparison: v1 vs Stage 3

| Aspect | Version 1 (ClosureLiteSimple) | Stage 3 |
|--------|------------------------------|---------|
| **Visual Encoder** | Single ViT | SigLIP + ResNet (optional) |
| **Fusion** | GatedFusion (3-way gate) | AdaptiveFusion (modality-aware) |
| **Missing Modalities** | Dummy inputs (zeros) | Modality mask + dynamic gating |
| **Single-Modality Queries** | Non-discriminative | Fully supported |
| **Visual Features** | Compositional dominance | Semantic + compositional |
| **Training** | MPM + POP + RPP | Contrastive + Reconstruction + Alignment |
| **Domain Adaptation** | None | Fine-tunable adaptation layers |

## Integration with Other Stages

### From Stage 2 (PSS/CoSMo)

**Input**: Page type classifications
```json
{
  "book_id": {
    "page_001": "narrative",
    "page_002": "advertisement",
    "page_003": "narrative"
  }
}
```

**Stage 3 filters**: Only processes `"narrative"` pages

### To Stage 4 (Sequence Modeling)

**Output**: Panel embeddings per page
```python
{
  'panel_embeddings': (num_pages, max_panels, 512),
  'panel_masks': (num_pages, max_panels),
  'page_metadata': [...]
}
```

**Stage 4 uses**: Transformer encoder over panel sequences to model narrative flow

## Future Enhancements

### 1. Character Detection Integration
- Add character bounding boxes as additional modality
- Character-specific visual features
- Character re-identification across panels

### 2. Style Features
- Artistic style embeddings (art style, coloring, line work)
- Publisher/era-specific adaptations
- Style transfer capabilities

### 3. Reading Order Refinement
- Integrate reading order predictions
- Sequential panel embeddings
- Better handling of non-standard layouts

### 4. Hierarchical Features
- Panel → Strip → Page → Issue hierarchy
- Multi-scale attention mechanisms
- Cross-page context

### 5. Zero-Shot Capabilities
- Open-vocabulary panel detection
- CLIP-style text-to-panel retrieval
- Natural language panel queries

## Performance Expectations

Based on v1 experience and architectural improvements:

### Training
- **Time**: ~2-3 days on single GPU for 80K pages
- **Memory**: ~8-12GB VRAM with batch_size=4
- **Convergence**: 15-20 epochs typically sufficient

### Inference
- **Speed**: ~50-100 panels/second on GPU
- **Accuracy**: Expected improvements:
  - +30% on image-only queries (vs v1)
  - +20% on semantic clustering (vs v1)
  - Similar or better on multi-modal queries

### Embedding Quality
- **Semantic Coherence**: Better than v1 (less compositional dominance)
- **Modality Independence**: Full support (vs none in v1)
- **Discriminative Power**: Higher for single-modality queries

## Known Limitations

1. **Computational Cost**: Multi-backbone approach is slower than single encoder
2. **Memory Requirements**: Higher than v1 due to multiple encoders
3. **Cold Start**: Requires Stage 2 (PSS) labels for training
4. **Text Quality**: Still dependent on VLM/OCR quality from Stage 1
5. **Panel Detection**: Inherits any panel detection errors from Stage 1

## References

- Version 2.0 possibilities.md: Framework specification
- Model_debugging.md: v1 failure analysis
- ClosureLiteSimple: v1 implementation
- CoSMo paper: PSS methodology
- ComicsPAP: Panel sequence understanding
