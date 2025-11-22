# Stage 4: Semantic Sequence Modeling

## Overview

Stage 4 is the fourth component of the Version 2.0 framework, responsible for modeling sequential relationships between panels and generating contextualized semantic embeddings. It takes panel embeddings from Stage 3 and learns narrative flow, character coherence, and visual/textual closure.

## Position in Version 2.0 Pipeline

```
Stage 1: Raw Data Ingestion
    ↓
Stage 2: Page Stream Segmentation (PSS)
    ↓
Stage 3: Panel Feature Generation
    ↓
Stage 4: Semantic Sequence Modeling ← YOU ARE HERE
    ↓
Stage 5: Queryable Embeddings
```

## Theoretical Foundation

Stage 4 is inspired by two major research works:

### 1. ComicsPAP (arXiv:2503.08561)

ComicsPAP introduces a "Pick-a-Panel" paradigm with five key tasks:

1. **Sequence Filling**: Select the right panel to reconstruct story progression
2. **Character Coherence**: Maintain consistent character appearances across panels
3. **Visual Closure**: Infer correct continuation of visual actions/events
4. **Text Closure**: Select panels where text/dialogue appropriately continues
5. **Caption Relevance**: Align panels with appropriate captions

Key insight: State-of-the-art LMMs perform near chance on these tasks, showing that sequential comic understanding requires specialized modeling.

### 2. Text-Cloze Task (arXiv:2403.03719)

The Text-Cloze task tests if a model can select the correct text for a comic panel given its visual and textual context:

- **Challenge**: Comics require "closure" - inferring missing information across panels
- **Approach**: Multimodal transformer with domain-adapted visual encoder
- **Performance**: Multimodal transformers outperform RNNs by ~10%

### Key Insights for Stage 4

1. **Sequential Context is Essential**: Single-panel analysis is insufficient
2. **Multimodal Integration**: Visual and text must be jointly modeled
3. **Narrative Understanding**: Models must learn causality and story flow
4. **Character Tracking**: Maintaining identity across style variations is crucial

## Architecture

### Core Components

#### 1. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    # Sinusoidal encoding for panel positions
    # Preserves sequential information in transformer
```

Adds position-aware signal to panel embeddings so the model knows panel order.

#### 2. Panel Sequence Transformer

```python
class PanelSequenceTransformer(nn.Module):
    # BERT-like transformer encoder
    # Processes sequences of panel embeddings
    # Outputs contextualized representations
```

**Architecture**:
- 6 transformer encoder layers (configurable)
- 8 attention heads per layer
- Pre-norm for training stability
- Attention masking for variable-length sequences

**Key Feature**: Strip-level aggregation using learned query:
```python
# Aggregate panels into semantic strip embedding
strip_query = learnable_parameter  # (1, 1, D)
strip_embedding = MultiheadAttention(
    query=strip_query,
    key=contextualized_panels,
    value=contextualized_panels
)
```

### Task-Specific Heads

#### 1. Panel Picking Head (ComicsPAP Sequence Filling)

```
Input: context_embedding (B, D), candidates (B, K, D)
    ↓
Concatenate context with each candidate
    ↓
Score network: Linear(2D → D) → GELU → Linear(D → 1)
    ↓
Output: scores (B, K) for each candidate
```

**Training**: Cross-entropy with correct panel index

**Usage**: Given sequence with masked panel, select correct one from candidates

#### 2. Character Coherence Head

```
Input: panel_sequence (B, N, D), candidate_panel (B, D)
    ↓
Average existing panels → context
    ↓
Concatenate context + candidate
    ↓
Consistency scorer → coherence_score
    ↓
Output: score indicating character consistency
```

**Training**: Binary classification (consistent vs inconsistent)

**Usage**: Ensure selected panels maintain character visual identity

#### 3. Closure Heads (Visual & Text)

```
Input: preceding_panels (B, N, D), candidate_panel (B, D)
    ↓
Use last preceding panel as context
    ↓
Concatenate context + candidate
    ↓
Closure network (3-layer MLP)
    ↓
Output: plausibility score
```

**Training**: Cross-entropy over candidates

**Usage**: 
- Visual Closure: Predict continuation of visual actions
- Text Closure: Predict continuation of dialogue/narration

#### 4. Caption Relevance Head

```
Input: panel_embedding (B, D), caption_embedding (B, D)
    ↓
Bilinear scorer: Bilinear(D, D → 1)
    ↓
Output: relevance score
```

**Training**: Contrastive loss (match panel to correct caption)

**Usage**: Align panels with textual narrative descriptions

#### 5. Text-Cloze Head

```
Input: 
  - visual_context (B, D)
  - surrounding_text (B, D)
  - candidate_texts (B, K, D)
    ↓
Encode context: Linear(2D → D)
    ↓
Score each candidate with bilinear
    ↓
Output: scores (B, K)
```

**Training**: Cross-entropy with correct text index

**Usage**: Predict missing text in panel from visual+textual context

#### 6. Reading Order Head

```
Input: panel_embeddings (B, N, D)
    ↓
For each pair (i, j):
  score = Bilinear(panel_i, panel_j)
    ↓
Output: order_matrix (B, N, N)
```

**Training**: Binary cross-entropy with adjacency matrix

**Usage**: Refine reading order predictions from Stage 1

## Training Strategy

### Multi-Task Learning

Stage 4 uses multi-task learning with task-specific sampling:

```python
task_weights = {
    'panel_picking': 1.0,      # Primary task
    'character_coherence': 0.5,
    'visual_closure': 0.8,
    'text_closure': 0.8,
    'caption_relevance': 0.5,
    'text_cloze': 1.0,         # Primary task
    'reading_order': 0.7
}
```

Each batch randomly samples tasks according to weights.

### Training Objectives

#### 1. Task-Specific Losses

Each task has its own loss:
- Panel picking: Cross-entropy
- Closures: Cross-entropy
- Text-cloze: Cross-entropy
- Reading order: Binary cross-entropy
- Character coherence: Binary classification

#### 2. Contrastive Panel Loss

Encourages panels within same sequence to be similar:

```python
# For each sequence
sim_matrix = normalize(panels) @ normalize(panels).T
# Maximize similarity among panels (excluding self)
loss = -mean(sim_matrix[non_diagonal])
```

**Why**: Panels on same page share style, characters, narrative context

#### 3. Combined Loss

```
L_total = Σ(task_losses) + 
          0.5 * L_contrastive + 
          0.3 * L_reading_order
```

Weights are configurable for experimentation.

## Data Format

### Input from Stage 3

```python
{
  'panel_embeddings': (B, N, 512),  # From Stage 3
  'panel_mask': (B, N),              # Valid panel indicators
  'metadata': {
    'book_id': str,
    'page_name': str,
    'num_panels': int
  }
}
```

### Training Sample Creation

#### Panel Picking Example:
```python
# Original sequence: [p1, p2, p3, p4, p5]
# Randomly mask position 2
# Context: [p1, p3, p4, p5]
# Candidates: [p2, p4, p1, p5, p3]  # Shuffled, correct is p2
# Label: 0  # Index of correct panel in candidates
```

#### Closure Example:
```python
# Original: [p1, p2, p3, p4, p5]
# Split at position 3
# Preceding: [p1, p2]
# Candidates: [p3, p5, p4, p1, p2]  # Shuffled, correct is p3
# Label: 0
```

### Output to Stage 5

```python
{
  'contextualized_panels': (num_pages, max_panels, 512),
  'strip_embeddings': (num_pages, 512),
  'panel_masks': (num_pages, max_panels)
}
```

These embeddings capture:
- Narrative context (from transformer)
- Panel relationships (from attention)
- Sequential dependencies (from positional encoding)

## Key Design Decisions

### 1. Transformer over RNN

**Choice**: Transformer encoder

**Rationale**:
- Better long-range dependencies
- Parallel processing (faster training)
- Attention provides interpretability
- Proven success in Text-Cloze research

**Evidence**: Text-Cloze paper shows 10% improvement over RNN baseline

### 2. Pre-Norm Transformer

**Choice**: Layer normalization before attention/FFN

**Rationale**:
- More stable training
- Better gradient flow
- Standard in modern transformers (GPT, BERT variants)

### 3. Discriminative Tasks

**Choice**: Select from candidates (not generative)

**Rationale**:
- More tractable for current compute
- ComicsPAP framework proven effective
- Easier to evaluate and debug
- Can extend to generative later

### 4. Separate Closure Heads

**Choice**: Distinct heads for visual vs text closure

**Rationale**:
- Different feature requirements
- Allows specialized learning
- ComicsPAP treats them separately
- Can weight differently in training

### 5. Multi-Task Learning

**Choice**: Single model for all tasks

**Rationale**:
- Shared representations improve generalization
- Efficient use of parameters
- Tasks reinforce each other (e.g., closure helps picking)
- Matches real-world understanding (humans use all cues)

## Integration with Other Stages

### From Stage 3

**Input**: Panel embeddings with modality fusion
- Each panel: (512,) multimodal embedding
- Already contains visual, text, compositional information
- Stage 4 adds sequential context

### To Stage 5

**Output**: Contextualized embeddings for storage
- Panel-level: Rich contextual understanding
- Strip-level: Semantic summary of entire sequence
- Ready for Zarr storage and querying

### Full Pipeline Context

```
Stage 1: Raw comic → Panel crops + text
Stage 2: Page type classification
Stage 3: Panel embedding (V+T+C fusion)
Stage 4: Sequential contextualization ← adds narrative understanding
Stage 5: Storage and querying
```

## Performance Expectations

Based on related research and architecture:

### Training
- **Time**: 3-5 days for 100K sequences (single GPU)
- **Memory**: 10-16GB VRAM at batch_size=8
- **Convergence**: 20-30 epochs typically sufficient

### Accuracy (Expected)
- **Panel Picking**: 60-70% top-1 accuracy
- **Closure Tasks**: 55-65% top-1 accuracy
- **Reading Order**: 75-85% pairwise accuracy
- **Text-Cloze**: 50-60% top-1 accuracy

**Baseline**: Random chance = 20% (5 candidates)

### Improvements Over Stage 3
- +15-25% on narrative understanding tasks
- Better character tracking across panels
- Improved sequential coherence

## Comparison: v1 vs Stage 4

| Aspect | v1 (ClosureLiteSimple) | Stage 4 |
|--------|----------------------|---------|
| **Sequential Modeling** | Simple attention | Transformer encoder |
| **Context** | Page-level only | Full sequence context |
| **Tasks** | MPM, POP, RPP | ComicsPAP + Text-Cloze |
| **Narrative Understanding** | Limited | Strong (multiple tasks) |
| **Character Tracking** | None | Dedicated head |
| **Closure** | None | Visual + Text heads |
| **Reading Order** | Basic prediction | Pairwise refinement |

## Usage Examples

### Training

```bash
python train_stage4.py \
  --train_embeddings /path/to/train_embeddings.zarr \
  --train_metadata /path/to/train_metadata.json \
  --val_embeddings /path/to/val_embeddings.zarr \
  --val_metadata /path/to/val_metadata.json \
  --d_model 512 \
  --num_layers 6 \
  --nhead 8 \
  --batch_size 8 \
  --epochs 30 \
  --use_wandb
```

### Inference: Encode Comic Strip

```python
from stage4_sequence_modeling_framework import Stage4SequenceModel
import torch

# Load model
model = Stage4SequenceModel(d_model=512, num_layers=6)
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
model.eval()

# Load panel embeddings from Stage 3
panel_embeddings = load_stage3_embeddings(page_id)  # (N, 512)
panel_mask = torch.ones(panel_embeddings.shape[0])

# Add batch dimension
panel_embeddings = panel_embeddings.unsqueeze(0)  # (1, N, 512)
panel_mask = panel_mask.unsqueeze(0)  # (1, N)

# Encode strip
with torch.no_grad():
    strip_embedding = model.encode_strip(panel_embeddings, panel_mask)

print(f"Strip embedding shape: {strip_embedding.shape}")  # (1, 512)
```

### Inference: Panel Picking Task

```python
# Create masked sequence
context_embedding = compute_context(panels_except_masked)
candidates = get_candidate_panels(5)  # 5 candidates including correct

# Predict
with torch.no_grad():
    scores = model.panel_picking_head(
        context_embedding.unsqueeze(0),
        candidates.unsqueeze(0)
    )
    predicted_idx = scores.argmax(dim=-1).item()

print(f"Predicted panel: {predicted_idx}")
```

### Inference: Visual Closure

```python
# Preceding panels before gap
preceding = panels[:split_point]  # (N, 512)
candidate_panel = next_panel  # (512,)

# Predict closure plausibility
with torch.no_grad():
    score = model.visual_closure_head(
        preceding.unsqueeze(0),
        candidate_panel.unsqueeze(0)
    )
    
print(f"Closure plausibility: {score.item():.4f}")
```

## Future Enhancements

### 1. Generative Extension

Current: Discriminative (select from candidates)
Future: Generative (produce panel descriptions, predict text)

**Approach**:
- Add decoder to transformer
- Train with autoregressive objective
- Generate panel descriptions, dialogue

### 2. Character Re-identification

Current: Character coherence as binary classification
Future: Explicit character tracking across panels

**Approach**:
- Character detection from Stage 1
- Character embeddings per panel
- Cross-panel matching network
- Character-aware attention

### 3. Hierarchical Modeling

Current: Single-page sequences
Future: Multi-page narrative arcs

**Approach**:
- Page-level transformer (Stage 4)
- Issue-level transformer (new stage)
- Arc-level embeddings
- Long-range dependencies

### 4. Causal Attention

Current: Bidirectional attention (BERT-style)
Future: Causal attention for prediction

**Approach**:
- Autoregressive generation
- Panel-by-panel prediction
- Real-time reading simulation

### 5. Cross-Modal Attention

Current: Process fused embeddings from Stage 3
Future: Direct cross-modal attention

**Approach**:
- Separate visual, text encoders
- Cross-attention between modalities
- More flexible fusion

## Known Limitations

1. **Fixed Sequence Length**: Max 16 panels (memory constraint)
   - Longer sequences need truncation or sliding window
   
2. **Discriminative Only**: Candidates required for training
   - Need negative sampling strategy
   - Can't generate novel content
   
3. **No Character Tracking**: Basic coherence, not re-identification
   - Needs character detection integration
   
4. **Page-Level Only**: No multi-page modeling
   - Narrative arcs span multiple pages
   
5. **Training Data**: Requires Stage 3 embeddings
   - End-to-end training not yet supported

## References

- **ComicsPAP**: https://arxiv.org/abs/2503.08561
- **Text-Cloze**: https://arxiv.org/abs/2403.03719
- **CoMix**: https://github.com/emanuelevivoli/CoMix
- **Version 2.0 Framework**: ../Version 2.0 possibilities.md
- **Stage 3 Architecture**: Stage 3 Architecture.md
- **ComicBERT**: https://link.springer.com/chapter/10.1007/978-3-031-70645-5_16

## Conclusion

Stage 4 brings sequential narrative understanding to the Comic Analysis pipeline. By implementing ComicsPAP and Text-Cloze inspired tasks within a transformer architecture, it captures:

- Panel dependencies
- Narrative flow
- Character coherence
- Visual and textual closure
- Reading order

These contextualized embeddings enable Stage 5 to provide rich semantic search and narrative analysis capabilities that were impossible with isolated panel embeddings alone.
