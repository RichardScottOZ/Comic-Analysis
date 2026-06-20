# Stage 4: Semantic Sequence Modeling

## Quick Start

Stage 4 models sequential relationships between panels using Transformer architecture, implementing ComicsPAP and Text-Cloze inspired tasks.

### Installation

```bash
# Install requirements (from repository root)
pip install -r requirements.txt

# Stage 4 has same dependencies as Stage 3
# No additional packages needed
```

### Training

```bash
cd src/version2

# Basic training
python train_stage4.py \
  --train_embeddings /path/to/train_embeddings.zarr \
  --train_metadata /path/to/train_metadata.json \
  --val_embeddings /path/to/val_embeddings.zarr \
  --val_metadata /path/to/val_metadata.json

# Advanced training with wandb
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
  --use_wandb \
  --run_name my_stage4_experiment
```

### Inference

```python
from stage4_sequence_modeling_framework import Stage4SequenceModel
import torch

# Load trained model
model = Stage4SequenceModel(
    d_model=512,
    nhead=8,
    num_layers=6
)
checkpoint = torch.load('checkpoints/stage4/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Encode a comic strip
panel_embeddings = torch.randn(1, 10, 512)  # (B, N, D) from Stage 3
panel_mask = torch.ones(1, 10)  # All valid

with torch.no_grad():
    outputs = model(panel_embeddings, panel_mask)
    contextualized_panels = outputs['contextualized_panels']  # (1, 10, 512)
    strip_embedding = outputs['strip_embedding']  # (1, 512)

print(f"Contextualized panels: {contextualized_panels.shape}")
print(f"Strip embedding: {strip_embedding.shape}")
```

## Architecture Overview

### Core Components

1. **Panel Sequence Transformer**
   - Transformer encoder (BERT-like)
   - Processes sequences of panel embeddings
   - Outputs contextualized representations
   - Strip-level aggregation with learned query

2. **Task-Specific Heads**
   - **Panel Picking**: Select correct panel for masked position (ComicsPAP)
   - **Character Coherence**: Maintain character consistency
   - **Visual Closure**: Predict continuation of visual actions
   - **Text Closure**: Predict continuation of text/dialogue
   - **Caption Relevance**: Align panels with captions
   - **Text-Cloze**: Predict missing text from context
   - **Reading Order**: Refine panel sequencing

## Inspired By

### ComicsPAP (arXiv:2503.08561)

Five key tasks for comic understanding:
1. Sequence Filling
2. Character Coherence
3. Visual Closure
4. Text Closure
5. Caption Relevance

### Text-Cloze (arXiv:2403.03719)

- Contextual text prediction
- Multimodal transformer approach
- 10% improvement over RNN baselines

## Training Tasks

### Multi-Task Learning

Stage 4 trains on multiple tasks simultaneously:

| Task | Weight | Description |
|------|--------|-------------|
| Panel Picking | 1.0 | Select correct panel for sequence |
| Visual Closure | 0.8 | Predict visual continuation |
| Text Closure | 0.8 | Predict text continuation |
| Reading Order | 0.7 | Refine panel sequence |
| Character Coherence | 0.5 | Maintain character identity |
| Caption Relevance | 0.5 | Align panels with captions |
| Text-Cloze | 1.0 | Predict missing text |

### Training Objectives

1. **Task-Specific Losses**: Cross-entropy for each task
2. **Contrastive Loss**: Panels in same sequence should be similar
3. **Combined Loss**: Weighted sum of all objectives

## Data Format

### Input from Stage 3

Stage 4 expects pre-computed panel embeddings:

```
embeddings_dir/
├── panel_embeddings.zarr  # From Stage 3
│   └── [embedding_0, embedding_1, ...]
└── metadata.json          # Sequence metadata
    └── [{
         "book_id": "comic_001",
         "page_name": "page_05",
         "num_panels": 6,
         "embedding_indices": [10, 11, 12, 13, 14, 15]
        }, ...]
```

### Output to Stage 5

```python
{
  'contextualized_panels': (num_pages, max_panels, 512),
  'strip_embeddings': (num_pages, 512),
  'panel_masks': (num_pages, max_panels)
}
```

## Key Features

### 1. Sequential Context

Unlike Stage 3 (independent panels), Stage 4 models panel dependencies:

```python
# Stage 3: Each panel processed independently
panel_emb_1 = stage3(panel_1)  # No context
panel_emb_2 = stage3(panel_2)  # No context

# Stage 4: Panels processed with sequence context
contextualized = stage4([panel_emb_1, panel_emb_2, ...])
# Now panel_1's embedding considers panel_2, panel_3, etc.
```

### 2. Strip-Level Embeddings

Aggregates entire sequence into single embedding:

```python
strip_embedding = model.encode_strip(panel_embeddings, panel_mask)
# Use for: similarity search, recommendation, clustering
```

### 3. Multiple Task Heads

Single model supports all ComicsPAP tasks:

```python
# Panel picking
scores = model.panel_picking_head(context, candidates)

# Visual closure
score = model.visual_closure_head(preceding, candidate)

# Reading order
order_matrix = model.reading_order_head(panels)
```

## Common Use Cases

### Use Case 1: Comic Strip Similarity

```python
# Encode multiple strips
strips = [load_strip(i) for i in range(100)]
embeddings = []

for strip in strips:
    emb = model.encode_strip(strip['panels'], strip['mask'])
    embeddings.append(emb)

embeddings = torch.stack(embeddings)

# Find similar strips
query_emb = model.encode_strip(query_strip)
similarities = cosine_similarity(query_emb, embeddings)
top_k = similarities.topk(k=10)
```

### Use Case 2: Panel Picking (Missing Panel)

```python
# Sequence with masked position
masked_seq = [p1, p2, MASK, p4, p5]
context = compute_context(masked_seq)

# Candidates (including correct panel p3)
candidates = [p3, p5, p1, p2, p4]

# Predict
scores = model.panel_picking_head(context, candidates)
predicted_idx = scores.argmax()  # Should point to p3
```

### Use Case 3: Narrative Continuation

```python
# Given first half of sequence
preceding_panels = sequence[:N//2]

# Generate candidates for next panel
candidates = generate_candidates(5)

# Score each candidate
scores = []
for candidate in candidates:
    score = model.visual_closure_head(preceding_panels, candidate)
    scores.append(score)

# Select best continuation
best_idx = torch.tensor(scores).argmax()
next_panel = candidates[best_idx]
```

### Use Case 4: Reading Order Correction

```python
# Shuffled panels
shuffled_panels = random.shuffle(correct_sequence)

# Predict pairwise ordering
order_matrix = model.reading_order_head(shuffled_panels)

# Reconstruct sequence (greedy or beam search)
corrected_sequence = reconstruct_order(order_matrix)
```

## Configuration

Edit training hyperparameters:

```bash
# Model architecture
--d_model 512           # Model dimension
--nhead 8               # Attention heads
--num_layers 6          # Transformer layers
--dim_feedforward 2048  # FFN dimension

# Training
--batch_size 8
--epochs 30
--lr 1e-4
--weight_decay 0.01

# Task weights
--contrastive_weight 0.5
--reading_order_weight 0.3

# Data
--min_panels 3
--max_panels 16
--num_candidates 5
```

## Performance Expectations

Based on related research:

| Metric | Expected | Baseline (Random) |
|--------|----------|-------------------|
| Panel Picking Accuracy | 60-70% | 20% |
| Visual Closure Accuracy | 55-65% | 20% |
| Text Closure Accuracy | 50-60% | 20% |
| Reading Order Accuracy | 75-85% | 50% |

**Training Time**: 3-5 days (100K sequences, single GPU)
**Memory**: 10-16GB VRAM at batch_size=8

## Comparison: Stage 3 vs Stage 4

| Aspect | Stage 3 | Stage 4 |
|--------|---------|---------|
| **Input** | Panel images/text/comp | Panel embeddings |
| **Processing** | Independent panels | Sequential context |
| **Output** | Panel embeddings | Contextualized + strip embeddings |
| **Context** | None | Full sequence |
| **Tasks** | Contrastive/reconstruction | ComicsPAP + Text-Cloze |
| **Narrative** | Not modeled | Explicitly modeled |

Stage 3 creates rich panel representations.  
Stage 4 adds narrative understanding.

## Troubleshooting

### Out of Memory

- Reduce `batch_size` (try 4 or 2)
- Reduce `max_panels` (try 12 instead of 16)
- Reduce `num_layers` (try 4 instead of 6)
- Use gradient checkpointing (add to code)

### Poor Task Performance

- Check if embeddings from Stage 3 are discriminative
- Increase `num_layers` for more capacity
- Adjust task weights to prioritize difficult tasks
- Verify candidate generation is correct

### Slow Training

- Increase `num_workers` for data loading
- Use mixed precision training (add to code)
- Reduce `dim_feedforward` (try 1024)
- Use smaller transformer (4 layers, 4 heads)

### Convergence Issues

- Lower learning rate (try 5e-5)
- Increase warmup epochs
- Check gradient norms (should be < 10)
- Verify data quality (no NaN embeddings)

## Integration with Pipeline

### From Stage 3

Stage 4 requires pre-computed embeddings:

```bash
# 1. Run Stage 3 to generate embeddings
python generate_stage3_embeddings.py \
  --model_checkpoint stage3_best.pt \
  --output_path embeddings.zarr

# 2. Create metadata file
python create_metadata.py \
  --embeddings embeddings.zarr \
  --output metadata.json

# 3. Train Stage 4
python train_stage4.py \
  --train_embeddings embeddings.zarr \
  --train_metadata metadata.json
```

### To Stage 5

Stage 4 outputs are stored for querying:

```python
# Generate all contextualized embeddings
contextualized_data = generate_stage4_embeddings(
    model, all_sequences
)

# Save to Zarr for Stage 5
save_to_zarr(
    contextualized_data,
    output_path='stage4_embeddings.zarr'
)
```

## References

- [ComicsPAP Paper](https://arxiv.org/abs/2503.08561)
- [Text-Cloze Paper](https://arxiv.org/abs/2403.03719)
- [Stage 4 Architecture](../../documentation/future_work/Stage%204%20Architecture.md)
- [CoMix Repository](https://github.com/emanuelevivoli/CoMix)

## Citation

If you use Stage 4 in your research:

```
Version 2.0 Comic Analysis Framework - Stage 4: Semantic Sequence Modeling
Based on ComicsPAP and Text-Cloze methodologies
https://github.com/RichardScottOZ/Comic-Analysis
```

## Future Work

- [ ] Generative extension (generate text/descriptions)
- [ ] Character re-identification integration
- [ ] Multi-page narrative modeling
- [ ] Hierarchical transformers (page → issue → arc)
- [ ] Cross-modal attention mechanisms

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
