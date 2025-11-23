# Stage 3: Domain-Adapted Multimodal Panel Feature Generation

## Quick Start

Stage 3 generates rich multimodal panel embeddings from narrative pages identified by Stage 2 (CoSMo/PSS).

### Installation

```bash
# Install requirements (from repository root)
pip install -r requirements.txt

# Additional dependencies for Stage 3
pip install timm  # For ResNet models
```

### Training

```bash
cd src/version2

# Basic training with default config
python train_stage3.py \
  --data_root /path/to/comics \
  --train_pss_labels /path/to/train_pss.json \
  --val_pss_labels /path/to/val_pss.json

# Training with multi-backbone and wandb logging
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
  --run_name my_stage3_experiment
```

### Inference

```python
from stage3_panel_features_framework import PanelFeatureExtractor
import torch
from PIL import Image
from torchvision import transforms as T

# Load trained model
model = PanelFeatureExtractor(
    visual_backbone='both',
    feature_dim=512
)
checkpoint = torch.load('checkpoints/stage3/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('panel.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Get embedding (image-only query)
with torch.no_grad():
    embedding = model.encode_image_only(image_tensor)
    
print(f"Panel embedding shape: {embedding.shape}")  # (1, 512)
```

## Architecture Overview

Stage 3 consists of three main components:

### 1. Visual Encoders

- **DomainAdaptedSigLIP**: Pre-trained SigLIP with domain adaptation layers
- **DomainAdaptedResNet**: ResNet50 with domain adaptation layers
- **MultiBackboneVisualEncoder**: Fuses multiple visual backbones

### 2. Text & Composition Encoders

- **TextEncoder**: SentenceTransformer-based text encoding
- **CompositionEncoder**: MLP for spatial/layout features

### 3. Adaptive Fusion

- **AdaptiveFusion**: Modality-aware fusion mechanism
- Handles missing modalities gracefully
- Prevents dummy input contamination (v1 issue)

## Key Improvements Over v1

| Feature | v1 (ClosureLiteSimple) | Stage 3 |
|---------|----------------------|---------|
| Visual Encoder | Single ViT | SigLIP + ResNet |
| Single-Modality Queries | Non-discriminative | Fully supported |
| Fusion Mechanism | GatedFusion | AdaptiveFusion |
| Missing Modalities | Dummy inputs | Modality masks |
| Domain Adaptation | None | Fine-tunable layers |

See `documentation/future_work/Stage 3 Architecture.md` for detailed comparison.

## Data Format

### Input Directory Structure

```
data_root/
├── book_001/
│   ├── page_001.jpg
│   ├── page_001.json
│   ├── page_002.jpg
│   ├── page_002.json
│   └── ...
├── book_002/
│   └── ...
└── pss_labels.json  # From Stage 2
```

### PSS Labels Format (from Stage 2)

```json
{
  "book_001": {
    "page_001": "narrative",
    "page_002": "advertisement",
    "page_003": "narrative",
    "page_004": "cover"
  },
  "book_002": {
    ...
  }
}
```

### Panel JSON Format (from Stage 1)

```json
{
  "image_width": 1988,
  "image_height": 3057,
  "panels": [
    {
      "bbox": [100, 200, 500, 400],
      "text": "Panel dialogue and narration",
      "confidence": 0.95
    },
    {
      "bbox": [600, 200, 900, 400],
      "text": "Another panel's text",
      "confidence": 0.92
    }
  ]
}
```

## Training Objectives

Stage 3 uses three complementary objectives:

### 1. Contrastive Learning
Panels from same page should be similar.

### 2. Panel Reconstruction
Predict masked panel from context of other panels.

### 3. Modality Alignment
Vision and text for same panel should align.

**Combined Loss**:
```
L = L_contrastive + 0.5 * L_reconstruction + 0.3 * L_alignment
```

## Configuration

Edit `stage3_config.yaml` for experiment settings:

```yaml
model:
  visual_backbone: 'both'  # 'siglip', 'resnet', or 'both'
  visual_fusion: 'attention'  # 'concat', 'attention', or 'gate'
  feature_dim: 512
  freeze_backbones: true

training:
  batch_size: 4
  epochs: 20
  learning_rate: 0.0001
  temperature: 0.07
```

## Common Use Cases

### Use Case 1: Image Similarity Search

```python
# Extract embeddings for all panels in database
database_embeddings = []
for page in dataset:
    panel_images = load_panels(page)
    with torch.no_grad():
        embeddings = model.encode_image_only(panel_images)
    database_embeddings.append(embeddings)

# Query with new image
query_image = load_image('query.jpg')
query_embedding = model.encode_image_only(query_image)

# Find similar panels
similarities = cosine_similarity(query_embedding, database_embeddings)
top_k = similarities.topk(k=10)
```

### Use Case 2: Multi-Modal Query

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Query with both image and text
image = load_image('panel.jpg')
text = "A superhero flying through the sky"

batch = {
    'images': transform(image).unsqueeze(0),
    'input_ids': tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids'],
    'attention_mask': tokenizer(text, return_tensors='pt', padding=True, truncation=True)['attention_mask'],
    'comp_feats': torch.zeros(1, 7),  # Placeholder for query
    'modality_mask': torch.tensor([[1.0, 1.0, 0.0]])  # Vision + text, no comp
}

with torch.no_grad():
    query_embedding = model(batch)
```

### Use Case 3: Batch Processing Pages

```python
from torch.utils.data import DataLoader
from stage3_dataset import Stage3PanelDataset, collate_stage3

# Create dataset
dataset = Stage3PanelDataset(
    root_dir='/path/to/comics',
    pss_labels_path='pss_labels.json',
    only_narrative=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collate_stage3,
    num_workers=4
)

# Process all pages
all_embeddings = []
for batch in dataloader:
    B, N = batch['panel_mask'].shape
    
    # Flatten for processing
    flat_batch = {
        'images': batch['images'].view(B*N, 3, 224, 224),
        'input_ids': batch['input_ids'].view(B*N, -1),
        'attention_mask': batch['attention_mask'].view(B*N, -1),
        'comp_feats': batch['comp_feats'].view(B*N, 7),
        'modality_mask': batch['modality_mask'].view(B*N, 3)
    }
    
    with torch.no_grad():
        embeddings = model(flat_batch)
        embeddings = embeddings.view(B, N, -1)
    
    all_embeddings.append(embeddings)
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` (try 2 or 1)
- Reduce `feature_dim` (try 256 instead of 512)
- Use single backbone (`visual_backbone='siglip'`)
- Reduce `max_panels_per_page`

### Poor Image-Only Queries

- Check if `freeze_backbones=True` - try unfreezing for fine-tuning
- Increase `alignment_weight` in training
- Use `visual_backbone='both'` for richer features
- Check if images are properly normalized

### Slow Training

- Increase `num_workers` for data loading
- Use `pin_memory=True`
- Enable mixed precision training (add to script)
- Use single backbone if multi-backbone is too slow

### Modality Imbalance

- Adjust loss weights in config
- Check `modality_mask` to ensure proper flagging
- Filter dataset to only include pages with all modalities

## Integration with Pipeline

### Input from Stage 2 (PSS)

Stage 3 expects PSS labels indicating which pages are narrative:

```bash
# Stage 2 output
pss_labels.json  # Page classifications

# Stage 3 processes only 'narrative' pages
```

### Output to Stage 4

Stage 3 generates panel embeddings for sequence modeling:

```python
{
  'panel_embeddings': (num_pages, max_panels, 512),
  'panel_masks': (num_pages, max_panels),
  'metadata': [...]
}
```

Stage 4 will use these as input to its Transformer encoder.

## Performance Benchmarks

Based on expected performance (to be updated with actual results):

| Metric | Expected Value |
|--------|---------------|
| Training time (80K pages) | 2-3 days (single GPU) |
| Inference speed | 50-100 panels/sec (GPU) |
| Memory usage | 8-12 GB VRAM |
| Image-only retrieval accuracy | +30% vs v1 |
| Multi-modal retrieval accuracy | Similar or better vs v1 |

## Citation

If you use Stage 3 in your research, please cite:

```
Version 2.0 Comic Analysis Framework - Stage 3: Domain-Adapted Multimodal Panel Feature Generation
https://github.com/RichardScottOZ/Comic-Analysis
```

## References

- [Version 2.0 possibilities](../../documentation/future_work/Version%202.0%20possibilities.md)
- [Stage 3 Architecture](../../documentation/future_work/Stage%203%20Architecture.md)
- [Model Debugging (v1 issues)](../../documentation/Model_debugging.md)
- [CoSMo Paper](https://github.com/mserra0/CoSMo-ComicsPSS)

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
