# CoSMo PSS Inference & Optimization

## Goals
- Portable path configuration
- Mixed precision & inference mode for backbone
- Two-phase inference (embed precompute + classification-only)
- Dynamic batch fallback for constrained GPUs

## Added Utilities
1. `env_paths.py` for environment-driven directories.
2. `precompute_embeddings.py` generates page-level visual + text embeddings.
3. `classify_precomputed.py` consumes embeddings for fast taxonomy updates.
4. `dynamic_batch.py` reduces batch size automatically on OOM.

## Usage

### Precompute Embeddings
```bash
python -m cosmo.precompute_embeddings PSS_ROOT=/data/books PSS_DATA=/data/ann PSS_PRECOMPUTE=/data/precomputed
```

### Classification-Only Inference
```bash
python -m cosmo.classify_precomputed PSS_PRECOMPUTE=/data/precomputed PSS_CLASSIFIER_CKPT=/checkpoints/best_BookBERT.pt
```

### Environment Variables
- `PSS_ROOT`: Root directory for book images (default: `<project_root>/data/books`)
- `PSS_DATA`: Data directory for annotations (default: `<project_root>/src/cosmo/data`)
- `PSS_PRECOMPUTE`: Directory for precomputed embeddings (default: `<project_root>/cosmo/precomputed`)
- `PSS_CHECKPOINTS`: Directory for model checkpoints (default: `<project_root>/cosmo/checkpoints`)
- `PSS_OUTPUT`: Output directory for results (default: `<project_root>/cosmo/output`)
- `PSS_FP16`: Enable FP16 mixed precision (default: `1`)
- `PSS_VIS_MODEL`: Visual backbone model (default: `google/siglip-so400m-patch14-384`)
- `PSS_TEXT_MODEL`: Text embedding model (default: `Qwen/Qwen3-Embedding-0.6B`)
- `PSS_PRECOMP_BATCH`: Batch size for precomputation (default: `32`)
- `PSS_CLASSIFIER_CKPT`: Path to classifier checkpoint
- `PSS_NUM_HEADS`: Number of attention heads (default: `4`)
- `PSS_NUM_LAYERS`: Number of transformer layers (default: `4`)
- `PSS_DROPOUT`: Dropout probability (default: `0.4`)
- `PSS_HIDDEN`: Hidden dimension (default: `256`)
- `PSS_VIS_DIM`: Visual feature dimension (default: `1152`)
- `PSS_TXT_DIM`: Text feature dimension (default: `1024`)
- `PSS_NUM_CLASSES`: Number of output classes (default: `9`)
- `PSS_BERT_INPUT`: BERT input dimension (default: `768`)
- `PSS_PROJ_DIM`: Projection dimension (default: `1024`)
- `PSS_POSITIONAL`: Positional embedding type (default: `absolute`)

## Performance Expectations
- End-to-end (no cache): 100–120 ms/page with SigLIP so400m.
- Cached classification: <10 ms/page; cost per 1K pages drops drastically.

## Architecture

### Two-Phase Inference Flow
1. **Phase 1: Precompute Embeddings** (`precompute_embeddings.py`)
   - Loads visual backbone (SigLIP/CLIP) with mixed precision support
   - Processes book images in batches with automatic FP16 casting
   - Loads text embedding model (Qwen/SentenceTransformer)
   - Extracts and saves visual and text embeddings per book
   - Uses `torch.inference_mode()` and `torch.cuda.amp.autocast()` for efficiency

2. **Phase 2: Classification** (`classify_precomputed.py`)
   - Loads precomputed embeddings from disk
   - Fuses visual and text features
   - Runs lightweight BookBERT classifier
   - Saves probability distributions for each page

### BookBERT Model Architecture
The `BookBERTMultimodal2` model consists of:
- Input projection layer: fuses visual and text features
- Positional embeddings (absolute or learned)
- Transformer encoder with configurable layers and attention heads
- Dropout regularization
- Classification head

The model includes a `forward_sequence` method for single-sequence inference, optimized for the classification-only phase.

### Dynamic Batch Handling
The `dynamic_batch.py` utility automatically reduces batch size when GPU memory is insufficient:
- Starts with a specified batch size
- On CUDA OOM error, empties cache and halves batch size
- Retries until successful or batch size reaches zero

## Integration with Existing Pipeline
The `pss_multimodal.py` training script has been updated to:
- Use `env_paths.py` for path configuration instead of hard-coded Windows paths
- Support environment variable overrides for all paths
- Maintain backward compatibility with existing functionality

## Next Steps
- Integrate windowed evaluation for very long omnibuses.
- Add optional hierarchical boundary labels (Episode-Start vs Story-Start).
- Distill classifier head for ultra-fast CPU inference.
- Add memory profiling and benchmark tools.

## Implementation Details

### Mixed Precision Support
Precomputation uses PyTorch's automatic mixed precision (AMP):
```python
with torch.inference_mode(), torch.cuda.amp.autocast(enabled=USE_FP16, dtype=torch.float16 if USE_FP16 else torch.float32):
    out = backbone(**inputs)
```

This reduces memory footprint by ~50% and increases throughput by ~2x on modern GPUs.

### Path Resolution
The `env_paths.py` utility provides a centralized way to manage paths:
- Resolves paths relative to project root
- Supports environment variable overrides
- Handles user home directory expansion
- Returns Path objects for cross-platform compatibility

### Error Handling
- Missing book directories are logged and skipped
- OCR file loading failures return empty strings
- Length mismatches between visual and text embeddings are detected and logged
- CUDA out of memory errors trigger automatic batch size reduction

## Testing
To verify the implementation:
1. Check that imports work correctly from the cosmo directory
2. Verify path resolution with different environment variables
3. Test precomputation on a small subset of books
4. Test classification with precomputed embeddings
5. Verify mixed precision reduces memory usage
6. Test dynamic batch handling with constrained GPU memory

## Compatibility Notes
- Python 3.8+ required
- PyTorch 1.10+ required for mixed precision support
- Transformers library required for visual backbones
- SentenceTransformers required for text embeddings
- Compatible with both CUDA and CPU execution
