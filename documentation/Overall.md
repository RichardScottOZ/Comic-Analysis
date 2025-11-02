# CoMix Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack & Environment](#tech-stack--environment)
3. [Core Workflow](#core-workflow)
4. [Data Preparation Pipeline](#data-preparation-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Quality Assurance & Analysis](#quality-assurance--analysis)
7. [Command Reference](#command-reference)
8. [Troubleshooting & Tips](#troubleshooting--tips)

---

## Project Overview

CoMix is a comprehensive framework for understanding comics and manga through machine learning. The system processes comic book files (CBZ, PDF, EPUB, etc.), performs panel detection, extracts text using Vision-Language Models (VLMs), and aligns these different data sources to create high-quality datasets for training ML models on tasks like panel detection, reading order prediction, captioning, and art style analysis.

### Key Goals
- Replicate and benchmark academic paper results for comic analysis
- Create clean, aligned datasets from multiple data sources (R-CNN detections + VLM analysis)
- Train models that understand panel composition, reading order, and narrative flow
- Enable semantic search and analysis across large comic collections

---

## Tech Stack & Environment

### Core Technologies
- **Language**: Python 3.8+
- **ML Framework**: PyTorch
- **Data Processing**: Pandas, OpenCV
- **Environment Management**: Conda (recommended)
- **Storage**: External drives (E:\, D:\) for datasets

### Installation
```bash
# Clone and install in editable mode
pip install -e .

# Note: tokenizers requires Rust compiler
```

### Key Dependencies
- PyTorch with CUDA support (for GPU training)
- transformers, datasets (HuggingFace ecosystem)
- wandb (experiment tracking)
- zarr, xarray (embedding storage)
- PIL, opencv-python (image processing)
- Various VLM-specific packages (qwen_vl_utils, etc.)

### Environment Setup
```bash
conda activate sfmcp  # or ai_scientist for some tasks
```

---

## Core Workflow

The CoMix pipeline consists of four major stages:

### 1. Data Acquisition & Conversion
Convert raw comic files to a standard format (CBZ) and extract images.

**Sources**: Calibre Library, Humble Bundle, DriveThru Comics, Amazon dataset

### 2. Dual Analysis Pipeline
Run two parallel analyses on extracted images:
- **R-CNN Panel Detection**: Faster R-CNN to detect panel bounding boxes (COCO format)
- **VLM Text Extraction**: Vision-Language Models to extract dialogue, narration, SFX, characters

### 3. Alignment & Quality Filtering
Match R-CNN detections with VLM outputs, compute quality metrics, and filter for "perfect matches":
- Panel count alignment (R-CNN panels ≈ VLM panels)
- Text coverage quality
- Reading order coherence
- Generate manifests for training

### 4. Model Training
Train Closure-Lite models on aligned datasets for:
- Reading order prediction
- Panel relationship understanding
- Attention-based page embeddings

---

## Data Preparation Pipeline

### Step 1: Conversion to CBZ

Convert various formats (PDF, EPUB, etc.) to CBZ archives.

```powershell
python benchmarks\detections\openrouter\convert_to_cbz.py \
  --input-dir "D:\CalibreComics" \
  --output-dir "D:\CalibreComics_converted" \
  --preserve-structure \
  --skip-existing
```

**Output**: CBZ files preserving directory structure

### Step 2: Image Extraction

Extract images from CBZ/CBR archives.

```powershell
python benchmarks\detections\openrouter\extract_calibre_comics.py \
  --input-dir "D:\CalibreComics" \
  --output-dir "E:\CalibreComics_extracted" \
  --skip-existing
```

**Output**: Directories of extracted page images (JPG, PNG)

### Step 3A: R-CNN Panel Detection

Run Faster R-CNN on extracted images to detect panel bounding boxes.

```powershell
python benchmarks\detections_2000ad\faster_rcnn_calibre.py \
  --input-path "E:\CalibreComics_extracted" \
  --output-path "E:\CalibreComics\test_dections\predictions.json" \
  --save-vis 196 \
  --conf-threshold 0.5
```

**Output**: COCO-format JSON with panel detections

### Step 3B: VLM Text Analysis

Run VLM models on images to extract text, characters, and narrative elements.

```powershell
python benchmarks\detections\openrouter\batch_comic_analysis_multi.py \
  --input-dir "E:\CalibreComics_extracted" \
  --output-dir "E:\CalibreComics_analysis" \
  --model google/gemma-3-4b-it \
  --api-key sk-or-v1-... \
  --temperature 0.1 \
  --skip-existing \
  --max-images 20000000 \
  --error-log-dir "E:\CalibreComics_error_logs"
```

**VLM Model Notes**:
- **Gemma 3 4B**: Baseline performance, good for testing
- **Gemma 3 12B Free**: Better quality, recommended for production
- **Gemini Flash 1.5/2.5**: Good quality but higher cost
- **Meta Llama 4 Scout**: Excellent quality/price ratio

**Output**: Per-page JSON files with panels, dialogue, narration, SFX, characters

### Step 4: Alignment & Quality Analysis

Create alignment CSV matching R-CNN and VLM outputs, with quality scores.

```powershell
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py \
  --coco "E:\CalibreComics\test_dections\predictions.json" \
  --vlm_dir "E:\CalibreComics_analysis" \
  --vlm_map "perfect_match_training\all_vlm_jsons_calibre_training_map.json" \
  --output_csv "calibre_rcnn_vlm_analysis_v2.csv" \
  --num_workers 8 \
  --image_roots "E:\CalibreComics_extracted" \
  --require_image_exists \
  --min_text_coverage 0.6
```

**Key Parameters**:
- `--vlm_map`: Precomputed JSON→image mapping (speeds up matching)
- `--min_text_coverage`: Minimum ratio of panels with text (0.6 = 60%)
- `--num_workers`: Parallel workers (8-16 recommended for Windows)

**Output Files**:
- Main CSV: `calibre_rcnn_vlm_analysis_v2.csv`
- Subset lists: `_perfect_matches.txt`, `_near_perfect.txt`, `_high_quality.txt`
- Training manifest: `_perfect_match_training_manifest.csv`

**Quality Scoring** (0-4 scale):
- **Score 4**: Perfect matches (panel ratio = 1.0, high text coverage)
- **Score 3**: Near-perfect (ratio 0.9-1.1)
- **Score 2**: Acceptable (ratio 0.75-1.25, some text)
- **Score 0-1**: Poor quality (exclude from training)

### Step 5: DataSpec Generation

Convert aligned data to DataSpec format (unified schema for training).

```powershell
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py \
  --coco "E:\CalibreComics\test_dections\predictions.json" \
  --vlm_dir "E:\CalibreComics_analysis" \
  --vlm_map "perfect_match_training\all_vlm_jsons_calibre_training_map.json" \
  --output_csv "calibre_rcnn_vlm_analysis_v2.csv" \
  --num_workers 8 \
  --image_roots "E:\CalibreComics_extracted" \
  --require_image_exists \
  --min_text_coverage 0.6 \
  --emit_dataspec_everything \
  --dataspec_out_dir "perfect_match_training\calibre_dataspec_all" \
  --dataspec_min_det_score 0.5 \
  --dataspec_list_out "perfect_match_training\dataspec_all_list.txt" \
  --dataspec_naming preserve_tree
```

**DataSpec v0.3 Schema** (per page):
```json
{
  "page_image_path": "E:\\CalibreComics_extracted\\series\\issue_01.jpg",
  "panels": [
    {
      "panel_coords": [x, y, width, height],  // pixels
      "text": {
        "dialogue": ["What's happening?", "I don't know!"],
        "narration": ["Meanwhile..."],
        "sfx": ["BOOM!"]
      }
    }
  ]
}
```

### Step 6: Subset Filtering (Perfect Match Only)

Filter DataSpec to include only perfect matches.

```powershell
python benchmarks\detections\openrouter\build_dataspec_subset_list.py \
  --dataspec_list "perfect_match_training\dataspec_all_list.txt" \
  --subset_list "calibre_rcnn_vlm_analysis_v2_perfect_matches.txt" \
  --out_list "perfect_match_training\dataspec_perfect_list.txt"
```

**Output**: Text file with paths to perfect-match DataSpec JSONs only

### Optional: Page Type Annotation

Tag pages by type (cover, interior, ads, etc.) for filtering.

```powershell
python benchmarks\detections\openrouter\annotate_page_types.py \
  --input_csv "calibre_rcnn_vlm_analysis_v2.csv"
```

**Page Types**: cover, back_cover, title_page, splash, interior, ad_third_party, letters, editorial, backmatter, preview_comic_pages, etc.

**Use Case**: Filter training to interior pages only, exclude ads/covers

---

## Training Pipeline

### Closure-Lite Simple Model

The primary model architecture for reading order and panel understanding.

**Architecture Components**:
- Vision encoder: ViT (google/vit-base-patch16-224)
- Text encoder: RoBERTa (roberta-base)
- Panel composition module: MLP fusion
- Attention pooling: Multi-head attention for page embeddings
- Classifiers: Reading order (next panel), panel-on-page detection

### Basic Training Command

```powershell
# PowerShell / WSL
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple" \
  --model simple \
  --batch_size 4 \
  --num_workers 8 \
  --epochs 5 \
  --wandb_project "closure-lite-calibre"
```

**Key Training Parameters**:
- `--batch_size`: 4-8 recommended (GPU memory dependent)
- `--num_workers`: 8 for WSL/Linux, 1-4 for Windows (pickling overhead)
- `--model simple`: Use Closure-Lite Simple (no sequence model)
- `--lr`: Learning rate (default 3e-4)
- `--num_heads`: Attention heads (default 4)
- `--temperature`: Softmax temperature for reading order (default 0.1)

### Training Variants

**1. Base Simple Model** (standard training):
```powershell
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple" \
  --model simple --batch_size 4 --epochs 5 --num_workers 8
```

**2. MPM Denoising** (masked panel modeling with noise):
```powershell
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple_denoise" \
  --model simple --batch_size 4 --epochs 5 --num_workers 8 \
  --mpm_denoise --mpm_weight 0.1 --mpm_noise_std 0.1 --mpm_stopgrad
```

**3. Context Reconstruction** (masked panel modeling with reconstruction):
```powershell
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple_contextrecon" \
  --model simple --batch_size 4 --epochs 5 --num_workers 8 \
  --mpm_context_recon --mpm_weight 0.1 --mpm_stopgrad
```

**4. Combined (Denoise + Context)**:
```powershell
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple_denoise_context" \
  --model simple --batch_size 4 --epochs 5 --num_workers 8 \
  --mpm_denoise --mpm_context_recon --mpm_weight 0.1 --mpm_noise_std 0.1 --mpm_stopgrad
```

### Training Results (5-Epoch Calibre Run)

**Dataset**: 79,113 perfect-match pages from Calibre

**Variant Performance**:
| Variant | Final Loss | MPM Loss | POP Loss | RPP Loss | Notes |
|---------|-----------|----------|----------|----------|-------|
| Base Simple | 0.3189 | 0.0000 | 0.0215 | 0.7077 | Baseline |
| Denoise MPM | 0.3224 | 0.0001 | 0.0157 | 0.7074 | Slightly better POP |
| Context Recon | 0.3185 | 0.0001 | 0.0224 | 0.7078 | Best overall |
| Denoise + Context | 0.3203 | 0.0001 | 0.0166 | 0.7074 | Balanced |

**Loss Components**:
- **POP (Panel-on-Page)**: Binary classification (is panel on page?)
- **MPM (Masked Panel Modeling)**: Auxiliary task for denoising/reconstruction
- **RPP (Reading Panel Prediction)**: Multi-class reading order prediction

### Resuming Training

To continue from a checkpoint (e.g., extend 5→11 epochs):

```powershell
python benchmarks\detections\openrouter\train_closure_lite_with_list.py \
  --json_list_file "perfect_match_training\dataspec_perfect_list.txt" \
  --image_root "E:\CalibreComics_extracted" \
  --output_dir "closure_lite_output\calibre_perfect_simple_e11" \
  --model simple --batch_size 4 --epochs 11 --num_workers 8 \
  --resume "closure_lite_output\calibre_perfect_simple\latest_checkpoint.pth"
```

---

## Quality Assurance & Analysis

### Analyzing VLM vs R-CNN Alignment

The alignment analyzer computes metrics for matching quality.

**Key Metrics**:
- **Panel Count Ratio**: `vlm_panels / rcnn_panels`
  - Perfect: ratio = 1.0
  - Near-perfect: 0.9 ≤ ratio ≤ 1.1
- **Text Coverage**: Fraction of panels with any text (dialogue/narration/SFX)
- **Quality Score**: 0-4 composite score

### Typical Quality Distribution (Calibre)

From a 336,881-page analysis:
- **Perfect matches**: 79,125 (23.5%)
- **Near-perfect**: 1,943 (0.6%)
- **High quality** (score 4): 80,753 (24.0%)
- **Medium quality** (score 3): 66,579 (19.8%)
- **Low quality** (score 0-2): 189,308 (55.6%) — exclude from training

### Duplicate Detection

Tag duplicate pages (exact MD5 or perceptual hash) for deduplication.

```powershell
python benchmarks\detections\openrouter\tag_duplicates.py \
  --input_csv "calibre_rcnn_vlm_analysis_v2.csv" \
  --use_phash
```

**Output Columns**: `dup_md5`, `dup_group_exact`, `dup_phash`, `dup_group_phash`, `dup_canonical_key`, `dup_reason`

**Policy**:
- **Training**: Keep one canonical page per duplicate group (reduce overfitting)
- **Embeddings/Search**: Keep all (users expect both single-issue and collected edition pages)

---

## Embeddings & Search

### Generating Panel/Page Embeddings

Extract embeddings from trained Closure-Lite models for semantic search.

#### Step 1: Create Zarr Skeleton (Optional, for Incremental Writes)

```powershell
python benchmarks\embeddings\create_zarr_skeleton.py \
  --json_list "perfect_match_training\dataspec_perfect_list.txt" \
  --output_dir "embeddings_calibre_skeleton" \
  --max_panels 12 \
  --embedding_dim 384
```

#### Step 2: Generate Embeddings

```powershell
python -u benchmarks\detections\openrouter\generate_embeddings_zarr.py \
  --checkpoint "closure_lite_output\calibre_perfect_simple_denoise_context\best_checkpoint.pth" \
  --calibre_json_list "perfect_match_training\dataspec_perfect_list.txt" \
  --calibre_image_root "E:\CalibreComics_extracted" \
  --output_dir "embeddings_calibre_skeleton" \
  --batch_size 8 \
  --device cpu \
  --incremental \
  --concurrency_safe \
  --overwrite
```

**Key Flags**:
- `--incremental`: Write to existing Zarr store (enables live progress monitoring)
- `--concurrency_safe`: Safe for multi-process writes
- `--debug_manifest`, `--debug_empty`: Diagnostic modes
- `--max_samples`: Limit for smoke tests

#### Step 3: Monitor Progress (During Long Runs)

```powershell
python tools\monitor_zarr_progress.py \
  --zarr_dir "embeddings_calibre_skeleton\combined_embeddings.zarr" \
  --poll_secs 30
```

#### Step 4: Postprocess Embeddings

Compute L2-normalized embeddings and build FAISS indices.

```powershell
python benchmarks\embeddings\postprocess_embeddings.py \
  --zarr_dir "embeddings_calibre_skeleton\combined_embeddings.zarr" \
  --output_dir "embeddings_calibre_skeleton" \
  --build_faiss \
  --faiss_index_dir "embeddings_calibre_skeleton\faiss"
```

**Outputs**:
- `page_embeddings_norm`, `panel_embeddings_norm`: L2-normalized arrays
- `page_id_index.json`: Index mapping (row → canonical page ID)
- FAISS indices (optional): For fast cosine-similarity search

### Zarr Dataset Structure

**Coordinates**:
- `page_id` (dim): Canonical key (normalized full manifest path)
- `manifest_path` (coord): Original manifest path
- `source` (coord): `amazon` or `calibre`
- `series`, `volume`, `issue`, `page_num` (coords): Parsed metadata
- `panel_id` (dim): 0..(max_panels-1), default 12
- `embedding_dim` (dim): 384 (for ViT-base embeddings)
- `coord_dim` (dim): 4 (x, y, w, h for bounding boxes)

**Data Variables**:
- `panel_embeddings` (page_id, panel_id, embedding_dim): Per-panel embeddings
- `page_embeddings` (page_id, embedding_dim): Attention-pooled page embeddings
- `attention_weights` (page_id, panel_id): Attention scores
- `reading_order` (page_id, panel_id): Predicted next-panel logits
- `panel_coordinates` (page_id, panel_id, coord_dim): Bounding boxes
- `text_content` (page_id, panel_id): Text snippets (fixed-width unicode)
- `panel_mask` (page_id, panel_id): Boolean mask (True = real panel)

**Compression**: Blosc (zstd), chunked as (1000, 12, 384) for efficient page-based reads

### Querying Embeddings

Use normalized embeddings for cosine-similarity searches.

```powershell
python benchmarks\detections\openrouter\query_embeddings_zarr.py \
  --zarr_dir "embeddings_calibre\combined_embeddings.zarr" \
  --query_text "superhero flying" \
  --top_k 20 \
  --use_faiss
```

---

## Command Reference

### Common Workflows (One-Liners)

#### Full Calibre Pipeline (From Scratch)

```powershell
# 1. Convert to CBZ
python benchmarks\detections\openrouter\convert_to_cbz.py --input-dir "D:\CalibreComics" --output-dir "D:\CalibreComics_converted" --preserve-structure --skip-existing

# 2. Extract images
python benchmarks\detections\openrouter\extract_calibre_comics.py --input-dir "D:\CalibreComics" --output-dir "E:\CalibreComics_extracted" --skip-existing

# 3. R-CNN detections
python benchmarks\detections_2000ad\faster_rcnn_calibre.py --input-path "E:\CalibreComics_extracted" --output-path "E:\CalibreComics\test_dections\predictions.json" --conf-threshold 0.5

# 4. VLM analysis
python benchmarks\detections\openrouter\batch_comic_analysis_multi.py --input-dir "E:\CalibreComics_extracted" --output-dir "E:\CalibreComics_analysis" --model google/gemma-3-12b-it:free --api-key sk-or-v1-... --skip-existing --max-workers 4

# 5. Alignment CSV + DataSpec
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py --coco "E:\CalibreComics\test_dections\predictions.json" --vlm_dir "E:\CalibreComics_analysis" --vlm_map "perfect_match_training\all_vlm_jsons_calibre_training_map.json" --output_csv "calibre_rcnn_vlm_analysis_v2.csv" --num_workers 8 --image_roots "E:\CalibreComics_extracted" --require_image_exists --emit_dataspec_everything --dataspec_out_dir "perfect_match_training\calibre_dataspec_all" --dataspec_list_out "perfect_match_training\dataspec_all_list.txt"

# 6. Filter to perfect matches
python benchmarks\detections\openrouter\build_dataspec_subset_list.py --dataspec_list "perfect_match_training\dataspec_all_list.txt" --subset_list "calibre_rcnn_vlm_analysis_v2_perfect_matches.txt" --out_list "perfect_match_training\dataspec_perfect_list.txt"

# 7. Train model
python benchmarks\detections\openrouter\train_closure_lite_with_list.py --json_list_file "perfect_match_training\dataspec_perfect_list.txt" --image_root "E:\CalibreComics_extracted" --output_dir "closure_lite_output\calibre_perfect_simple" --model simple --batch_size 4 --epochs 5 --num_workers 8
```

#### Quick Smoke Tests

```powershell
# Test alignment on 100 pages
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py --coco "E:\CalibreComics\test_dections\predictions.json" --vlm_dir "E:\CalibreComics_analysis" --output_csv "calibre_test_100.csv" --num_workers 1 --limit 100

# Test training on 200 pages
python benchmarks\detections\openrouter\train_closure_lite_with_list.py --json_list_file "perfect_match_training\dataspec_perfect_list.txt" --image_root "E:\CalibreComics_extracted" --output_dir "closure_lite_smoke" --model simple --batch_size 4 --epochs 1 --max_samples 200
```

---

## Troubleshooting & Tips

### Common Issues

**1. VLM Analysis Fails / Connection Errors**

*Symptoms*: JSON parse errors, connection timeouts, rate limits

*Solutions*:
- Switch to `:free` variants (e.g., `google/gemma-3-12b-it:free`)
- Reduce `--max-workers` (default 8 → 4 or 1)
- Use `--skip-existing` to resume interrupted runs
- Add `--error-log-dir` to capture failed images

**Recommended Models** (by use case):
- **Testing/Smoke**: `google/gemma-3-4b-it`
- **Production**: `google/gemma-3-12b-it:free` or `meta-llama/llama-4-scout`
- **High Quality**: `google/gemini-2.5-flash-lite` (paid, higher cost)

**2. Image Path Mismatches in Training**

*Symptoms*: "Image not found" errors, zero JSONs loaded

*Solutions*:
- Ensure `--image_root` matches the extraction directory
- On WSL: Use `/mnt/e/...` paths; dataset auto-converts `E:\...` → `/mnt/e/...`
- Verify DataSpec `page_image_path` values with:
  ```powershell
  python - <<'PY'
  import json
  with open("path/to/dataspec.json") as f:
      print(json.load(f)["page_image_path"])
  PY
  ```

**3. Alignment CSV Shows Low Match Rate**

*Symptoms*: <10% perfect matches, many "unmatched" rows

*Solutions*:
- Use `--vlm_map` (precomputed VLM→image mapping) for robust matching
- Check `--min_text_coverage` (lower to 0.5 or 0.4 if needed)
- Inspect VLM JSON quality: some models output null/empty fields
- Consider relaxing `--panel_category_names` (auto-detect may miss variants)

**4. Training Shows "Clamped/Zeroed Invalid Gradients"**

*Symptoms*: Warnings about clamped gradients during training

*Status*: Expected behavior; the framework clamps NaN/Inf gradients to zero (safety mechanism)

*Action*: No user action needed; track loss trends instead

**5. Embeddings: Missing Pages or Low Write Counts**

*Symptoms*: Zarr shows fewer pages than expected, `skipped_no_mapping` entries

*Solutions*:
- Verify training list paths are absolute and normalized (use `os.path.normpath`)
- Check `inputs_used.jsonl` and `audit_mapping.jsonl` for exact mismatch reasons
- Ensure Zarr skeleton was created from the same manifest as training list
- Use `--debug_manifest` and `--debug_empty` for diagnostics

### Performance Tips

**Windows-Specific**:
- Use `--num_workers 1` for training on Windows (avoid pickling overhead)
- Keep data on SSD (E:\ drive) vs. slower HDD (D:\)
- Add Windows Defender exclusions for `E:\CalibreComics_analysis` and `E:\CalibreComics_extracted`

**WSL-Specific**:
- Use WSL2 for GPU training (native CUDA support)
- Mount external drives at `/mnt/e/...` for data access
- Prefer `--num_workers 8` on WSL (better multiprocessing)

**Memory Management**:
- Reduce `--batch_size` if OOM (4 is safe, 8 if GPU memory allows)
- Use `--max_samples` for smoke tests (avoid full dataset loads)
- For large Zarr writes, use `--incremental` mode

**Speed Optimizations**:
- Precompute VLM map: `build_vlm_precomputed_map.py`
- Use `--fast_only` flag in alignment (skip slow heuristics)
- Parallelize VLM analysis: run multiple instances on different subsets

### Workflow Optimizations

**Batch Processing Calibre Library**:
1. Split by author/series folders and run VLM analysis in parallel
2. Combine VLM outputs and build unified map
3. Run alignment once on combined outputs

**Incremental Updates**:
- Use `--skip-existing` for VLM analysis (resume interrupted runs)
- Append new comics to existing DataSpec and retrain incrementally

**Quality Gates**:
- Filter to `score >= 4` (perfect matches only) for initial training
- Optionally include `score 3` (near-perfect) for data augmentation
- Exclude `score 0-2` (low quality) entirely

---

## Additional Resources

### Key Scripts by Category

**Data Preparation**:
- `convert_to_cbz.py`: Format conversion
- `extract_calibre_comics.py`: Image extraction
- `batch_comic_analysis_multi.py`: VLM analysis

**Panel Detection**:
- `faster_rcnn_calibre.py`: R-CNN for Calibre
- `01_bootstrap_grounding_dino_HF.py`, `02_comixs_prod_detect_HF.py`: Alternative detectors

**Alignment & Quality**:
- `create_perfect_match_filter_calibre_v2.py`: Main alignment script
- `analyze_rcnn_vlm_alignment_calibre.py`: Detailed analysis
- `annotate_page_types.py`: Page type tagging
- `tag_duplicates.py`: Duplicate detection

**DataSpec Generation**:
- `coco_to_dataspec_calibre.py`: COCO → DataSpec converter
- `build_dataspec_subset_list.py`: Subset filtering

**Training**:
- `train_closure_lite_with_list.py`: Main training script (simple model)
- `closure_lite_simple_framework.py`: Model architecture
- `closure_lite_dataset.py`: DataLoader

**Embeddings**:
- `generate_embeddings_zarr.py`: Bulk embedding extraction
- `postprocess_embeddings.py`: Normalization and FAISS indexing
- `query_embeddings_zarr.py`: Search interface

**Demo/Evaluation**:
- `demo_compare_models.py`: Multi-model comparison
- `demo_closure_lite_simple.py`: Single-model demo

### Conda Environments

- **sfmcp**: Main environment for VLM analysis and training
- **ai_scientist**: Alternative environment for training (WSL)

### External Tools

- **WandB**: Experiment tracking (`--wandb_project` flag)
- **FAISS**: Fast similarity search (optional, built during postprocess)
- **OpenRouter**: VLM API gateway (requires API key)

---

## Appendix: Annotation Format (Manga109 / FORMAT.md)

CoMix supports XML-based annotations for panels, text, characters, and balloons (Manga109 standard).

**Root**: `<annotations>`
**Book**: `<book title="...">`
- `<characters>`: List of character definitions
- `<stories>`: Story arcs
- `<pages>`: Page-level annotations

**Page**: `<page id="..." width="..." height="..." type="..." story_id="...">`
- `<panel>`: 4-point polygon (panel boundary)
- `<text>`: 4-point polygon (text region, with optional `balloon_id`)
- `<character>`: 4-point polygon (character body, with optional `character_id`)
- `<balloon>`: N-point polygon (speech balloon)
- `<face>`: Character face region
- `<onomatopoeia>`: Sound effects
- `<link_sbsc>`: 2-point polyline (speaker → text link)

**Attributes**: `id`, `points`, `character_id`, `text_id`, `balloon_id`

---

## Version Notes

**Last Updated**: 2025-10-26

**Major Changes**:
- Unified alignment workflow (`create_perfect_match_filter_calibre_v2.py`)
- DataSpec v0.3 schema with `panel_coords` and text aggregation
- Zarr-based embedding storage with incremental writes
- Page type annotation for filtering
- Duplicate detection (MD5 + pHash)
- MPM training variants (denoising, context reconstruction)

**Planned Enhancements**:
- Page-type-aware training (exclude covers/ads by default)
- OCR/VLM hybrid text validation
- Embeddings update pass (incremental Zarr appends)
- FAISS auto-loading in query service

---

*End of Documentation*
