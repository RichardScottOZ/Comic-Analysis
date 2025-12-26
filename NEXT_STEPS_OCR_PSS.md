# Summary: Ready to Run OCR + PSS Pipeline

## What We Just Built

You now have a complete serverless pipeline ready to process your 1.2M comic pages:

### 1. S3 Manifest Creator (`create_s3_manifest.py`)
- Scans S3 buckets for comic page images
- Creates manifest CSV files compatible with OCR and PSS
- Supports sampling for testing
- Works with your existing S3 structure in `calibrecomics-extracted`

### 2. Updated Cost Analysis
- **PSS on 1.2M pages**: ~$10-15 (AWS Batch GPU Spot, 2-3 hours)
- **OCR on 1.2M pages**: ~$30 (Tesseract Lambda, 30 minutes)
- **Total**: ~$42 and 3.5 hours to get complete PSS labels

### 3. Test Workflow Documentation
- Step-by-step guide for 4K page test (~$0.11, 7 minutes)
- Validates complete OCR → PSS pipeline before scaling
- Uses NeonIchiban or samples from new comics

## Current Status: Ready to Execute

### Your S3 Buckets (us-east-1: calibrecomics-extracted)
- `amazon_analysis/` - Amazon comics analysis results
- `CalibreComics_analysis/` - Calibre comics analysis results  
- `CalibreComics_extracted/` - Original Calibre extraction
- `CalibreComics_extracted_20251107/` - **NEW: Post-PSS comics** ← Need OCR + PSS
- `NeonIchiban/` - **Small test set** ← Perfect for 4K test

## Critical Insight: The Dependency Chain

You correctly identified the blocker! To run PSS, you need:

```
Comic Images → OCR (get text) → PSS Precompute (vision + text embeddings) → PSS Classify (page labels)
```

**Without OCR text for each page, PSS cannot generate text embeddings!**

## Recommended Next Steps

### Immediate: 4K Page Test (~$0.11, 7-12 minutes)

1. **Create test manifest** (30 seconds, free):
   ```bash
   python src/version1/create_s3_manifest.py \
     --bucket calibrecomics-extracted \
     --prefixes NeonIchiban/ \
     --output_csv manifests/neon_test.csv \
     --region us-east-1 \
     --sample 4000
   ```

2. **Run Tesseract OCR** (~5 min, $0.10):
   ```bash
   python src/version1/batch_ocr_processing_lithops.py \
     --manifest manifests/neon_test.csv \
     --method tesseract \
     --output-bucket calibrecomics-extracted \
     --output-prefix ocr_results/test_4k \
     --backend aws_lambda \
     --workers 500 \
     --lang eng
   ```

3. **Run PSS precompute** (~5 min, $0.01):
   ```bash
   python src/cosmo/lithops_precompute.py \
     --books-root s3://calibrecomics-extracted \
     --annotations test_books.json \
     --output-bucket calibrecomics-extracted \
     --backend aws_batch \
     --workers 20
   ```

4. **Run PSS classify** (~1 min, free):
   ```bash
   python src/cosmo/classify_precomputed.py \
     --embeddings-bucket calibrecomics-extracted \
     --model RichardScottOZ/cosmo-v4 \
     --output pss_labels/test_4k.json
   ```

### After Successful Test: Scale to Full Dataset

**Phase 1: OCR All 1.2M Pages** (~30 min, $30)
```bash
# Create full manifest
python src/version1/create_s3_manifest.py \
  --bucket calibrecomics-extracted \
  --scan-all \
  --output_csv manifests/all_1.2M.csv \
  --region us-east-1

# Run OCR
python src/version1/batch_ocr_processing_lithops.py \
  --manifest manifests/all_1.2M.csv \
  --method tesseract \
  --output-bucket calibrecomics-extracted \
  --output-prefix ocr_results \
  --backend aws_lambda \
  --workers 1000 \
  --lang eng
```

**Phase 2: PSS All 1.2M Pages** (~2-3 hours, $12)
```bash
# Precompute embeddings
python src/cosmo/lithops_precompute.py \
  --books-root s3://calibrecomics-extracted \
  --annotations all_books.json \
  --output-bucket calibrecomics-extracted \
  --output-prefix pss_embeddings \
  --backend aws_batch \
  --workers 50

# Classify pages
python src/cosmo/classify_precomputed.py \
  --embeddings-bucket calibrecomics-extracted/pss_embeddings \
  --model RichardScottOZ/cosmo-v4 \
  --output pss_labels/all_1.2M.json
```

## What You Get After PSS

PSS labels tell you what type each page is:

```json
{
  "comic_hash_abc123": {
    "pages": [
      {"index": 0, "label": "cover", "confidence": 0.98},
      {"index": 1, "label": "credits", "confidence": 0.92},
      {"index": 2, "label": "story", "confidence": 0.99},
      {"index": 3, "label": "story", "confidence": 0.99},
      ...
      {"index": 45, "label": "ads", "confidence": 0.87},
      {"index": 46, "label": "back_cover", "confidence": 0.95}
    ]
  }
}
```

### Page Types
- **cover**: Front cover
- **credits**: Credits/indicia page
- **story**: Narrative comic pages (main content)
- **ads**: Advertisement pages
- **art**: Preview art, pinups, internal covers
- **text**: Text-heavy pages (intro, author notes)
- **back_cover**: Back cover

## Unlocking Downstream Processing

With PSS labels, you can efficiently:

### 1. Stage 3 Panel Features (Only Story Pages)
Filter to only `label == "story"` pages (~70% of pages = 840K)
- Run panel detection (Fast R-CNN)
- Generate panel embeddings
- **Savings**: Skip 30% of pages that aren't story content!

### 2. OCR Strategy by Page Type
- **story pages**: Already have VLM dialogue extraction, maybe skip OCR
- **text/ads/credits**: Run high-quality OCR (already done with Tesseract!)
- **covers**: Extract titles with VLM
- **Optimization**: Different processing per page type

### 3. Query Interface with Filtering
```python
# Search only story pages
results = search_embeddings(
    query="superhero fighting villain",
    page_types=["story"]
)

# Search covers
results = search_embeddings(
    query="Superman comic cover",
    page_types=["cover"]
)
```

## Prerequisites to Run Test

### AWS Setup
1. ✅ S3 bucket access: `calibrecomics-extracted` in us-east-1
2. ⚠️ AWS credentials configured (AWS CLI)
3. ⚠️ ECR repository for Docker images
4. ⚠️ IAM permissions for Lambda and Batch

### Lithops Setup
1. ⚠️ Install: `pip install lithops[aws]`
2. ⚠️ Configure: `lithops.yaml` with AWS settings
3. ⚠️ Build runtime: `Dockerfile.lithops.ocr.cpu`
4. ⚠️ Deploy runtime: `lithops runtime deploy`

### Python Dependencies
```bash
pip install boto3 lithops tqdm transformers sentence-transformers torch
```

## Files You Now Have

### Created Files
1. `src/version1/create_s3_manifest.py` - S3 manifest creator
2. `documentation/Quick_Start_OCR_PSS_Test.md` - Step-by-step test guide
3. `PSS_COST_ANALYSIS.md` - Detailed cost analysis for 1.2M pages

### Existing Files (Already Working)
1. `src/version1/batch_ocr_processing_lithops.py` - Lithops OCR processor
2. `src/cosmo/lithops_precompute.py` - PSS embedding precompute
3. `src/cosmo/classify_precomputed.py` - PSS classification
4. `Dockerfile.lithops.ocr.cpu` - OCR runtime for Lambda
5. `Dockerfile.lithops.gpu` - GPU runtime for Batch

## Decision Point

**You asked: "Work out OCR on Lithops or something else first?"**

**Answer: Yes, run OCR first! Here's why:**

1. **OCR is required for PSS** - Can't run PSS without text for each page
2. **OCR is cheap** - $30 for 1.2M pages with Tesseract
3. **OCR is fast** - 30 minutes for entire dataset
4. **OCR is low-risk** - Simple test validates entire pipeline

**Recommendation:** Run the 4K page test NOW (~$0.11, 7 minutes) to validate the pipeline, then immediately scale to full 1.2M pages.

## Cost Comparison: Test vs Full

| Dataset | OCR Cost | PSS Cost | Total | Time |
|---------|----------|----------|-------|------|
| **4K test** | $0.10 | $0.01 | **$0.11** | 7 min |
| **100K sample** | $2.50 | $1.00 | **$3.50** | 35 min |
| **1.2M full** | $30.00 | $12.00 | **$42.00** | 3.5 hrs |

## Expected Timeline

Starting from now:

**Day 1 (Today):**
- Hour 1: Setup Lithops, configure AWS (if not done)
- Hour 2: Run 4K test, validate results
- Hour 3: If successful, start 1.2M OCR run
- Hour 3.5: OCR completes

**Day 2:**
- Start 1.2M PSS run (2-3 hours)
- PSS completes
- **Result**: All 1.2M pages have PSS labels!

**Day 3+:**
- Filter story pages (~840K)
- Run Stage 3 panel features
- Run Stage 4 sequence modeling
- Build query interface

## Summary

**Yes, do OCR on Lithops first!** It's the critical dependency for PSS, it's cheap ($30), fast (30 min), and you have everything ready to go. Start with the 4K test to validate, then scale immediately.

**Next action:** Follow `documentation/Quick_Start_OCR_PSS_Test.md` to run the test.
