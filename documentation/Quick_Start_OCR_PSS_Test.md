# Quick Start: OCR + PSS Test Run (4,000 Pages)

This guide walks through running a complete test of the OCR → PSS pipeline on 4,000 pages from your NeonIchiban or new comics collection.

**Total estimated cost: ~$0.11 (10-11 cents)**
**Total estimated time: ~7 minutes**

## Prerequisites

1. AWS CLI configured with credentials for `calibrecomics-extracted` bucket access
2. Python 3.8+ with required packages:
   ```bash
   pip install boto3 lithops tqdm
   ```
3. Lithops configured for AWS Lambda (see Configuration section below)

## Step 1: Create Test Manifest from S3

### Option A: NeonIchiban (Small, Complete Collection)

```bash
cd src/version1

python create_s3_manifest.py \
  --bucket calibrecomics-extracted \
  --prefixes NeonIchiban/ \
  --output_csv ../../manifests/neon_test.csv \
  --region us-east-1
```

This will scan all images in the NeonIchiban directory. If it's more than 4K images, use `--sample 4000`.

### Option B: Sample from New Comics (CalibreComics_extracted_20251107)

```bash
python create_s3_manifest.py \
  --bucket calibrecomics-extracted \
  --prefixes CalibreComics_extracted_20251107/ \
  --output_csv ../../manifests/test_4k.csv \
  --sample 4000 \
  --region us-east-1
```

This will sample 4,000 random pages from your new comics collection.

**Expected output:**
```
--- Starting S3 Manifest Creation ---
Bucket: s3://calibrecomics-extracted
Region: us-east-1
Prefixes: 1

  Scanning s3://calibrecomics-extracted/CalibreComics_extracted_20251107/...
  ✓ Found 4000 images

📊 Total images found: 4000

--- Writing manifest CSV ---
Writing manifest: 100%|████████████| 4000/4000

✅ Manifest creation complete!
   Output: ../../manifests/test_4k.csv
   Total entries: 4,000
   Total size: 1.23 GB
```

## Step 2: Configure Lithops for AWS Lambda

Create or update `lithops.yaml` in the repository root:

```yaml
lithops:
    backend: aws_lambda
    storage: aws_s3

aws:
    region: us-east-1
    
aws_lambda:
    runtime: comic-ocr-tesseract-runtime
    runtime_memory: 1024  # 1GB for Tesseract
    runtime_timeout: 300  # 5 minutes max per page
    max_workers: 500
    
aws_s3:
    storage_bucket: calibrecomics-extracted
    region: us-east-1
```

## Step 3: Build and Deploy OCR Runtime

Build the Tesseract OCR Docker image:

```bash
# From repository root
docker build -f Dockerfile.lithops.ocr.cpu -t comic-ocr-tesseract .
```

Tag and push to ECR (Amazon Elastic Container Registry):

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository (if doesn't exist)
aws ecr create-repository --repository-name comic-ocr-tesseract --region us-east-1

# Tag image
docker tag comic-ocr-tesseract:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/comic-ocr-tesseract:latest

# Push to ECR
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/comic-ocr-tesseract:latest
```

Deploy to Lambda:

```bash
lithops runtime deploy comic-ocr-tesseract-runtime -b aws_lambda
```

**Note:** You only need to do this once. The runtime will be reused for all future OCR runs.

## Step 4: Run Tesseract OCR on Test Set

```bash
cd src/version1

python batch_ocr_processing_lithops.py \
  --manifest ../../manifests/test_4k.csv \
  --method tesseract \
  --output-bucket calibrecomics-extracted \
  --output-prefix ocr_results/test_4k \
  --backend aws_lambda \
  --workers 500 \
  --lang eng
```

**Expected output:**
```
Loading manifest from: ../../manifests/test_4k.csv
Loaded 4000 images from manifest

Starting Lithops OCR processing:
  Backend: aws_lambda
  OCR Method: tesseract
  Workers: 500
  Images: 4,000
  Output: s3://calibrecomics-extracted/ocr_results/test_4k/

Submitting tasks to Lithops...
Processing: 100%|████████████| 4000/4000 [00:03<00:00, 1250.00it/s]

Processing complete!
  Successful: 3,998/4,000
  Failed: 2
  Total pages: 4,000
  Processing time: 5.2 minutes
  Cost estimate: ~$0.10
```

**Results location:** `s3://calibrecomics-extracted/ocr_results/test_4k/`

Each page will have a JSON file:
```
ocr_results/test_4k/
  NeonIchiban/comic1/page_001_ocr.json
  NeonIchiban/comic1/page_002_ocr.json
  ...
```

**JSON format:**
```json
{
  "OCRResult": {
    "full_text": "THE AMAZING ADVENTURES...",
    "text_regions": [
      {
        "text": "THE AMAZING",
        "confidence": 0.95,
        "bbox": [10, 20, 200, 50]
      }
    ]
  },
  "metadata": {
    "canonical_id": "NeonIchiban/comic1/page_001",
    "ocr_method": "tesseract",
    "timestamp": "2025-12-26T23:30:00Z"
  }
}
```

## Step 5: Verify OCR Output Format

Download a few sample OCR results to verify format:

```bash
# Download sample
aws s3 cp s3://calibrecomics-extracted/ocr_results/test_4k/NeonIchiban/comic1/page_001_ocr.json sample_ocr.json

# Check content
cat sample_ocr.json | python -m json.tool
```

Verify the JSON has:
- ✅ `OCRResult` key with text data
- ✅ Matches format expected by PSS precompute script

## Step 6: Run PSS Precompute (Generate Embeddings)

**Note:** This step requires GPU and is more expensive. For the test, we'll run it locally or with AWS Batch.

### Option A: Local GPU (If Available)

```bash
cd src/cosmo

python precompute_embeddings.py \
  --books-root s3://calibrecomics-extracted \
  --annotations test_4k_books.json \
  --output-dir embeddings/test_4k
```

### Option B: AWS Batch with GPU (Recommended)

Configure `lithops.yaml` for Batch:

```yaml
aws_batch:
    runtime: cosmo-pss-gpu
    runtime_memory: 8192
    runtime_timeout: 900
    compute_backend: SPOT
    instance_types: ['g4dn.xlarge']
    max_workers: 20
```

Run precompute:

```bash
python lithops_precompute.py \
  --books-root s3://calibrecomics-extracted \
  --annotations test_4k_books.json \
  --output-bucket calibrecomics-extracted \
  --output-prefix pss_embeddings/test_4k \
  --backend aws_batch \
  --workers 20
```

**Cost estimate:** ~$0.01 for 4K pages (~5 minutes on 20 GPU instances)

## Step 7: Run PSS Classification

```bash
cd src/cosmo

python classify_precomputed.py \
  --embeddings-dir embeddings/test_4k \
  --model RichardScottOZ/cosmo-v4 \
  --output-dir pss_labels/test_4k
```

**Output:** JSON file with page type labels (cover, credits, story, ads, etc.)

```json
{
  "NeonIchiban/comic1": {
    "pages": [
      {"index": 0, "label": "cover", "confidence": 0.98},
      {"index": 1, "label": "credits", "confidence": 0.92},
      {"index": 2, "label": "story", "confidence": 0.99},
      {"index": 3, "label": "story", "confidence": 0.97},
      {"index": 4, "label": "ads", "confidence": 0.89}
    ]
  }
}
```

## Validation Checklist

After completing the test run, verify:

- [ ] Manifest CSV created successfully from S3
- [ ] OCR JSON files generated for all pages
- [ ] OCR JSON format matches PSS expectations (`OCRResult` key)
- [ ] PSS embeddings generated without errors
- [ ] PSS labels look reasonable (covers at start, story pages in middle, etc.)
- [ ] No excessive errors or failures

## Cost Breakdown (4,000 Pages)

| Step | Service | Cost | Time |
|------|---------|------|------|
| Manifest creation | Local | $0.00 | 30 sec |
| Tesseract OCR | Lambda | **$0.10** | 5 min |
| PSS precompute | Batch GPU Spot | **$0.01** | 5 min |
| PSS classify | Lambda/Local | $0.00 | 1 min |
| **Total** | | **$0.11** | **~12 min** |

## Troubleshooting

### Error: "Runtime not found"
**Solution:** Deploy the OCR runtime first (see Step 3)

### Error: "Access denied to S3"
**Solution:** Check AWS credentials and IAM permissions for the bucket

### Error: "OCRResult key not found"
**Solution:** Check OCR output format in Step 5, may need to adjust PSS precompute script

### Error: "Lambda timeout"
**Solution:** Increase `runtime_timeout` in lithops.yaml to 600 (10 minutes)

## Next Steps After Successful Test

Once the 4K page test completes successfully:

1. **Medium run:** 100K pages (~$2.50, 30 minutes)
2. **Full run:** 1.2M pages (~$42, 3 hours)
3. **Run Stage 3+4:** Generate panel and sequence embeddings
4. **Build query interface:** Create searchable database

## Files Created

```
manifests/
  test_4k.csv                          # Manifest of 4K pages
  
s3://calibrecomics-extracted/
  ocr_results/test_4k/                 # OCR JSON files
    NeonIchiban/.../page_*.json
  pss_embeddings/test_4k/              # PSS embeddings
    visual/*.pt
    text/*.pt
  pss_labels/test_4k/                  # PSS classifications
    labels.json
```

## Support

If you encounter issues:
1. Check the Lithops logs: `lithops logs`
2. Verify S3 permissions
3. Confirm Docker runtime is deployed
4. Check CloudWatch logs for Lambda errors

---

**Ready to scale up?** Once this test succeeds, you can process your full 1.2M page dataset with confidence!
