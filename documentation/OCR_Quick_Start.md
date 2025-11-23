# OCR Quick Start Guide

## What is OCR in Comic Analysis?

The OCR (Optical Character Recognition) module extracts literal text from comic pages. It complements the existing VLM (Vision-Language Model) analysis by focusing on:

- **Text Pages**: Pages with primarily text content
- **Advertisements**: Ad pages with product information
- **Credits**: Credits, indicia, and legal text
- **Back Matter**: Text-heavy supplementary content

While VLM analysis extracts dialogue and narrative from comic panels, OCR is designed for literal text extraction from text-heavy content.

## Installation

### Step 1: Core Dependencies (Already Installed)

The OCR module uses existing dependencies:
- NumPy
- Pillow
- Requests

### Step 2: Install OCR Engine (Choose One or More)

#### Option A: Tesseract (Recommended for beginners)

```bash
# Install Python package
pip install pytesseract

# Install Tesseract binary
# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# On macOS:
brew install tesseract

# On Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Option B: EasyOCR (Better for multiple languages)

```bash
pip install easyocr
```

#### Option C: PaddleOCR (Fast and accurate)

```bash
pip install paddleocr paddlepaddle
```

#### Option D: VLM-based OCR (Advanced, requires API key)

```bash
# No installation needed
# Just set your OpenRouter API key:
export OPENROUTER_API_KEY="your-api-key-here"
```

## Quick Usage

### Check Available Methods

```bash
cd src/version1
python batch_ocr_processing.py --list-methods
```

### Process a Single Image (Example)

```bash
cd src/version1/ocr
python example_ocr_usage.py --image /path/to/comic_page.jpg --demo cpu
```

### Batch Process Comic Pages

#### Step 1: Prepare Manifest File

Create a CSV file (e.g., `manifest.csv`) with your comic pages:

```csv
canonical_id,absolute_image_path
comic1_page_001,/path/to/comic1/page_001.jpg
comic1_page_002,/path/to/comic1/page_002.jpg
comic1_page_003,/path/to/comic1/page_003.jpg
```

#### Step 2: Run Batch OCR

```bash
cd src/version1

# Using Tesseract (fast, good for clean text)
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method tesseract \
  --max_workers 8

# Using EasyOCR (better for complex text)
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method easyocr \
  --max_workers 4

# Using Qwen VL (advanced, requires API key)
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method qwen \
  --api_key $OPENROUTER_API_KEY \
  --max_workers 2
```

### View Results

Results are saved as JSON files in the output directory:

```bash
cat ocr_results/comic1_page_001_ocr.json
```

Example output:

```json
{
  "canonical_id": "comic1_page_001",
  "source_image_path": "/path/to/comic1/page_001.jpg",
  "ocr_method": "tesseract",
  "timestamp": "2025-11-23T02:00:00",
  "text_regions": [
    {
      "text": "Extracted text from the page",
      "confidence": 0.95,
      "bbox": [100, 200, 150, 30]
    }
  ],
  "full_text": "Combined text from all regions",
  "num_regions": 10
}
```

## Choosing an OCR Method

| Method | Best For | Speed | Accuracy | GPU Support |
|--------|----------|-------|----------|-------------|
| Tesseract | Clean printed text, bulk processing | ⚡⚡⚡ | ⭐⭐⭐ | No |
| EasyOCR | Multi-language, stylized text | ⚡⚡ | ⭐⭐⭐⭐ | Yes |
| PaddleOCR | Balanced speed/accuracy, rotated text | ⚡⚡⚡ | ⭐⭐⭐⭐ | Yes |
| Qwen VL | Complex layouts, mixed content | ⚡ | ⭐⭐⭐⭐⭐ | Cloud |
| Gemma | General text extraction | ⚡ | ⭐⭐⭐⭐ | Cloud |
| Deepseek | Cost-effective VLM | ⚡ | ⭐⭐⭐⭐ | Cloud |

## Common Use Cases

### Use Case 1: Extract Text from Advertisement Pages

```bash
# Create manifest with only ad pages
cat > ads_manifest.csv << EOF
canonical_id,absolute_image_path
ad_page_001,/comics/issue1/ad_page_001.jpg
ad_page_002,/comics/issue1/ad_page_002.jpg
EOF

# Process with Tesseract
python batch_ocr_processing.py \
  --manifest_file ads_manifest.csv \
  --output_dir ad_ocr_results \
  --method tesseract
```

### Use Case 2: Extract Credits and Indicia

```bash
# Process credit pages with high accuracy
python batch_ocr_processing.py \
  --manifest_file credits_manifest.csv \
  --output_dir credits_ocr \
  --method paddleocr \
  --gpu
```

### Use Case 3: Multi-language Comics

```bash
# Use EasyOCR with multiple languages
python batch_ocr_processing.py \
  --manifest_file multilang_manifest.csv \
  --output_dir multilang_ocr \
  --method easyocr \
  --lang "en,ja,ko"
```

## Integration with Existing Workflow

### Current Workflow

```
Comics → Page Images → VLM Analysis → Embeddings
```

### Enhanced Workflow with OCR

```
Comics → Page Images → VLM Analysis (panels) → Embeddings
                    ↘ OCR Analysis (text pages) ↗
```

Both VLM and OCR results can be used together for embedding generation.

## Troubleshooting

### Problem: "No module named 'pytesseract'"

**Solution**: Install pytesseract and tesseract binary

```bash
pip install pytesseract
# On Ubuntu:
sudo apt-get install tesseract-ocr
```

### Problem: "OCR method not available"

**Solution**: Check available methods

```bash
python batch_ocr_processing.py --list-methods
```

Install the required package for your chosen method.

### Problem: Low OCR accuracy

**Solutions**:
1. Try a different OCR method (e.g., PaddleOCR or Qwen)
2. Check image quality and resolution
3. Preprocess images (deskew, denoise)
4. Adjust language settings if non-English

### Problem: API rate limits (VLM methods)

**Solution**: Reduce concurrent workers

```bash
python batch_ocr_processing.py \
  --method qwen \
  --max_workers 2  # Reduced from default
```

## Next Steps

1. **Test Different Methods**: Run example script to compare OCR methods
   ```bash
   python ocr/example_ocr_usage.py --image test.jpg --demo all
   ```

2. **Process Your Comics**: Create a manifest and run batch processing

3. **Compare with VLM**: Compare OCR results with VLM analysis for your use case

4. **Read Full Documentation**: See `documentation/OCR_Integration.md` and `src/version1/ocr/README.md`

## Additional Resources

- **Detailed Documentation**: `documentation/OCR_Integration.md`
- **Module Documentation**: `src/version1/ocr/README.md`
- **Example Script**: `src/version1/ocr/example_ocr_usage.py`
- **Batch Processing Script**: `src/version1/batch_ocr_processing.py`

## Getting Help

```bash
# View help for batch processing
python batch_ocr_processing.py --help

# View help for example script
python ocr/example_ocr_usage.py --help

# Check available methods
python batch_ocr_processing.py --list-methods
```

## Summary

The OCR module provides:
- ✅ 6 OCR methods (3 CPU-based, 3 VLM-based)
- ✅ Configurable via command-line
- ✅ Batch processing with multiprocessing
- ✅ Same manifest format as VLM analysis
- ✅ JSON output for easy integration
- ✅ Example scripts and comprehensive documentation

**Choose your OCR method and start extracting text from your comic pages today!**
