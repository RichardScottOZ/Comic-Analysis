# OCR Integration for Comic Analysis

## Overview

The OCR (Optical Character Recognition) module provides text extraction capabilities that complement the existing VLM (Vision-Language Model) analysis pipeline. While VLM analysis focuses on extracting dialogue, captions, and narrative from comic panels, OCR is designed for literal text extraction from text-heavy pages like advertisements, credits, title pages, and back matter.

## Purpose

The OCR functionality was added to address a specific gap identified in the workflow:

> "my first vlm run was focused on getting dialogue and captions here but not the literal text for text and ad pages"

This module enables extraction of:
- Text from advertisement pages
- Credits and indicia
- Title pages and headers
- Legal text and disclaimers
- Any text-heavy content where VLM dialogue extraction is not suitable

## Architecture

The OCR module is designed with a plugin architecture that supports multiple OCR backends:

```
src/version1/ocr/
├── __init__.py           # Module exports
├── base.py               # Base classes and interfaces
├── cpu_ocr.py            # Traditional OCR implementations
├── vlm_ocr.py            # VLM-based OCR implementations
├── factory.py            # Factory for creating OCR processors
├── README.md             # Detailed documentation
└── example_ocr_usage.py  # Example usage
```

## Supported OCR Methods

### CPU-based Methods (Traditional OCR)

1. **Tesseract OCR**
   - Industry standard, free and open-source
   - Good for printed text
   - Fast processing
   - Requires: `pytesseract` and tesseract binary

2. **EasyOCR**
   - Deep learning-based
   - Supports 80+ languages
   - GPU acceleration available
   - Requires: `easyocr`

3. **PaddleOCR**
   - Fast and accurate
   - Supports rotation/angle detection
   - GPU acceleration available
   - Requires: `paddleocr`, `paddlepaddle`

### VLM-based Methods (Advanced OCR)

4. **Qwen OCR**
   - Uses Qwen 2.5 VL model
   - State-of-the-art vision understanding
   - Handles complex layouts
   - Requires: OpenRouter API key

5. **Gemma OCR**
   - Uses Google Gemma model
   - Good performance
   - Cost-effective options available
   - Requires: OpenRouter API key

6. **Deepseek OCR**
   - Uses Deepseek model
   - Cost-effective
   - Good for general text extraction
   - Requires: OpenRouter API key

## Installation

### Core Dependencies

The OCR module uses the existing comic analysis dependencies. No additional core dependencies are required.

### Optional OCR Engines

Install OCR engines as needed:

```bash
# Tesseract (requires system binary)
pip install pytesseract
# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# On macOS:
brew install tesseract

# EasyOCR
pip install easyocr

# PaddleOCR
pip install paddleocr paddlepaddle  # CPU version
# OR
pip install paddleocr paddlepaddle-gpu  # GPU version
```

### API Keys for VLM Methods

For VLM-based OCR, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Quick Start

Check available OCR methods:

```bash
cd src/version1
python batch_ocr_processing.py --list-methods
```

### Batch Processing

Process a manifest file with OCR:

```bash
# Using Tesseract (CPU, fast)
python batch_ocr_processing.py \
  --manifest_file path/to/manifest.csv \
  --output_dir ocr_results \
  --method tesseract \
  --max_workers 8

# Using EasyOCR with GPU
python batch_ocr_processing.py \
  --manifest_file path/to/manifest.csv \
  --output_dir ocr_results \
  --method easyocr \
  --gpu \
  --max_workers 4

# Using Qwen VL (advanced)
python batch_ocr_processing.py \
  --manifest_file path/to/manifest.csv \
  --output_dir ocr_results \
  --method qwen \
  --api_key $OPENROUTER_API_KEY \
  --max_workers 2
```

### Programmatic Usage

```python
from ocr import create_ocr_processor

# Create an OCR processor
ocr = create_ocr_processor('tesseract', {'lang': 'eng'})

# Process an image
results = ocr.process_image('path/to/image.jpg')

# Extract full text
full_text = ocr.get_full_text(results)

# Access individual results
for result in results:
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
    if result.bbox:
        print(f"BBox: {result.bbox}")
```

### Example Scripts

Run the example script to test different OCR methods:

```bash
cd src/version1/ocr
python example_ocr_usage.py --image path/to/image.jpg --demo all
```

## Integration with Existing Pipeline

The OCR module integrates seamlessly with the existing comic analysis pipeline:

### Current Workflow

1. **Comic Processing**: Convert comics to individual page images
2. **VLM Analysis** (`batch_comic_analysis_multi.py`): Extract dialogue, captions, and narrative
3. **Panel Detection**: Fast-RCNN detects panel bounding boxes
4. **Embedding Generation**: Generate queryable embeddings

### Enhanced Workflow with OCR

1. **Comic Processing**: Convert comics to individual page images
2. **VLM Analysis**: Extract dialogue and narrative (for story pages)
3. **OCR Analysis** (`batch_ocr_processing.py`): Extract literal text (for text pages, ads, credits)
4. **Panel Detection**: Fast-RCNN detects panel bounding boxes
5. **Embedding Generation**: Generate queryable embeddings from both VLM and OCR data

### Manifest Format

Both VLM and OCR batch scripts use the same manifest CSV format:

```csv
canonical_id,absolute_image_path
comic1_page_001,/path/to/comic1/page_001.jpg
comic1_page_002,/path/to/comic1/page_002.jpg
```

## Output Format

OCR results are saved as JSON files:

```json
{
  "canonical_id": "comic_page_001",
  "source_image_path": "/path/to/image.jpg",
  "ocr_method": "tesseract",
  "timestamp": "2025-11-23T02:00:00",
  "text_regions": [
    {
      "text": "Extracted text",
      "confidence": 0.95,
      "bbox": [100, 200, 150, 30],
      "polygon": [[100, 200], [250, 200], [250, 230], [100, 230]],
      "metadata": {"angle": 2.5}
    }
  ],
  "full_text": "Combined text from all regions",
  "num_regions": 10
}
```

## Configuration

### OCR Method Selection

Choose an OCR method based on your requirements:

- **Fast bulk processing**: Tesseract
- **Multi-language support**: EasyOCR or PaddleOCR
- **Complex layouts**: Qwen or Deepseek VLM
- **Best accuracy**: PaddleOCR or Qwen VLM

### Performance Tuning

#### CPU Methods
- **Tesseract**: Use multiple workers (8-16) for parallel processing
- **EasyOCR**: 2-4 workers (model loading overhead)
- **PaddleOCR**: 4-8 workers (good balance)

#### VLM Methods
- **Rate Limiting**: Use 1-2 workers to avoid API rate limits
- **Cost Control**: Monitor API usage
- **Timeout**: Increase for large/complex images (default: 120s)

### Configuration Examples

```python
# Tesseract with custom config
config = {
    'lang': 'eng',
    'config': '--psm 6'  # Assume uniform block of text
}

# EasyOCR with GPU and multiple languages
config = {
    'languages': ['en', 'es', 'fr'],
    'gpu': True
}

# PaddleOCR with angle detection
config = {
    'lang': 'en',
    'use_gpu': True,
    'use_angle_cls': True
}

# Qwen with specific model
config = {
    'api_key': 'your-key',
    'model': 'qwen/qwen-2-vl-72b-instruct',
    'timeout': 180
}
```

## Testing Different OCR Methods

The OCR module is designed to be configurable for testing different methods:

```bash
# Test Tesseract
python batch_ocr_processing.py --manifest test_manifest.csv --output_dir test_ocr --method tesseract

# Test EasyOCR
python batch_ocr_processing.py --manifest test_manifest.csv --output_dir test_ocr --method easyocr

# Test PaddleOCR
python batch_ocr_processing.py --manifest test_manifest.csv --output_dir test_ocr --method paddleocr

# Test Qwen
python batch_ocr_processing.py --manifest test_manifest.csv --output_dir test_ocr --method qwen
```

Compare results to determine the best method for your specific comic dataset.

## Best Practices

1. **Method Selection**: Test multiple methods on a sample set to determine the best for your content
2. **Parallel Processing**: Use appropriate worker counts based on the OCR method
3. **Skip Existing**: Use output directory checking to avoid reprocessing
4. **Error Handling**: Review failed images and adjust parameters
5. **Cost Management**: For VLM methods, monitor API usage and costs

## Troubleshooting

### Common Issues

**ImportError: No module named 'pytesseract'**
- Solution: `pip install pytesseract` and install tesseract binary

**GPU not found for EasyOCR/PaddleOCR**
- Solution: Install GPU version or use `--gpu False`

**API rate limit errors**
- Solution: Reduce `--max_workers` to 1-2 for VLM methods

**Low OCR accuracy**
- Solution: Try different methods, adjust language settings, or preprocess images

### Getting Help

- See detailed documentation: `src/version1/ocr/README.md`
- Run example script: `python example_ocr_usage.py --help`
- Check available methods: `python batch_ocr_processing.py --list-methods`

## Future Enhancements

Potential improvements to the OCR module:

1. **Panel-level OCR**: Integrate with existing panel detection for per-panel text extraction
2. **Text Classification**: Classify text regions (dialogue, narration, SFX, captions)
3. **Quality Assessment**: Add OCR quality metrics and confidence thresholds
4. **Additional Backends**: Support for Google Cloud Vision, AWS Textract
5. **Language Detection**: Automatic language detection for multi-language comics
6. **Post-processing**: Spell checking and text cleanup

## References

- [CoSMo Paper](https://github.com/mserra0/CoSMo-ComicsPSS) - Uses Qwen 2.5 VL for OCR
- [Deepseek OCR Discussion](https://news.ycombinator.com/item?id=43549072)
- [Qwen Visual Grounding](https://pyimagesearch.com/2025/06/09/object-detection-and-visual-grounding-with-qwen-2-5/)
