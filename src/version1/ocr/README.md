# OCR Module for Comic Analysis

This module provides various OCR (Optical Character Recognition) methods for extracting text from comic pages. It complements the VLM (Vision-Language Model) analysis by focusing on literal text extraction from text pages, advertisements, and other text-heavy content.

## Overview

The OCR module supports two categories of OCR methods:

1. **CPU-based Traditional OCR**: Fast, offline methods suitable for bulk processing
2. **VLM-based OCR**: Cloud-based vision-language models with advanced understanding

## Supported OCR Methods

### CPU-based Methods

#### Tesseract OCR
- **Engine**: Tesseract OCR
- **Requirements**: `pytesseract`, tesseract binary
- **Pros**: Free, widely used, good for printed text
- **Cons**: Less accurate on stylized/handwritten text
- **Installation**: 
  ```bash
  pip install pytesseract
  # On Ubuntu/Debian:
  sudo apt-get install tesseract-ocr
  # On macOS:
  brew install tesseract
  ```

#### EasyOCR
- **Engine**: EasyOCR
- **Requirements**: `easyocr`
- **Pros**: Supports 80+ languages, GPU acceleration, good accuracy
- **Cons**: Slower than Tesseract, larger model size
- **Installation**:
  ```bash
  pip install easyocr
  ```

#### PaddleOCR
- **Engine**: PaddleOCR
- **Requirements**: `paddleocr`, `paddlepaddle`
- **Pros**: Fast, good accuracy, supports rotation/angle detection
- **Cons**: Additional dependency on PaddlePaddle
- **Installation**:
  ```bash
  pip install paddleocr
  # CPU version:
  pip install paddlepaddle
  # GPU version (CUDA 11.2):
  pip install paddlepaddle-gpu
  ```

### VLM-based Methods

#### Qwen OCR
- **Model**: Qwen 2.5 VL
- **Requirements**: OpenRouter API key
- **Pros**: State-of-the-art vision understanding, handles complex layouts
- **Cons**: API costs, requires internet
- **Default Model**: `qwen/qwen-2-vl-72b-instruct`

#### Gemma OCR
- **Model**: Gemma 2
- **Requirements**: OpenRouter API key
- **Pros**: Good performance, Google-backed
- **Cons**: Limited vision capabilities on some models, API costs
- **Default Model**: `google/gemma-2-9b-it:free`

#### Deepseek OCR
- **Model**: Deepseek
- **Requirements**: OpenRouter API key
- **Pros**: Cost-effective, good performance
- **Cons**: API costs, requires internet
- **Default Model**: `deepseek/deepseek-chat`

## Usage

### Basic Usage

```python
from ocr import create_ocr_processor

# Create a Tesseract OCR processor
ocr = create_ocr_processor('tesseract', {'lang': 'eng'})

# Process an image
results = ocr.process_image('path/to/image.jpg')

# Extract full text
full_text = ocr.get_full_text(results)

# Access individual results
for result in results:
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
    print(f"BBox: {result.bbox}")
```

### Using Different Methods

```python
# EasyOCR with GPU
ocr = create_ocr_processor('easyocr', {'languages': ['en'], 'gpu': True})

# PaddleOCR with angle classification
ocr = create_ocr_processor('paddleocr', {'lang': 'en', 'use_gpu': True, 'use_angle_cls': True})

# Qwen VL OCR
ocr = create_ocr_processor('qwen', {'api_key': 'your-api-key'})
```

### Batch Processing

Use the provided batch processing script:

```bash
# Check available methods
python batch_ocr_processing.py --list-methods

# Process with Tesseract
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method tesseract \
  --max_workers 8

# Process with EasyOCR on GPU
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method easyocr \
  --gpu \
  --max_workers 4

# Process with Qwen VL
python batch_ocr_processing.py \
  --manifest_file manifest.csv \
  --output_dir ocr_results \
  --method qwen \
  --api_key YOUR_OPENROUTER_KEY \
  --max_workers 2
```

## Output Format

The OCR results are saved as JSON files with the following structure:

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
      "metadata": {
        "angle": 2.5,
        "location": "top-left"
      }
    }
  ],
  "full_text": "Combined text from all regions",
  "num_regions": 10
}
```

## Configuration

### CPU OCR Configuration

```python
config = {
    'lang': 'en',          # Language code
    'gpu': False,          # Use GPU (EasyOCR/PaddleOCR)
    'use_angle_cls': True  # Angle detection (PaddleOCR)
}
```

### VLM OCR Configuration

```python
config = {
    'api_key': 'your-key',           # OpenRouter API key
    'model': 'qwen/qwen-2-vl-72b',  # Model override
    'timeout': 120                   # Request timeout
}
```

## Performance Considerations

### CPU Methods
- **Tesseract**: Fastest, best for clean printed text
- **EasyOCR**: Slower but more accurate, especially for non-Latin scripts
- **PaddleOCR**: Good balance of speed and accuracy

### VLM Methods
- **Rate Limiting**: Limit concurrent workers (2-4) to avoid API rate limits
- **Cost**: VLM methods incur API costs per image
- **Accuracy**: Better for complex layouts and mixed content

## Integration with Existing Pipeline

The OCR module is designed to complement the existing VLM analysis:

1. **VLM Analysis** (`batch_comic_analysis_multi.py`): Extracts dialogue, captions, and narrative from comic panels
2. **OCR Analysis** (`batch_ocr_processing.py`): Extracts literal text from text pages, ads, credits, etc.

Both can use the same manifest file and run independently or sequentially.

## Troubleshooting

### Import Errors
- Ensure the OCR package is installed: `pip install pytesseract easyocr paddleocr`
- For Tesseract, install the binary: `apt-get install tesseract-ocr` or `brew install tesseract`

### GPU Issues
- Check CUDA compatibility for EasyOCR and PaddleOCR
- Verify GPU drivers: `nvidia-smi`
- Use CPU fallback if GPU fails: `--gpu` flag removed

### API Errors
- Verify API key is set: `export OPENROUTER_API_KEY=your-key`
- Check API quotas and rate limits
- Reduce concurrent workers: `--max_workers 2`

## Future Enhancements

- Support for additional OCR engines (Google Cloud Vision, AWS Textract)
- Panel-level OCR integration with existing detection pipeline
- Text region classification (dialogue, narration, SFX, captions)
- OCR quality assessment and confidence thresholds
- Multi-language support and language detection
