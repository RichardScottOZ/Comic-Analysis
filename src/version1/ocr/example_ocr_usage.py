#!/usr/bin/env python
"""Example usage of the OCR module.

This script demonstrates how to use different OCR methods
to extract text from comic pages.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr import create_ocr_processor, list_available_methods


def print_results(results, method_name):
    """Print OCR results in a readable format."""
    print(f"\n{'='*60}")
    print(f"Results from {method_name}")
    print(f"{'='*60}")
    
    if not results:
        print("No text detected")
        return
    
    print(f"Found {len(results)} text regions:\n")
    
    for i, result in enumerate(results, 1):
        print(f"Region {i}:")
        print(f"  Text: {result.text}")
        print(f"  Confidence: {result.confidence:.2f}")
        
        if result.bbox:
            x, y, w, h = result.bbox
            print(f"  BBox: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")
        
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        print()
    
    # Get combined text
    from ocr.base import OCRBase
    full_text = ' '.join(r.text for r in results if r.text.strip())
    print(f"Full text: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
    print()


def demo_cpu_ocr(image_path):
    """Demonstrate CPU-based OCR methods."""
    print("\n" + "="*60)
    print("CPU-based OCR Methods Demo")
    print("="*60)
    
    # Try Tesseract
    print("\n--- Tesseract OCR ---")
    try:
        ocr = create_ocr_processor('tesseract', {'lang': 'eng'})
        if ocr.is_available():
            results = ocr.process_image(image_path)
            print_results(results, "Tesseract")
        else:
            print("Tesseract not available. Install with: pip install pytesseract")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try EasyOCR
    print("\n--- EasyOCR ---")
    try:
        ocr = create_ocr_processor('easyocr', {'languages': ['en'], 'gpu': False})
        if ocr.is_available():
            print("Processing with EasyOCR (this may take a moment)...")
            results = ocr.process_image(image_path)
            print_results(results, "EasyOCR")
        else:
            print("EasyOCR not available. Install with: pip install easyocr")
    except Exception as e:
        print(f"Error: {e}")
    
    # Try PaddleOCR
    print("\n--- PaddleOCR ---")
    try:
        ocr = create_ocr_processor('paddleocr', {'lang': 'en', 'use_gpu': False})
        if ocr.is_available():
            print("Processing with PaddleOCR...")
            results = ocr.process_image(image_path)
            print_results(results, "PaddleOCR")
        else:
            print("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle")
    except Exception as e:
        print(f"Error: {e}")


def demo_vlm_ocr(image_path, api_key):
    """Demonstrate VLM-based OCR methods."""
    print("\n" + "="*60)
    print("VLM-based OCR Methods Demo")
    print("="*60)
    
    if not api_key:
        print("\nVLM OCR requires an OpenRouter API key.")
        print("Set the OPENROUTER_API_KEY environment variable or pass --api-key")
        return
    
    # Try Qwen
    print("\n--- Qwen OCR ---")
    try:
        ocr = create_ocr_processor('qwen', {'api_key': api_key})
        if ocr.is_available():
            print("Processing with Qwen VL (this may take a moment)...")
            results = ocr.process_image(image_path)
            print_results(results, "Qwen")
        else:
            print("Qwen not available. Check API key.")
    except Exception as e:
        print(f"Error: {e}")


def demo_region_ocr(image_path):
    """Demonstrate region-specific OCR."""
    print("\n" + "="*60)
    print("Region-specific OCR Demo")
    print("="*60)
    
    # Define a region (x, y, width, height)
    # This is just an example - adjust based on your image
    region = [100, 100, 400, 200]  # Top-left quadrant
    
    print(f"\nProcessing region: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
    
    try:
        ocr = create_ocr_processor('tesseract', {'lang': 'eng'})
        if ocr.is_available():
            results = ocr.process_region(image_path, region)
            print_results(results, "Tesseract (Region)")
        else:
            print("Tesseract not available")
    except Exception as e:
        print(f"Error: {e}")


def check_available_methods():
    """Check and display available OCR methods."""
    print("\n" + "="*60)
    print("Checking Available OCR Methods")
    print("="*60 + "\n")
    
    methods = list_available_methods()
    
    print("Method Status:")
    print("-" * 40)
    for method, available in methods.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {method:12} - {status}")
    
    print("\nInstallation Instructions:")
    print("-" * 40)
    print("CPU Methods:")
    print("  pip install pytesseract easyocr paddleocr paddlepaddle")
    print("\nVLM Methods:")
    print("  Set OPENROUTER_API_KEY environment variable")
    print()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OCR Module Example Usage',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file to process'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.environ.get('OPENROUTER_API_KEY'),
        help='OpenRouter API key for VLM methods'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['cpu', 'vlm', 'region', 'all'],
        default='all',
        help='Which demo to run (default: all)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check available OCR methods and exit'
    )
    
    args = parser.parse_args()
    
    # Check available methods
    if args.check:
        check_available_methods()
        return
    
    # Validate image path
    if not args.image:
        print("Error: --image argument is required (unless using --check)")
        print("\nExample usage:")
        print("  python example_ocr_usage.py --image /path/to/comic_page.jpg")
        print("  python example_ocr_usage.py --check")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"Processing image: {args.image}")
    
    # Run demos
    if args.demo in ['cpu', 'all']:
        demo_cpu_ocr(args.image)
    
    if args.demo in ['vlm', 'all']:
        demo_vlm_ocr(args.image, args.api_key)
    
    if args.demo in ['region', 'all']:
        demo_region_ocr(args.image)
    
    print("\n" + "="*60)
    print("Demo Complete")
    print("="*60)
    print("\nFor batch processing, use: batch_ocr_processing.py")
    print("For more information, see: ocr/README.md")


if __name__ == "__main__":
    main()
