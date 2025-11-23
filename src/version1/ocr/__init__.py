"""OCR module for comic text extraction.

This module provides various OCR methods for extracting text from comic pages,
including both CPU-based traditional OCR and VLM-based OCR approaches.
"""

from .base import OCRBase, OCRResult
from .cpu_ocr import TesseractOCR, EasyOCR, PaddleOCR
from .vlm_ocr import QwenOCR, GemmaOCR, DeepseekOCR
from .factory import create_ocr_processor, list_available_methods

__all__ = [
    'OCRBase',
    'OCRResult',
    'TesseractOCR',
    'EasyOCR',
    'PaddleOCR',
    'QwenOCR',
    'GemmaOCR',
    'DeepseekOCR',
    'create_ocr_processor',
    'list_available_methods'
]
