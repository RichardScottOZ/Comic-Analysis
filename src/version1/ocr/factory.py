"""Factory for creating OCR processors."""

from typing import Optional, Dict, Any

from .base import OCRBase
from .cpu_ocr import TesseractOCR, EasyOCR, PaddleOCR
from .vlm_ocr import QwenOCR, GemmaOCR, DeepseekOCR


def create_ocr_processor(
    method: str,
    config: Optional[Dict[str, Any]] = None
) -> OCRBase:
    """Create an OCR processor instance.
    
    Args:
        method: OCR method name. Options:
            - 'tesseract': Tesseract OCR (CPU)
            - 'easyocr': EasyOCR (CPU/GPU)
            - 'paddleocr': PaddleOCR (CPU/GPU)
            - 'qwen': Qwen VL via OpenRouter
            - 'gemma': Gemma via OpenRouter
            - 'deepseek': Deepseek via OpenRouter
        config: Configuration dictionary for the OCR processor
        
    Returns:
        OCRBase instance
        
    Raises:
        ValueError: If method is not recognized
        
    Examples:
        >>> # Create Tesseract OCR
        >>> ocr = create_ocr_processor('tesseract', {'lang': 'eng'})
        
        >>> # Create EasyOCR with GPU
        >>> ocr = create_ocr_processor('easyocr', {'gpu': True})
        
        >>> # Create Qwen OCR
        >>> ocr = create_ocr_processor('qwen', {'api_key': 'your-key'})
    """
    if config is None:
        config = {}
    
    method = method.lower()
    
    if method == 'tesseract':
        return TesseractOCR(**config)
    elif method == 'easyocr':
        return EasyOCR(**config)
    elif method == 'paddleocr':
        return PaddleOCR(**config)
    elif method == 'qwen':
        return QwenOCR(**config)
    elif method == 'gemma':
        return GemmaOCR(**config)
    elif method == 'deepseek':
        return DeepseekOCR(**config)
    else:
        raise ValueError(
            f"Unknown OCR method: {method}. "
            f"Available methods: tesseract, easyocr, paddleocr, qwen, gemma, deepseek"
        )


def list_available_methods() -> Dict[str, bool]:
    """Check which OCR methods are available.
    
    Returns:
        Dictionary mapping method names to availability status
    """
    methods = {
        'tesseract': False,
        'easyocr': False,
        'paddleocr': False,
        'qwen': False,
        'gemma': False,
        'deepseek': False
    }
    
    # Check CPU methods
    try:
        ocr = TesseractOCR()
        methods['tesseract'] = ocr.is_available()
    except Exception:
        pass
    
    try:
        ocr = EasyOCR()
        methods['easyocr'] = ocr.is_available()
    except Exception:
        pass
    
    try:
        ocr = PaddleOCR()
        methods['paddleocr'] = ocr.is_available()
    except Exception:
        pass
    
    # Check VLM methods (they're available if API key is set)
    try:
        ocr = QwenOCR()
        methods['qwen'] = ocr.is_available()
    except Exception:
        pass
    
    try:
        ocr = GemmaOCR()
        methods['gemma'] = ocr.is_available()
    except Exception:
        pass
    
    try:
        ocr = DeepseekOCR()
        methods['deepseek'] = ocr.is_available()
    except Exception:
        pass
    
    return methods
