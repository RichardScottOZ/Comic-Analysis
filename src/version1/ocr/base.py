"""Base classes for OCR implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class OCRResult:
    """Result from OCR processing.
    
    Attributes:
        text: Extracted text
        confidence: Confidence score (0-1)
        bbox: Bounding box as [x, y, width, height]
        polygon: Optional polygon coordinates for rotated text
        metadata: Additional metadata (e.g., language, angle)
    """
    text: str
    confidence: float
    bbox: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None


class OCRBase(ABC):
    """Base class for all OCR implementations."""
    
    def __init__(self, **kwargs):
        """Initialize OCR processor with optional configuration."""
        self.config = kwargs
    
    @abstractmethod
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process a single image and return OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of OCRResult objects
        """
        pass
    
    @abstractmethod
    def process_region(self, image_path: str, bbox: List[float]) -> List[OCRResult]:
        """Process a specific region of an image.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box as [x, y, width, height]
            
        Returns:
            List of OCRResult objects
        """
        pass
    
    def get_full_text(self, results: List[OCRResult]) -> str:
        """Combine OCR results into a single text string.
        
        Args:
            results: List of OCRResult objects
            
        Returns:
            Combined text string
        """
        return ' '.join(r.text for r in results if r.text.strip())
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR backend is available.
        
        Returns:
            True if the backend is installed and ready
        """
        pass
