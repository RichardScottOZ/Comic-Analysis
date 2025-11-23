"""CPU-based OCR implementations using traditional OCR engines."""

from typing import List, Optional
import math
from PIL import Image
import numpy as np

from .base import OCRBase, OCRResult


class TesseractOCR(OCRBase):
    """OCR implementation using Tesseract."""
    
    def __init__(self, lang='eng', config='', **kwargs):
        """Initialize Tesseract OCR.
        
        Args:
            lang: Language code (default: 'eng')
            config: Tesseract config string
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.lang = lang
        self.config_str = config
        self._engine = None
    
    def _get_engine(self):
        """Lazy load pytesseract."""
        if self._engine is None:
            try:
                import pytesseract
                self._engine = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract not installed. Install with: pip install pytesseract"
                )
        return self._engine
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            self._get_engine()
            return True
        except ImportError:
            return False
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with Tesseract."""
        pytesseract = self._get_engine()
        
        try:
            img = Image.open(image_path)
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                img, 
                lang=self.lang,
                config=self.config_str,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                conf = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
                if conf < 0:  # Tesseract returns -1 for no confidence
                    conf = 0.0
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=[float(x), float(y), float(w), float(h)],
                    metadata={'level': data['level'][i]}
                ))
            
            return results
            
        except Exception as e:
            print(f"Error processing image with Tesseract: {e}")
            return []
    
    def process_region(self, image_path: str, bbox: List[float]) -> List[OCRResult]:
        """Process a specific region with Tesseract."""
        try:
            img = Image.open(image_path)
            x, y, w, h = [int(v) for v in bbox]
            cropped = img.crop((x, y, x + w, y + h))
            
            # Save to temp and process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cropped.save(tmp.name)
                results = self.process_image(tmp.name)
                
                # Adjust coordinates back to original image
                for result in results:
                    if result.bbox:
                        result.bbox[0] += x
                        result.bbox[1] += y
                
                return results
        except Exception as e:
            print(f"Error processing region with Tesseract: {e}")
            return []


class EasyOCR(OCRBase):
    """OCR implementation using EasyOCR."""
    
    def __init__(self, languages=['en'], gpu=False, **kwargs):
        """Initialize EasyOCR.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.languages = languages
        self.gpu = gpu
        self._reader = None
    
    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            except ImportError:
                raise ImportError(
                    "easyocr not installed. Install with: pip install easyocr"
                )
        return self._reader
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with EasyOCR."""
        reader = self._get_reader()
        
        try:
            # EasyOCR returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
            detections = reader.readtext(image_path)
            
            results = []
            for detection in detections:
                polygon, text, confidence = detection
                
                # Convert polygon to bbox
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                
                results.append(OCRResult(
                    text=text,
                    confidence=float(confidence),
                    bbox=[float(x), float(y), float(w), float(h)],
                    polygon=[[float(p[0]), float(p[1])] for p in polygon]
                ))
            
            return results
            
        except Exception as e:
            print(f"Error processing image with EasyOCR: {e}")
            return []
    
    def process_region(self, image_path: str, bbox: List[float]) -> List[OCRResult]:
        """Process a specific region with EasyOCR."""
        try:
            img = Image.open(image_path)
            x, y, w, h = [int(v) for v in bbox]
            cropped = img.crop((x, y, x + w, y + h))
            
            # Convert to numpy array
            img_array = np.array(cropped)
            
            reader = self._get_reader()
            detections = reader.readtext(img_array)
            
            results = []
            for detection in detections:
                polygon, text, confidence = detection
                
                # Convert polygon to bbox and adjust coordinates
                xs = [p[0] + x for p in polygon]
                ys = [p[1] + y for p in polygon]
                bbox_x, bbox_y = min(xs), min(ys)
                bbox_w, bbox_h = max(xs) - bbox_x, max(ys) - bbox_y
                
                adjusted_polygon = [[float(p[0] + x), float(p[1] + y)] for p in polygon]
                
                results.append(OCRResult(
                    text=text,
                    confidence=float(confidence),
                    bbox=[float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)],
                    polygon=adjusted_polygon
                ))
            
            return results
            
        except Exception as e:
            print(f"Error processing region with EasyOCR: {e}")
            return []


class PaddleOCR(OCRBase):
    """OCR implementation using PaddleOCR."""
    
    def __init__(self, lang='en', use_gpu=False, use_angle_cls=True, **kwargs):
        """Initialize PaddleOCR.
        
        Args:
            lang: Language code
            use_gpu: Whether to use GPU
            use_angle_cls: Whether to use angle classification
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self._ocr = None
    
    def _get_ocr(self):
        """Lazy load PaddleOCR."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR as PaddleOCREngine
                self._ocr = PaddleOCREngine(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except ImportError:
                raise ImportError(
                    "paddleocr not installed. Install with: pip install paddleocr"
                )
        return self._ocr
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        try:
            from paddleocr import PaddleOCR as PaddleOCREngine
            return True
        except ImportError:
            return False
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with PaddleOCR."""
        ocr = self._get_ocr()
        
        try:
            # PaddleOCR returns: [[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)], ...]
            result = ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return []
            
            results = []
            for line in result[0]:
                polygon = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                
                # Convert polygon to bbox
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                
                # Calculate angle
                p0, p1 = polygon[0], polygon[1]
                angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[float(x), float(y), float(w), float(h)],
                    polygon=[[float(p[0]), float(p[1])] for p in polygon],
                    metadata={'angle': abs(angle)}
                ))
            
            return results
            
        except Exception as e:
            print(f"Error processing image with PaddleOCR: {e}")
            return []
    
    def process_region(self, image_path: str, bbox: List[float]) -> List[OCRResult]:
        """Process a specific region with PaddleOCR."""
        try:
            img = Image.open(image_path)
            x, y, w, h = [int(v) for v in bbox]
            cropped = img.crop((x, y, x + w, y + h))
            
            # Convert to numpy array
            img_array = np.array(cropped)
            
            ocr = self._get_ocr()
            result = ocr.ocr(img_array, cls=True)
            
            if not result or not result[0]:
                return []
            
            results = []
            for line in result[0]:
                polygon = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                
                # Adjust polygon coordinates
                adjusted_polygon = [[float(p[0] + x), float(p[1] + y)] for p in polygon]
                
                # Convert to bbox
                xs = [p[0] for p in adjusted_polygon]
                ys = [p[1] for p in adjusted_polygon]
                bbox_x, bbox_y = min(xs), min(ys)
                bbox_w, bbox_h = max(xs) - bbox_x, max(ys) - bbox_y
                
                # Calculate angle
                p0, p1 = polygon[0], polygon[1]
                angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)],
                    polygon=adjusted_polygon,
                    metadata={'angle': abs(angle)}
                ))
            
            return results
            
        except Exception as e:
            print(f"Error processing region with PaddleOCR: {e}")
            return []
