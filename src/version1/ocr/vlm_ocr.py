"""VLM-based OCR implementations using vision-language models."""

from typing import List, Optional
import base64
import json
import requests
import os

from .base import OCRBase, OCRResult


class VLMOCRBase(OCRBase):
    """Base class for VLM-based OCR implementations."""
    
    def __init__(self, api_key=None, model=None, timeout=120, **kwargs):
        """Initialize VLM OCR.
        
        Args:
            api_key: API key for the service
            model: Model name to use
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URI."""
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    
    def _create_ocr_prompt(self) -> str:
        """Create prompt for OCR extraction."""
        return """Extract all text from this image. Return the text in JSON format with the following structure:
{
  "text_regions": [
    {
      "text": "extracted text",
      "location": "description of where the text is (e.g., 'top-left', 'center', 'bottom-right')",
      "confidence": 0.95
    }
  ]
}

Include all visible text including:
- Main text content
- Captions
- Labels
- Headers and footers
- Any other readable text

Return ONLY valid JSON."""
    
    def _parse_ocr_response(self, response_text: str) -> List[OCRResult]:
        """Parse OCR response into OCRResult objects.
        
        Args:
            response_text: Response text from the API
            
        Returns:
            List of OCRResult objects
        """
        try:
            # Try to extract JSON from response
            json_text = response_text.strip()
            
            # Handle markdown code blocks
            if json_text.startswith('```'):
                json_text = json_text.split('```')[1]
                if json_text.startswith('json'):
                    json_text = json_text[4:]
                json_text = json_text.strip()
            
            data = json.loads(json_text)
            
            results = []
            text_regions = data.get('text_regions', [])
            
            for i, region in enumerate(text_regions):
                text = region.get('text', '')
                if not text.strip():
                    continue
                
                confidence = float(region.get('confidence', 0.9))
                location = region.get('location', 'unknown')
                
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=None,  # VLMs typically don't provide exact bounding boxes
                    metadata={'location': location, 'order': i}
                ))
            
            return results
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return the whole text as a single result
            print(f"Failed to parse JSON response: {e}")
            if response_text.strip():
                return [OCRResult(
                    text=response_text.strip(),
                    confidence=0.5,
                    metadata={'parsing_failed': True}
                )]
            return []
        except Exception as e:
            print(f"Error parsing OCR response: {e}")
            return []
    
    def process_region(self, image_path: str, bbox: List[float]) -> List[OCRResult]:
        """Process a specific region. For VLMs, we crop first then process."""
        try:
            from PIL import Image
            import tempfile
            
            img = Image.open(image_path)
            x, y, w, h = [int(v) for v in bbox]
            cropped = img.crop((x, y, x + w, y + h))
            
            # Save to temp and process
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cropped.save(tmp.name)
                results = self.process_image(tmp.name)
                
                # Clean up
                os.unlink(tmp.name)
                
                return results
        except Exception as e:
            print(f"Error processing region: {e}")
            return []


class QwenOCR(VLMOCRBase):
    """OCR implementation using Qwen VL via OpenRouter."""
    
    def __init__(self, api_key=None, model="qwen/qwen-2-vl-72b-instruct", **kwargs):
        """Initialize Qwen OCR.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Qwen model to use
            **kwargs: Additional configuration
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(api_key=api_key, model=model, **kwargs)
    
    def is_available(self) -> bool:
        """Check if Qwen OCR is available."""
        return self.api_key is not None
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with Qwen VL."""
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")
        
        try:
            image_data_uri = self._encode_image(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._create_ocr_prompt()},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }],
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return self._parse_ocr_response(content)
            else:
                print(f"Qwen API error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error processing image with Qwen: {e}")
            return []


class GemmaOCR(VLMOCRBase):
    """OCR implementation using Gemma via OpenRouter."""
    
    def __init__(self, api_key=None, model="google/gemma-2-9b-it:free", **kwargs):
        """Initialize Gemma OCR.
        
        Note: Gemma models may have limited vision capabilities.
        Consider using a vision-capable variant if available.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Gemma model to use
            **kwargs: Additional configuration
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(api_key=api_key, model=model, **kwargs)
    
    def is_available(self) -> bool:
        """Check if Gemma OCR is available."""
        return self.api_key is not None
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with Gemma.
        
        Note: This uses the same API structure as Qwen.
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")
        
        try:
            image_data_uri = self._encode_image(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._create_ocr_prompt()},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }],
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return self._parse_ocr_response(content)
            else:
                print(f"Gemma API error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error processing image with Gemma: {e}")
            return []


class DeepseekOCR(VLMOCRBase):
    """OCR implementation using Deepseek via OpenRouter."""
    
    def __init__(self, api_key=None, model="deepseek/deepseek-chat", **kwargs):
        """Initialize Deepseek OCR.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Deepseek model to use
            **kwargs: Additional configuration
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(api_key=api_key, model=model, **kwargs)
    
    def is_available(self) -> bool:
        """Check if Deepseek OCR is available."""
        return self.api_key is not None
    
    def process_image(self, image_path: str) -> List[OCRResult]:
        """Process image with Deepseek."""
        if not self.api_key:
            raise ValueError("OpenRouter API key not set")
        
        try:
            image_data_uri = self._encode_image(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._create_ocr_prompt()},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }],
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return self._parse_ocr_response(content)
            else:
                print(f"Deepseek API error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error processing image with Deepseek: {e}")
            return []
