# Comic Embedding Framework: Preprocessing Instructions

Before running the embedding framework, several preprocessing steps must be completed to prepare your comic data. This guide outlines the required preprocessing pipeline, leveraging existing tools from the CoMix repository where appropriate.

## 1. Prerequisites

- Python 3.8+ with PyTorch 1.10+
- CUDA-compatible GPU (16GB A5000 or equivalent)
- The following libraries:
  - transformers
  - torchvision
  - PIL
  - numpy
  - matplotlib (for visualization)

## 2. Preprocessing Pipeline

### 2.1. Panel Segmentation

Use the CoMix repository's panel segmentation tools:

```bash
# Clone CoMix repository if not already available
git clone https://github.com/google-research/google-research
cd google-research/comix

# Run panel segmentation on your comic pages
python panel_segmentation.py --input_dir=/path/to/comic/pages --output_dir=/path/to/output
```

**Output needed**: Panel bounding boxes in format (x, y, width, height) for each page.

### 2.2. Text Detection and OCR

Extract text from panels using either:

1. CoMix text detection module:
```bash
python text_detection.py --panel_dir=/path/to/panels --output_dir=/path/to/text_output
```

2. Or use a dedicated OCR tool like Tesseract with comic-specific settings:
```bash
python comic_ocr.py --input_dir=/path/to/panels --output_json=/path/to/text_data.json
```

**Output needed**: Text content with bounding boxes for speech bubbles, captions, and sound effects.

### 2.3. Character Detection (Optional but Recommended)

If available in your CoMix implementation:
```bash
python character_detection.py --panel_dir=/path/to/panels --output_dir=/path/to/character_data
```

**Output needed**: Character bounding boxes within each panel.

### 2.4. Reading Order Annotation

Either:
1. Use CoMix's reading order inference:
```bash
python reading_order.py --panel_json=/path/to/panel_data.json --output_json=/path/to/ordered_panels.json
```

2. Or manually annotate panel reading order indices

**Output needed**: Sequential index for each panel indicating reading order.

## 3. Required JSON Format

The framework expects JSON files with the following structure:

```json
{
  "comic_id": "unique_comic_identifier",
  "title": "Comic Title",
  "pages": [
    {
      "page_id": "page001",
      "image_path": "/path/to/page_image.jpg",
      "panels": [
        {
          "panel_id": "panel001",
          "panel_coords": [x, y, width, height],
          "panel_index": 0,  // Reading order index
          "text_elements": [
            {
              "type": "dialogue|caption|sfx",
              "content": "Text content",
              "bbox": [x, y, width, height]
            }
          ],
          "character_coords": [
            {
              "bbox": [x, y, width, height],
              "character_id": "optional_character_id"
            }
          ]
        }
      ]
    }
  ]
}
```

## 4. Processing Script

Create a preprocessing script that combines all these steps:

```python
import os
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

def preprocess_comic(comic_folder, output_json):
    """
    Processes a comic book folder into the required JSON format
    
    Args:
        comic_folder: Path to folder containing comic page images
        output_json: Path where output JSON will be saved
    """
    # Use CoMix panel segmentation
    panel_output = Path(comic_folder) / "panel_data"
    os.makedirs(panel_output, exist_ok=True)
    
    subprocess.run([
        "python", 
        "/path/to/comix/panel_segmentation.py",
        "--input_dir", comic_folder,
        "--output_dir", str(panel_output)
    ])
    
    # Load panel data
    with open(panel_output / "panels.json", "r") as f:
        panel_data = json.load(f)
    
    # Run text detection
    text_output = Path(comic_folder) / "text_data"
    os.makedirs(text_output, exist_ok=True)
    
    subprocess.run([
        "python",
        "/path/to/comix/text_detection.py",
        "--panel_dir", str(panel_output),
        "--output_dir", str(text_output)
    ])
    
    # Load text data
    with open(text_output / "text.json", "r") as f:
        text_data = json.load(f)
    
    # Run reading order inference
    subprocess.run([
        "python",
        "/path/to/comix/reading_order.py",
        "--panel_json", str(panel_output / "panels.json"),
        "--output_json", str(panel_output / "ordered_panels.json")
    ])
    
    # Load reading order
    with open(panel_output / "ordered_panels.json", "r") as f:
        reading_order = json.load(f)
    
    # Combine all data into final format
    comic_data = {
        "comic_id": Path(comic_folder).stem,
        "title": Path(comic_folder).stem,
        "pages": []
    }
    
    # Process each page
    for page_id, page_info in panel_data.items():
        page_entry = {
            "page_id": page_id,
            "image_path": page_info["image_path"],
            "panels": []
        }
        
        # Add panel information
        for panel_id, panel_info in page_info["panels"].items():
            # Find reading order index
            panel_index = reading_order.get(panel_id, {}).get("index", 0)
            
            # Get text for this panel
            panel_text = []
            if panel_id in text_data:
                for text_item in text_data[panel_id]:
                    panel_text.append({
                        "type": text_item["type"],
                        "content": text_item["text"],
                        "bbox": text_item["bbox"]
                    })
            
            # Create panel entry
            panel_entry = {
                "panel_id": panel_id,
                "panel_coords": panel_info["bbox"],
                "panel_index": panel_index,
                "text_elements": panel_text,
                "character_coords": []  # Optional, if character detection was run
            }
            
            page_entry["panels"].append(panel_entry)
        
        # Sort panels by reading order
        page_entry["panels"].sort(key=lambda p: p["panel_index"])
        comic_data["pages"].append(page_entry)
    
    # Save final JSON
    with open(output_json, "w") as f:
        json.dump(comic_data, f, indent=2)
    
    return output_json

# Example usage
preprocess_comic("/path/to/comic_book", "/path/to/output.json")
```

## 5. Additional Preprocessing Requirements

### 5.1. Panel Masking for Self-Supervised Training

For training with the Masked Panel Modeling objective, create a separate script to generate masked panel data:

```python
def create_masked_panel_dataset(json_path, output_dir, mask_ratio=0.15):
    """
    Creates a dataset with randomly masked panels for self-supervised training
    
    Args:
        json_path: Path to processed comic JSON
        output_dir: Directory to save masked panel data
        mask_ratio: Fraction of panels to mask (default 15%)
    """
    with open(json_path, 'r') as f:
        comic_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For each page, create multiple masked versions
    masked_examples = []
    
    for page in comic_data['pages']:
        page_img = Image.open(page['image_path'])
        
        # Skip pages with fewer than 3 panels
        if len(page['panels']) < 3:
            continue
            
        for _ in range(3):  # Create 3 different maskings per page
            masked_page = page.copy()
            masked_panels = masked_page['panels'].copy()
            
            # Determine which panels to mask
            num_to_mask = max(1, int(len(masked_panels) * mask_ratio))
            mask_indices = np.random.choice(
                len(masked_panels), num_to_mask, replace=False)
            
            # Track masked panels
            masked_info = []
            
            for idx in mask_indices:
                panel = masked_panels[idx]
                
                # Store original panel info
                masked_info.append({
                    'panel_id': panel['panel_id'],
                    'panel_index': panel['panel_index'],
                    'original_coords': panel['panel_coords']
                })
                
                # Create white mask for the panel in the page image
                x, y, w, h = panel['panel_coords']
                mask = Image.new('RGB', (w, h), (255, 255, 255))
                page_img.paste(mask, (x, y))
                
                # Mark panel as masked in the data
                panel['is_masked'] = True
            
            # Save masked page image
            masked_page_path = f"{output_dir}/masked_{page['page_id']}_{len(masked_examples)}.jpg"
            page_img.save(masked_page_path)
            
            # Save example
            masked_examples.append({
                'masked_page_path': masked_page_path,
                'masked_panels': masked_info,
                'panels': masked_panels,
                'page_id': page['page_id']
            })
    
    # Save masked dataset
    with open(f"{output_dir}/masked_panels_dataset.json", 'w') as f:
        json.dump(masked_examples, f, indent=2)
```

### 5.2. Feature Extraction Cache

To save GPU memory during training and avoid redundant processing, pre-extract visual features:

```python
def precompute_visual_features(json_path, output_dir, vit_model, vit_processor):
    """
    Pre-extracts ViT features for all panels to avoid redundant processing
    
    Args:
        json_path: Path to processed comic JSON
        output_dir: Directory to save extracted features
        vit_model: Loaded ViT model
        vit_processor: ViT feature processor
    """
    with open(json_path, 'r') as f:
        comic_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    feature_cache = {}
    
    for page in comic_data['pages']:
        page_img = Image.open(page['image_path'])
        
        for panel in page['panels']:
            panel_id = panel['panel_id']
            x, y, w, h = panel['panel_coords']
            
            # Crop panel
            panel_img = page_img.crop((x, y, x+w, y+h))
            
            # Extract features
            with torch.no_grad():
                inputs = vit_processor(images=panel_img, return_tensors="pt").to(vit_model.device)
                outputs = vit_model(**inputs)
                features = outputs.pooler_output.cpu().numpy()
            
            # Store features
            feature_cache[panel_id] = features
    
    # Save feature cache
    np.save(f"{output_dir}/visual_features.npy", feature_cache)
    return feature_cache
```

## 6. Integration with Embedding Framework

After completing these preprocessing steps, your data will be ready for the embedding framework. The final pipeline involves:

1. Run the preprocessing pipeline to generate the JSON structure
2. Pre-extract visual features if needed to save GPU memory
3. Create masked panel datasets for self-supervised training
4. Run the embedding model training with the prepared data

## 7. Using CoMix Tools vs Custom Implementation

The CoMix repository tools are recommended for:
- Panel segmentation (more robust than generic object detection)
- Text bubble detection (specialized for comics)
- Reading order inference (handles complex comic layouts)

If the CoMix tools are unavailable or incompatible, you can substitute with:
- Panel segmentation: YOLOv5 or Mask R-CNN trained on comic panels
- Text detection: Tesseract OCR or EasyOCR with comic-specific settings
- Reading order: Left-to-right, top-to-bottom heuristic algorithm

The embedding framework is designed to be flexible regarding the source of preprocessing, as long as the final JSON structure matches the expected format.