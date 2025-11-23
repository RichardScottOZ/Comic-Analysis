"""
Stage 3 Dataset Loader

Dataset for training Stage 3 panel feature extractors.
Expects input from Stage 2 (PSS/CoSMo) that has classified narrative pages.

Dataset format:
- Only processes 'narrative' pages from PSS
- Loads panel crops, text, and compositional features
- Supports both training and inference modes
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T


class Stage3PanelDataset(Dataset):
    """
    Dataset for Stage 3 panel feature learning.
    
    Expected directory structure:
        root_dir/
            book_id/
                page_001.jpg
                page_001.json  # Contains OCR text and panel detections
                page_002.jpg
                page_002.json
                ...
            pss_labels.json  # From Stage 2 CoSMo: page type classifications
    
    The pss_labels.json format:
    {
        "book_id": {
            "page_001": "narrative",
            "page_002": "advertisement",
            "page_003": "narrative",
            ...
        }
    }
    """
    
    def __init__(self, 
                 root_dir: str,
                 pss_labels_path: str,
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_text_length: int = 128,
                 image_size: int = 224,
                 only_narrative: bool = True,
                 max_panels_per_page: int = 16):
        """
        Args:
            root_dir: Root directory containing book subdirectories
            pss_labels_path: Path to PSS labels JSON from Stage 2
            text_model_name: Tokenizer model name
            max_text_length: Maximum text sequence length
            image_size: Image size for panel crops
            only_narrative: Whether to only load narrative pages
            max_panels_per_page: Maximum number of panels per page
        """
        self.root_dir = root_dir
        self.text_model_name = text_model_name
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.only_narrative = only_narrative
        self.max_panels_per_page = max_panels_per_page
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Load PSS labels
        with open(pss_labels_path, 'r') as f:
            self.pss_labels = json.load(f)
        
        # Build dataset index
        self.samples = self._build_index()
        
        print(f"Stage 3 Dataset initialized:")
        print(f"  - Total pages: {len(self.samples)}")
        print(f"  - Only narrative: {only_narrative}")
        print(f"  - Max panels per page: {max_panels_per_page}")
    
    def _build_index(self) -> List[Dict]:
        """
        Build index of all panel samples.
        
        Returns:
            List of sample dictionaries containing paths and metadata
        """
        samples = []
        
        for book_id in os.listdir(self.root_dir):
            book_dir = os.path.join(self.root_dir, book_id)
            if not os.path.isdir(book_dir):
                continue
            
            # Check if book has PSS labels
            if book_id not in self.pss_labels:
                continue
            
            book_labels = self.pss_labels[book_id]
            
            # Find all pages
            for filename in os.listdir(book_dir):
                if not filename.endswith('.jpg') and not filename.endswith('.png'):
                    continue
                
                page_name = os.path.splitext(filename)[0]
                
                # Check PSS label
                if page_name not in book_labels:
                    continue
                
                page_type = book_labels[page_name]
                
                # Filter by page type if needed
                if self.only_narrative and page_type != 'narrative':
                    continue
                
                # Check for corresponding JSON
                json_path = os.path.join(book_dir, f"{page_name}.json")
                if not os.path.exists(json_path):
                    continue
                
                image_path = os.path.join(book_dir, filename)
                
                samples.append({
                    'book_id': book_id,
                    'page_name': page_name,
                    'page_type': page_type,
                    'image_path': image_path,
                    'json_path': json_path
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_page_data(self, json_path: str, image_path: str) -> Dict:
        """
        Load panel data from JSON file.
        
        Expected JSON format (from R-CNN + VLM):
        {
            "image_width": 1988,
            "image_height": 3057,
            "panels": [
                {
                    "bbox": [x, y, w, h],
                    "text": "Panel dialogue/narration",
                    "confidence": 0.95
                },
                ...
            ]
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load full page image
        page_image = Image.open(image_path).convert('RGB')
        page_width, page_height = page_image.size
        
        # Extract panels
        panels = data.get('panels', [])
        
        # Limit number of panels
        if len(panels) > self.max_panels_per_page:
            panels = panels[:self.max_panels_per_page]
        
        panel_data = []
        for panel in panels:
            bbox = panel['bbox']
            text = panel.get('text', '')
            
            # Crop panel from page
            x, y, w, h = bbox
            panel_crop = page_image.crop((x, y, x+w, y+h))
            
            # Compute compositional features
            comp_feats = self._compute_comp_features(bbox, page_width, page_height)
            
            panel_data.append({
                'image': panel_crop,
                'text': text,
                'bbox': bbox,
                'comp_feats': comp_feats
            })
        
        return {
            'panels': panel_data,
            'page_width': page_width,
            'page_height': page_height,
            'num_panels': len(panel_data)
        }
    
    def _compute_comp_features(self, bbox: List[float], 
                               page_width: int, page_height: int) -> np.ndarray:
        """
        Compute compositional features for a panel.
        
        Features (7-dimensional):
        1. Aspect ratio (w/h)
        2. Size ratio (panel_area / page_area)
        3. Character count (placeholder, set to 0)
        4. Shot mean (placeholder, set to 0)
        5. Shot max (placeholder, set to 0)
        6. Center X (normalized)
        7. Center Y (normalized)
        """
        x, y, w, h = bbox
        
        return np.array([
            w / h if h > 0 else 1.0,  # aspect ratio
            (w * h) / (page_width * page_height),  # size ratio
            0.0,  # character count (placeholder)
            0.0,  # shot mean (placeholder)
            0.0,  # shot max (placeholder)
            (x + w/2) / page_width,   # center x
            (y + h/2) / page_height   # center y
        ], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single page sample with all its panels.
        
        Returns:
            Dictionary containing:
            - images: (N, 3, H, W) panel images
            - input_ids: (N, max_length) tokenized text
            - attention_mask: (N, max_length) attention masks
            - comp_feats: (N, 7) compositional features
            - panel_mask: (N,) binary mask for valid panels
            - modality_mask: (N, 3) modality presence indicators
            - metadata: dict with book_id, page_name, etc.
        """
        sample = self.samples[idx]
        
        # Load page data
        page_data = self._load_page_data(sample['json_path'], sample['image_path'])
        panels = page_data['panels']
        num_panels = len(panels)
        
        # Prepare tensors
        images = []
        texts = []
        comp_feats = []
        modality_masks = []
        
        for panel in panels:
            # Process image
            panel_img = self.transform(panel['image'])
            images.append(panel_img)
            
            # Store text
            texts.append(panel['text'])
            
            # Store compositional features
            comp_feats.append(panel['comp_feats'])
            
            # Compute modality mask
            has_image = True
            has_text = len(panel['text'].strip()) > 0
            has_comp = True
            modality_masks.append([float(has_image), float(has_text), float(has_comp)])
        
        # Tokenize all texts at once
        text_encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # Stack tensors
        images = torch.stack(images)  # (N, 3, H, W)
        input_ids = text_encoding['input_ids']  # (N, max_length)
        attention_mask = text_encoding['attention_mask']  # (N, max_length)
        comp_feats = torch.from_numpy(np.stack(comp_feats))  # (N, 7)
        modality_masks = torch.tensor(modality_masks)  # (N, 3)
        
        # Create panel mask (all valid since we loaded them)
        panel_mask = torch.ones(num_panels, dtype=torch.bool)
        
        # Pad to max_panels_per_page if needed
        if num_panels < self.max_panels_per_page:
            pad_size = self.max_panels_per_page - num_panels
            
            # Pad images
            pad_img = torch.zeros(pad_size, 3, self.image_size, self.image_size)
            images = torch.cat([images, pad_img], dim=0)
            
            # Pad text
            pad_ids = torch.zeros(pad_size, self.max_text_length, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            
            pad_mask = torch.zeros(pad_size, self.max_text_length, dtype=torch.long)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
            
            # Pad comp features
            pad_comp = torch.zeros(pad_size, 7)
            comp_feats = torch.cat([comp_feats, pad_comp], dim=0)
            
            # Pad modality masks
            pad_mod = torch.zeros(pad_size, 3)
            modality_masks = torch.cat([modality_masks, pad_mod], dim=0)
            
            # Extend panel mask
            pad_panel_mask = torch.zeros(pad_size, dtype=torch.bool)
            panel_mask = torch.cat([panel_mask, pad_panel_mask], dim=0)
        
        return {
            'images': images,  # (max_panels, 3, H, W)
            'input_ids': input_ids,  # (max_panels, max_length)
            'attention_mask': attention_mask,  # (max_panels, max_length)
            'comp_feats': comp_feats,  # (max_panels, 7)
            'panel_mask': panel_mask,  # (max_panels,)
            'modality_mask': modality_masks,  # (max_panels, 3)
            'metadata': {
                'book_id': sample['book_id'],
                'page_name': sample['page_name'],
                'page_type': sample['page_type'],
                'num_panels': num_panels,
                'page_width': page_data['page_width'],
                'page_height': page_data['page_height']
            }
        }


def collate_stage3(batch: List[Dict]) -> Dict:
    """
    Collate function for Stage 3 dataset.
    
    Args:
        batch: List of samples from Stage3PanelDataset
        
    Returns:
        Batched dictionary
    """
    # Stack all tensors
    images = torch.stack([s['images'] for s in batch])  # (B, N, 3, H, W)
    input_ids = torch.stack([s['input_ids'] for s in batch])  # (B, N, max_length)
    attention_mask = torch.stack([s['attention_mask'] for s in batch])  # (B, N, max_length)
    comp_feats = torch.stack([s['comp_feats'] for s in batch])  # (B, N, 7)
    panel_mask = torch.stack([s['panel_mask'] for s in batch])  # (B, N)
    modality_mask = torch.stack([s['modality_mask'] for s in batch])  # (B, N, 3)
    
    # Collect metadata
    metadata = [s['metadata'] for s in batch]
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'comp_feats': comp_feats,
        'panel_mask': panel_mask,
        'modality_mask': modality_mask,
        'metadata': metadata
    }


if __name__ == "__main__":
    print("Stage 3 Dataset Loader")
    print("=" * 60)
    print("\nThis dataset:")
    print("1. Loads narrative pages classified by Stage 2 (PSS/CoSMo)")
    print("2. Extracts panel crops with text and compositional features")
    print("3. Prepares batches for Stage 3 panel feature learning")
    print("4. Supports modality masking for flexible training")
