"""
Stage 3 Dataset Loader (Manifest-Driven)

Dataset for training Stage 3 panel feature extractors.
Uses a Master Manifest to locate images across multiple drives/folders.
Uses a JSON root to locate aligned panel metadata.
"""

import os
import json
import torch
import csv
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T
from pathlib import Path

class Stage3PanelDataset(Dataset):
    """
    Manifest-Driven Dataset for Stage 3.
    
    Args:
        manifest_path: Path to master_manifest.csv (maps ID -> Image Path)
        json_root: Root directory containing aligned Stage 3 JSONs
        pss_labels_path: Path to PSS labels JSON (from Stage 2 export)
    """
    
    def __init__(self, 
                 manifest_path: str,
                 json_root: str,
                 pss_labels_path: str,
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_text_length: int = 128,
                 image_size: int = 224,
                 only_narrative: bool = True,
                 max_panels_per_page: int = 16):
        
        self.json_root = json_root
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
        print(f"Loading PSS Labels: {pss_labels_path}")
        with open(pss_labels_path, 'r') as f:
            self.pss_labels = json.load(f)
            
        # Build Index from Manifest
        self.samples = self._build_index(manifest_path)
        
        print(f"Stage 3 Dataset initialized:")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Only narrative: {only_narrative}")
    
    def _build_index(self, manifest_path) -> List[Dict]:
        samples = []
        print(f"Loading Manifest: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row['canonical_id']
                img_path = row['absolute_image_path']
                
                # Check PSS Label
                # PSS Labels key format: "BookID/PageID" or just "PageFilename"?
                # Assuming the export tool keys by canonical_id or similar unique ID.
                # We will support a direct lookup by canonical_id.
                
                page_type = self.pss_labels.get(cid, "unknown")
                
                if self.only_narrative and page_type != 'narrative' and page_type != 'story':
                    continue
                
                # Locate JSON
                # json_root / canonical_id.json
                # Handle Windows paths if needed
                cid_path = cid.replace('/', os.sep)
                json_path = os.path.join(self.json_root, f"{cid_path}.json")
                
                if not os.path.exists(json_path):
                    continue
                
                samples.append({
                    'canonical_id': cid,
                    'page_type': page_type,
                    'image_path': img_path,
                    'json_path': json_path
                })
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_page_data(self, json_path: str, image_path: str) -> Dict:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load full page image
        try:
            page_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data to avoid crashing (dataloader will filter or fail gracefully)
            return {'panels': [], 'page_width': 100, 'page_height': 100}

        page_width, page_height = page_image.size
        
        panels = data.get('panels', [])
        if len(panels) > self.max_panels_per_page:
            panels = panels[:self.max_panels_per_page]
        
        panel_data = []
        for panel in panels:
            # Stage 3 JSON format: bbox is [x, y, w, h]
            bbox = panel.get('bbox')
            if not bbox or len(bbox) != 4: continue
            
            text = panel.get('text', '')
            
            x, y, w, h = bbox
            # Clamp crop to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, page_width - x)
            h = min(h, page_height - y)
            
            if w <= 0 or h <= 0: continue
            
            panel_crop = page_image.crop((x, y, x+w, y+h))
            
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
            'page_height': page_height
        }
    
    def _compute_comp_features(self, bbox: List[float], 
                               page_width: int, page_height: int) -> np.ndarray:
        x, y, w, h = bbox
        return np.array([
            w / h if h > 0 else 1.0,
            (w * h) / (page_width * page_height),
            0.0, 0.0, 0.0,
            (x + w/2) / page_width,
            (y + h/2) / page_height
        ], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        page_data = self._load_page_data(sample['json_path'], sample['image_path'])
        panels = page_data['panels']
        num_panels = len(panels)
        
        images = []
        texts = []
        comp_feats = []
        modality_masks = []
        
        if num_panels == 0:
            # Handle empty page (should be rare for 'story' pages)
            # Create one dummy panel
            images.append(torch.zeros(3, self.image_size, self.image_size))
            texts.append("")
            comp_feats.append(torch.zeros(7))
            modality_masks.append([0.0, 0.0, 0.0])
            num_panels = 1
        else:
            for panel in panels:
                images.append(self.transform(panel['image']))
                texts.append(panel['text'])
                comp_feats.append(torch.from_numpy(panel['comp_feats']))
                
                has_image = 1.0
                has_text = 1.0 if len(panel['text'].strip()) > 0 else 0.0
                modality_masks.append([has_image, has_text, 1.0]) # 1.0 for comp
        
        # Tokenize
        text_encoding = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=self.max_text_length, return_tensors='pt'
        )
        
        images = torch.stack(images)
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        comp_feats = torch.stack(comp_feats)
        modality_masks = torch.tensor(modality_masks)
        panel_mask = torch.ones(len(images), dtype=torch.bool)
        
        # Padding
        if len(images) < self.max_panels_per_page:
            pad_size = self.max_panels_per_page - len(images)
            images = torch.cat([images, torch.zeros(pad_size, 3, self.image_size, self.image_size)], dim=0)
            input_ids = torch.cat([input_ids, torch.zeros(pad_size, self.max_text_length, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_size, self.max_text_length, dtype=torch.long)], dim=0)
            comp_feats = torch.cat([comp_feats, torch.zeros(pad_size, 7)], dim=0)
            modality_masks = torch.cat([modality_masks, torch.zeros(pad_size, 3)], dim=0)
            panel_mask = torch.cat([panel_mask, torch.zeros(pad_size, dtype=torch.bool)], dim=0)
            
        return {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'comp_feats': comp_feats,
            'panel_mask': panel_mask,
            'modality_mask': modality_masks,
            'metadata': {
                'canonical_id': sample['canonical_id'],
                'num_panels': num_panels
            }
        }

def collate_stage3(batch: List[Dict]) -> Dict:
    return {
        'images': torch.stack([s['images'] for s in batch]),
        'input_ids': torch.stack([s['input_ids'] for s in batch]),
        'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
        'comp_feats': torch.stack([s['comp_feats'] for s in batch]),
        'panel_mask': torch.stack([s['panel_mask'] for s in batch]),
        'modality_mask': torch.stack([s['modality_mask'] for s in batch]),
        'metadata': [s['metadata'] for s in batch]
    }