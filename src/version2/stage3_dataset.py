"""
Stage 3 Dataset Loader (Manifest-Driven)

Dataset for training Stage 3 panel feature extractors.
Uses an Image Map to locate images across multiple drives/folders.
Uses a JSON Map (Suffix-based) to locate aligned panel metadata.
"""

import os
import json
import torch
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T
from pathlib import Path
import time

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Stage3PanelDataset(Dataset):
    def __init__(self, 
                 image_map: Dict[str, str],
                 json_map: Dict[str, str],
                 json_root: str,
                 pss_labels_path: str,
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_text_length: int = 128,
                 image_size: int = 224,
                 only_narrative: bool = True,
                 max_panels_per_page: int = 16,
                 limit: Optional[int] = None):
        
        self.image_map = image_map
        self.json_map = json_map
        self.json_root = json_root
        self.text_model_name = text_model_name
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.only_narrative = only_narrative
        self.max_panels_per_page = max_panels_per_page
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loading PSS Labels: {pss_labels_path}")
        with open(pss_labels_path, 'r') as f:
            self.pss_labels = json.load(f)
            
        self.samples = self._build_index(limit)
        print(f"Stage 3 Dataset initialized: {len(self.samples)} samples.")
    
    def _normalize_key(self, cid):
        prefixes = ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/", "CalibreComics_extracted\\", "amazon/"]
        for p in prefixes:
            if cid.startswith(p):
                cid = cid.replace(p, "")
        res = cid.lower()
        for ext in ['.jpg', '.png', '.jpeg']:
            res = res.replace(ext, '')
        return res.replace('/', '_').replace('\\', '_').strip()

    def _build_index(self, limit=None) -> List[Dict]:
        samples = []
        skipped_label, skipped_json, skipped_img, added = 0, 0, 0, 0
        
        print(f"Index building started. Total labels: {len(self.pss_labels)}")
        
        for i, (cid, page_type) in tqdm(enumerate(self.pss_labels.items()), total=len(self.pss_labels), desc="Building Index"):
            if limit and added >= limit: break
            
            if self.only_narrative and page_type not in ['narrative', 'story']:
                skipped_label += 1
                continue
            
            img_path = self.image_map.get(cid)
            if not img_path or not os.path.exists(img_path):
                skipped_img += 1
                continue

            # Robust JSON Lookup (Suffix Strategy)
            calibre_id = self.json_map.get(cid)
            if not calibre_id:
                calibre_id = self.json_map.get(Path(cid).name)
            if not calibre_id:
                calibre_id = self.json_map.get(Path(cid).stem)
            
            # Fallback to normalized
            if not calibre_id:
                norm_key = self._normalize_key(cid)
                calibre_id = self.json_map.get(norm_key)

            if not calibre_id:
                skipped_json += 1
                if skipped_json <= 3:
                    print(f"[DEBUG FAIL] No JSON Map match for: '{cid}'")
                continue
                
            json_path = os.path.join(self.json_root, calibre_id.replace('/', os.sep) + ".json")
            if not os.path.exists(json_path):
                skipped_json += 1
                if skipped_json <= 3:
                    print(f"[DEBUG FAIL] Missing JSON File: {json_path}")
                continue
            
            samples.append({'canonical_id': calibre_id, 'page_type': page_type, 'image_path': img_path, 'json_path': json_path})
            added += 1
                
        print(f"\n--- Index Build Summary ---\nTotal: {len(self.pss_labels)} | Added: {added} | Skip (Label): {skipped_label} | Skip (Img): {skipped_img} | Skip (JSON): {skipped_json}")
        return samples
    
    def __len__(self) -> int: return len(self.samples)
    
    def _load_page_data(self, json_path: str, image_path: str) -> Dict:
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            page_image = Image.open(image_path).convert('RGB')
        except Exception: return {'panels': [], 'page_width': 100, 'page_height': 100}
        
        pw, ph = page_image.size
        panels = data.get('panels', [])[:self.max_panels_per_page]
        panel_data = []
        for p in panels:
            bbox = p.get('bbox')
            if not bbox or len(bbox) != 4: continue
            x, y, w, h = bbox
            x, y = max(0, x), max(0, y)
            w, h = min(w, pw - x), min(h, ph - y)
            if w <= 0 or h <= 0: continue
            panel_data.append({
                'image': page_image.crop((x, y, x+w, y+h)), 
                'text': p.get('text', ''), 
                'bbox': bbox,
                'comp_feats': self._compute_comp_features(bbox, pw, ph)
            })
        return {'panels': panel_data, 'page_width': pw, 'page_height': ph}
    
    def _compute_comp_features(self, bbox: List[float], pw: int, ph: int) -> np.ndarray:
        x, y, w, h = bbox
        return np.array([w/h if h>0 else 1.0, (w*h)/(pw*ph), 0.0, 0.0, 0.0, (x+w/2)/pw, (y+h/2)/ph], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        page_data = self._load_page_data(sample['json_path'], sample['image_path'])
        panels = page_data['panels']
        images, texts, comp_feats, modality_masks = [], [], [], []
        if not panels:
            images.append(torch.zeros(3, self.image_size, self.image_size)); texts.append(""); comp_feats.append(torch.zeros(7)); modality_masks.append([0.0, 0.0, 0.0]); num_panels = 1
        else:
            for p in panels:
                images.append(self.transform(p['image'])); texts.append(p['text']); comp_feats.append(torch.from_numpy(p['comp_feats'])); has_text = 1.0 if len(p['text'].strip()) > 0 else 0.0; modality_masks.append([1.0, has_text, 1.0])
            num_panels = len(panels)
        text_encoding = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_text_length, return_tensors='pt')
        images = torch.stack(images); input_ids = text_encoding['input_ids']; attention_mask = text_encoding['attention_mask']; comp_feats = torch.stack(comp_feats); modality_masks = torch.tensor(modality_masks); panel_mask = torch.ones(len(images), dtype=torch.bool)
        if len(images) < self.max_panels_per_page:
            pad = self.max_panels_per_page - len(images)
            images = torch.cat([images, torch.zeros(pad, 3, self.image_size, self.image_size)], 0); input_ids = torch.cat([input_ids, torch.zeros(pad, self.max_text_length, dtype=torch.long)], 0); attention_mask = torch.cat([attention_mask, torch.zeros(pad, self.max_text_length, dtype=torch.long)], 0); comp_feats = torch.cat([comp_feats, torch.zeros(pad, 7)], 0); modality_masks = torch.cat([modality_masks, torch.zeros(pad, 3)], 0); panel_mask = torch.cat([panel_mask, torch.zeros(pad, dtype=torch.bool)], 0)
        return {'images': images, 'input_ids': input_ids, 'attention_mask': attention_mask, 'comp_feats': comp_feats, 'panel_mask': panel_mask, 'modality_mask': modality_masks, 'metadata': {'canonical_id': sample['canonical_id'], 'num_panels': num_panels}}

def collate_stage3(batch: List[Dict]) -> Dict:
    return {k: torch.stack([s[k] for s in batch]) if torch.is_tensor(batch[0][k]) else [s[k] for s in batch] for k in batch[0].keys()}
