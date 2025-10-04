"""
CLOSURE-Lite Dataset for DataSpec v0.3 format
"""

import json
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from pathlib import Path
from closure_lite_framework import build_adjacency_and_next, comp_features_for_panel

class ComicsPageDataset(torch.utils.data.Dataset):
    def __init__(self, json_paths, image_root, max_panels=12, rtl=False, text_model='roberta-base'):
        # Store file paths instead of loading all data
        self.json_paths = json_paths
        print(f"Dataset initialized with {len(json_paths)} JSON files")
        print("Pages will be loaded on-demand during training")
        
        self.image_root = image_root
        self.max_panels = max_panels
        self.rtl = rtl
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])

    def __len__(self): return len(self.json_paths)

    def __getitem__(self, idx):
        # Load page data on-demand
        json_path = self.json_paths[idx]
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): 
                    data = [data]
                # Use first page from file (most files have 1 page)
                page = data[0]
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            # Return a dummy page if loading fails
            page = {
                'page_image_path': '',
                'panels': [],
                'reading_order': []
            }
        # Resolve image path - try relative to image_root first, then absolute
        img_path = (
            page.get('page_image_path') or
            page.get('image_path') or
            page.get('img_path') or
            page.get('image') or
            page.get('path') or
            ''
        )

        missing_image = False
        img = None
        W, H = 1000, 1000
        panels = page.get('panels') or []

        if img_path:
            # Convert Windows paths to WSL paths if needed
            if img_path.startswith('E:/') or img_path.startswith('E:\\'):
                img_path = "/mnt/e" + img_path[2:].replace('\\', '/')
            elif img_path.startswith('E:'):
                img_path = "/mnt/e" + img_path[1:].replace('\\', '/')

            # If it's a relative path, make it relative to image_root
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root, img_path)

            # If still doesn't exist, try to find it in the image_root
            if not os.path.exists(img_path):
                img_filename = os.path.basename(img_path)
                found = False
                for root, dirs, files in os.walk(self.image_root):
                    if img_filename in files:
                        img_path = os.path.join(root, img_filename)
                        found = True
                        break
                if not found:
                    missing_image = True
            
            if not missing_image and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    W, H = img.size
                except Exception as e:
                    print(f"Warning: failed to open image {img_path}: {e}")
                    missing_image = True
        else:
            # No image path available
            missing_image = True

        # crop panels and build per-panel data
        # crop panels and build per-panel data
        crops, texts, comps, boxes = [], [], [], []
        if not missing_image and img is not None:
            for p in panels[:self.max_panels]:
                try:
                    x,y,w,h = p['panel_coords']
                except Exception:
                    # Skip malformed panel entry
                    continue
                crop = img.crop((x, y, x+w, y+h))
                crops.append(self.tf(crop))
                # aggregate text
                tdict = p.get('text', {}) or {}
                parts = []
                for k in ('dialogue','narration','sfx'):
                    vals = tdict.get(k) or []
                    # Handle both strings and lists of strings
                    for v in vals:
                        if isinstance(v, str) and v.strip():
                            parts.append(v.strip())
                        elif isinstance(v, list):
                            parts.extend([str(item).strip() for item in v if str(item).strip()])
                text = ' | '.join(parts) if parts else ''
                texts.append(text)
                comps.append(comp_features_for_panel(p, W, H))
                boxes.append([x/W, y/H, w/W, h/H])
        else:
            # Fallback: no image available; create an empty sample (no panels)
            pass

        # reading order + adjacency
        adj_mask, next_idx, order = build_adjacency_and_next(panels[:len(crops)], W, H, rtl=self.rtl)
        # pad to max_panels
        N = len(crops)
        padN = self.max_panels - N
        if padN > 0:
            pad_img = torch.zeros(3,224,224)
            crops += [pad_img]*padN
            texts += ['']*padN
            comps += [np.zeros(7, dtype=np.float32)]*padN
            boxes += [[0,0,0,0]]*padN
            adj_pad = np.zeros((self.max_panels, self.max_panels), dtype=np.int64)
            adj_pad[:N,:N] = adj_mask
            adj_mask = adj_pad
            next_pad = np.full((self.max_panels,), -100, dtype=np.int64)
            next_pad[:N] = next_idx
            next_idx = next_pad

        # tokenize text
    tok = self.tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        batch = {
            'images': torch.stack(crops),           # (N,3,224,224)
            'input_ids': tok['input_ids'],          # (N,L)
            'attention_mask': tok['attention_mask'],
            'comp_feats': torch.tensor(np.stack(comps), dtype=torch.float32),  # (N,7)
            'boxes': torch.tensor(boxes, dtype=torch.float32),                 # (N,4)
            'panel_mask': torch.zeros(self.max_panels, dtype=torch.bool).index_fill_(0, torch.arange(N), True),
            'adj_mask': torch.tensor(adj_mask, dtype=torch.long),              # (N,N)
            'next_idx': torch.tensor(next_idx, dtype=torch.long),              # (N,)
        }
        # Add original page data for visualization
        batch['original_page'] = page
        # Also add the JSON file name for reference
        batch['json_file'] = os.path.basename(json_path)
        
        return batch

def collate_pages(batch_list):
    # each item is a page dict
    keys = batch_list[0].keys()
    out = {}
    for k in keys:
        if k in ('original_page', 'json_file'):
            out[k] = [b[k] for b in batch_list]  # Keep as list of dicts/strings
        elif k in ('panel_mask', 'next_idx'):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N) or (B,N)
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N,...) or (B,N,N)
    return out

def create_dataloader(json_dir, image_root, batch_size=4, max_panels=12, rtl=False, num_workers=2, max_samples=None):
    """Create dataloader from DataSpec JSON files with optional sampling"""
    # Convert Windows paths to WSL paths if needed
    if json_dir.startswith('E:/') or json_dir.startswith('E:\\'):
        json_dir = "/mnt/e" + json_dir[2:].replace('\\', '/')
    if image_root.startswith('E:/') or image_root.startswith('E:\\'):
        image_root = "/mnt/e" + image_root[2:].replace('\\', '/')
    
    print(f"Looking for JSON files in: {json_dir}")
    all_json_paths = list(Path(json_dir).glob("*.json"))
    print(f"Found {len(all_json_paths)} JSON files")
    
    if len(all_json_paths) == 0:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    # Sample files if max_samples is specified
    if max_samples and max_samples < len(all_json_paths):
        import random
        json_paths = random.sample(all_json_paths, max_samples)
        print(f"Sampled {len(json_paths)} files for training")
    else:
        json_paths = all_json_paths
        print(f"Using all {len(json_paths)} files")
    
    dataset = ComicsPageDataset(json_paths, image_root, max_panels=max_panels, rtl=rtl)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_pages,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    import sys
    if len(sys.argv) > 1:
        json_dir = sys.argv[1]
        image_root = sys.argv[2] if len(sys.argv) > 2 else "E:/amazon"
        
        print(f"Testing dataset with JSON dir: {json_dir}")
        print(f"Image root: {image_root}")
        
        dataloader = create_dataloader(json_dir, image_root, batch_size=2)
        
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Panel mask: {batch['panel_mask'].sum(dim=1)}")
            if i >= 2:  # Test first 3 batches
                break
        print("Dataset test completed!")
    else:
        print("Usage: python closure_lite_dataset.py <json_dir> [image_root]")
