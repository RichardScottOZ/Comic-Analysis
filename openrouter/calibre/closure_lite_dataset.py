"""
CLOSURE-Lite Dataset for DataSpec v0.3 format
"""

import json
import os
import torch
import numpy as np
from PIL import Image, ImageFile
# Allow loading of truncated images to avoid runtime crashes on partially corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from transformers import AutoTokenizer
from pathlib import Path
import re
from closure_lite_framework import build_adjacency_and_next, comp_features_for_panel

def _to_wsl_path(p: str) -> str:
    """Convert Windows drive paths (e.g., E:\foo or E:/foo or E:foo) to WSL (/mnt/e/foo) when on POSIX.
    If not POSIX or not a drive path, return unchanged.
    """
    try:
        if not isinstance(p, str):
            return p
        if os.name != 'posix':
            return p
        m = re.match(r'^[A-Za-z]:(.*)$', p)
        if m:
            drive = p[0].lower()
            rest = m.group(1)
            # Insert leading slash if missing
            if rest and not (rest.startswith('\\') or rest.startswith('/')):
                rest = '/' + rest
            rest = rest.replace('\\', '/')
            return f"/mnt/{drive}{rest}"
        return p
    except Exception:
        return p

class ComicsPageDataset(torch.utils.data.Dataset):
    def __init__(self, json_paths, image_root, max_panels=12, rtl=False, text_model='roberta-base', precomputed_map: dict | None = None):
        # Store file paths instead of loading all data
        self.json_paths = json_paths
        print(f"Dataset initialized with {len(json_paths)} JSON files")
        print("Pages will be loaded on-demand during training")
        
        # Normalize image_root for WSL if needed
        self.image_root = _to_wsl_path(image_root)
        self.max_panels = max_panels
        self.rtl = rtl
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])
        # Lazy-built image index: basename(lower) -> full path(s)
        self._image_index = None
        self._allowed_suffixes = (
            '.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp',
            # common composites from conversion pipelines
            '.jpg.png', '.jpeg.png', '.png.png', '.jpg.jpeg', '.png.jpg',
            '.tif.png', '.tiff.png', '.tif.jpg', '.tiff.jpg', '.tif.jpeg', '.tiff.jpeg',
            # rare multi-step
            '.jpg.png.png', '.jpeg.png.png'
        )
        self._index_notice_printed = False
        # Cache for resolved paths; prefill with precomputed map if provided
        self._resolved_cache = {}
        if precomputed_map:
            # Normalize keys for consistency
            for jp, ip in precomputed_map.items():
                if isinstance(jp, str) and isinstance(ip, str):
                    self._resolved_cache[jp] = ip

    def _use_wsl_paths(self) -> bool:
        """Return True if we should normalize Windows drive paths to /mnt/* style.
        Heuristic: only when image_root already looks like a WSL path.
        """
        try:
            return isinstance(self.image_root, str) and self.image_root.startswith('/mnt/')
        except Exception:
            return False

    def _build_image_index(self):
        index = {}
        stem_index = {}
        norm_index = {}
        last_token_index = {}
        last_num_index = {}
        try:
            if not self._index_notice_printed:
                print(f"[ComicsPageDataset] Building global image index under '{self.image_root}' (one-time per worker)...")
                self._index_notice_printed = True
            for root, _, files in os.walk(self.image_root):
                for fname in files:
                    fl = fname.lower()
                    full = os.path.join(root, fname)
                    index.setdefault(fl, []).append(full)
                    stem = os.path.splitext(fl)[0]
                    stem_index.setdefault(stem, []).append(full)
                    normstem = re.sub(r"[^a-z0-9]+", "", stem)
                    norm_index.setdefault(normstem, []).append(full)
                    # last token (split on non-alnum)
                    parts = [p for p in re.split(r"[^a-z0-9]+", stem) if p]
                    if parts:
                        last = parts[-1]
                        last_token_index.setdefault(last, []).append(full)
                        # numeric variant without leading zeros
                        if last.isdigit():
                            nz = last.lstrip('0') or '0'
                            last_num_index.setdefault(nz, []).append(full)
            print(f"[ComicsPageDataset] Indexed {sum(len(v) for v in index.values())} files.")
        except Exception:
            index = {}
            stem_index = {}
            norm_index = {}
            last_token_index = {}
            last_num_index = {}
        self._image_index = index
        self._image_index_by_stem = stem_index
        self._image_index_by_normstem = norm_index
        self._image_index_by_lasttoken = last_token_index
        self._image_index_by_lastnum = last_num_index

    def __len__(self): return len(self.json_paths)

    def resolve_image_path(self, json_path: str) -> str | None:
        """Resolve the image path for a given JSON using the same inference logic,
        without opening the image or constructing tensors. Returns a path if it exists, else None.
        """
        # Cache shortcut
        cached = self._resolved_cache.get(json_path)
        if cached and os.path.exists(cached):
            return cached

        # Load the first page from JSON
        page = None
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    page = data
                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    page = data[0]
        except Exception:
            page = None

        img_path = None
        inferred = False

        # Accept multiple schema variants from the JSON
        if page:
            img_path = (
                page.get('page_image_path')
                or page.get('image_path')
                or page.get('image')
                or page.get('IMAGE_PATH')
            )
            if isinstance(img_path, str):
                img_path = _to_wsl_path(img_path)

        # If no explicit image path field, try to infer from JSON filename and parent folder (Calibre VLM format)
        if not img_path:
            json_base = os.path.splitext(os.path.basename(json_path))[0]
            parent_name = os.path.basename(os.path.dirname(json_path))
            cand1 = os.path.join(self.image_root, parent_name, json_base)
            cand2 = os.path.join(self.image_root, json_base)
            for cand in (cand1, cand2):
                if os.path.exists(cand):
                    img_path = cand
                    inferred = True
                    break
            if not inferred and ('.' not in os.path.basename(json_base)):
                for ext in self._allowed_suffixes:
                    for cand in (cand1 + ext, cand2 + ext):
                        if os.path.exists(cand):
                            img_path = cand
                            inferred = True
                            break
                    if inferred:
                        break
            if not inferred:
                parent_dir = os.path.join(self.image_root, parent_name)
                if os.path.isdir(parent_dir):
                    try:
                        wanted = os.path.basename(json_base).lower()
                        suffixes = self._allowed_suffixes
                        for fname in os.listdir(parent_dir):
                            fl = fname.lower()
                            if (os.path.splitext(fl)[0] == wanted) or (fl.startswith(wanted) and any(fl.endswith(s) for s in suffixes)):
                                img_path = os.path.join(parent_dir, fname)
                                inferred = True
                                break
                    except Exception:
                        pass
        # If still not inferred, try a global index lookup (once per dataset)
        if not img_path:
            if self._image_index is None:
                self._build_image_index()
            try:
                json_base = os.path.splitext(os.path.basename(json_path))[0]
                wanted = os.path.basename(json_base).lower()
                if '.' in wanted:
                    hits = self._image_index.get(wanted)
                    if hits:
                        img_path = hits[0]
                if not img_path:
                    for ext in self._allowed_suffixes:
                        hits = self._image_index.get((wanted + ext))
                        if hits:
                            img_path = hits[0]
                            break
                if not img_path:
                    for name, paths in self._image_index.items():
                        if name.startswith(wanted) and any(name.endswith(s) for s in self._allowed_suffixes):
                            img_path = paths[0]
                            break
                # 4) stem match (filename without extension)
                if not img_path and hasattr(self, '_image_index_by_stem'):
                    hits = self._image_index_by_stem.get(wanted)
                    if hits:
                        img_path = hits[0]
                # 5) normalized stem match (alnum-only)
                if not img_path and hasattr(self, '_image_index_by_normstem'):
                    wanted_norm = re.sub(r"[^a-z0-9]+", "", wanted)
                    hits = self._image_index_by_normstem.get(wanted_norm)
                    if hits:
                        img_path = hits[0]
                # 6) last-token match (handles *_001 -> 001.jpg)
                if not img_path and hasattr(self, '_image_index_by_lasttoken'):
                    parts = [p for p in re.split(r"[^a-z0-9]+", wanted) if p]
                    if parts:
                        last = parts[-1]
                        hits = self._image_index_by_lasttoken.get(last)
                        if hits:
                            img_path = hits[0]
                        # numeric variant without leading zeros
                        if not img_path and last.isdigit() and hasattr(self, '_image_index_by_lastnum'):
                            nz = last.lstrip('0') or '0'
                            hits = self._image_index_by_lastnum.get(nz)
                            if hits:
                                img_path = hits[0]
            except Exception:
                pass

        # Normalize and final existence check
        if img_path:
            # Convert any Windows drive path to WSL if under POSIX
            img_path = _to_wsl_path(img_path)
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root, img_path)
            if not os.path.exists(img_path):
                parent_dir = os.path.join(self.image_root, os.path.basename(os.path.dirname(json_path)))
                img_filename = os.path.basename(img_path)
                if os.path.isdir(parent_dir):
                    try:
                        for fname in os.listdir(parent_dir):
                            if fname.lower() == img_filename.lower():
                                img_path = os.path.join(parent_dir, fname)
                                break
                    except Exception:
                        pass
        if img_path and os.path.exists(img_path):
            self._resolved_cache[json_path] = img_path
            return img_path
        return None

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
        
        # Resolve image path with shared resolver (no image loading here)
        img_path = self.resolve_image_path(json_path)
        if not img_path:
            # Provide useful context when schema differs
            available_keys = list(page.keys()) if isinstance(page, dict) else []
            wanted = os.path.splitext(os.path.basename(json_path))[0].lower()
            parent_name = os.path.basename(os.path.dirname(json_path))
            print("[DATASET ERROR] Could not resolve image path!")
            print(f"  JSON file: {json_path}")
            print(f"  Available keys: {available_keys}")
            print(f"  image_root: {self.image_root}")
            print(f"  parent_dir: {parent_name}")
            print(f"  wanted: {wanted}")
            raise KeyError(
                f"No image path field found in page and inference failed. Tried ['page_image_path','image_path','image','IMAGE_PATH']. "
                f"Available keys: {available_keys}. json_path='{json_path}', image_root='{self.image_root}', parent_dir='{parent_name}', wanted='{wanted}'"
            )
        
        # Convert Windows paths to WSL paths only when image_root is WSL-like
        if self._use_wsl_paths():
            if img_path.startswith('E:/') or img_path.startswith('E:\\'):
                img_path = "/mnt/e" + img_path[2:].replace('\\', '/')
            elif img_path.startswith('E:'):
                img_path = "/mnt/e" + img_path[1:].replace('\\', '/')
        
        # If it's a relative path, make it relative to image_root
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)
        
        # If still doesn't exist, try a bounded search: within the parent folder under image_root
        if not os.path.exists(img_path):
            parent_dir = os.path.join(self.image_root, os.path.basename(os.path.dirname(json_path)))
            img_filename = os.path.basename(img_path)
            if os.path.isdir(parent_dir):
                try:
                    for fname in os.listdir(parent_dir):
                        if fname.lower() == img_filename.lower():
                            img_path = os.path.join(parent_dir, fname)
                            break
                except Exception:
                    pass

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
    # Cache is already updated in resolve_image_path
        
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        panels = page['panels']
        # crop panels and build per-panel data
        crops, texts, comps, boxes = [], [], [], []
        for p in panels[:self.max_panels]:
            x,y,w,h = p['panel_coords']
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
    json_dir = _to_wsl_path(json_dir)
    image_root = _to_wsl_path(image_root)
    
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
