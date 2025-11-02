"""
CLOSURE-Lite Dataset for DataSpec v0.3 format
"""

import json
import os
import torch
import numpy as np
from PIL import Image, ImageFile, ImageOps
import warnings
# Allow loading of truncated images to avoid runtime crashes on partially corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Disable decompression bomb warnings for very large comic pages; we immediately crop and downscale panels.
try:
    Image.MAX_IMAGE_PIXELS = None
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
except Exception:
    pass

def _safe_open_image(img_path: str):
    """Open very large images safely, disabling decompression-bomb checks and
    performing an eager load. If opening still fails, raise an error with context.
    """
    # Ensure checks are disabled right before open (defensive in case other code reset it)
    try:
        Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    try:
        img = Image.open(img_path)
        # Normalize orientation based on EXIF to avoid landscape/rotation oddities
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert('RGB')
        # Force actual read to catch issues early
        img.load()
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to open image '{img_path}': {e}")
import torchvision.transforms as T
from transformers import AutoTokenizer
from pathlib import Path
import re
from closure_lite_framework import build_adjacency_and_next, comp_features_for_panel

def _load_json_with_fallbacks(json_path: str):
    """Robustly load a JSON file trying UTF-8 first, then common fallbacks.
    Returns a Python object or raises the last exception if all attempts fail.
    """
    # Try strict UTF-8 first (most DataSpec files are UTF-8)
    encs = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("cp1252", "strict"),
        ("latin-1", "strict"),
        ("utf-8", "replace"),  # as a last resort, replace bad bytes in strings
    ]
    last_err = None
    for enc, err in encs:
        try:
            with open(json_path, 'r', encoding=enc, errors=err) as f:
                return json.load(f)
        except Exception as e:
            last_err = e
            continue
    # If we reach here, all attempts failed
    raise last_err if last_err else RuntimeError(f"Failed to read JSON: {json_path}")

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
        self._zero_panel_logs = 0

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
            data = _load_json_with_fallbacks(json_path)
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
            data = _load_json_with_fallbacks(json_path)
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
        
        img = _safe_open_image(img_path)
        W, H = img.size

        # --- Helper: robustly extract a panel bbox from varied schemas ---
        def _bbox_from_any(p: dict, W: int, H: int):
            try:
                # 1) Standard DataSpec
                if isinstance(p.get('panel_coords'), (list, tuple)) and len(p['panel_coords']) == 4:
                    x,y,w,h = p['panel_coords']
                    return int(x), int(y), int(w), int(h)
                # 2) Common VLM shapes
                b = p.get('bbox') or p.get('box') or p.get('rect') or None
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    x1,y1,x2y2_1,x2y2_2 = b
                    # Heuristic: detect [x,y,w,h] vs [x1,y1,x2,y2]
                    # If third value is greater than width or looks like x2>x1, treat as x2,y2
                    # Otherwise treat as w,h
                    if (x2y2_1 > 1 and x2y2_2 > 1 and (x2y2_1 > x1 or x2y2_2 > y1)):
                        x1 = float(x1); y1 = float(y1); x2 = float(x2y2_1); y2 = float(x2y2_2)
                        x, y, w, h = x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
                    else:
                        x, y, w, h = float(x1), float(y1), float(x2y2_1), float(x2y2_2)
                    # Detect normalized coords (<=1.1) and scale
                    if max(x, y, w, h) <= 1.1:
                        x, y, w, h = x*W, y*H, w*W, h*H
                    return int(max(0, x)), int(max(0, y)), int(max(1, w)), int(max(1, h))
                # 3) Polygon segmentation → bbox
                poly = p.get('polygon') or p.get('segmentation')
                if isinstance(poly, (list, tuple)) and len(poly) >= 4:
                    # flatten [[x,y],...] or [x1,y1,x2,y2,...]
                    pts = []
                    if all(isinstance(t, (list, tuple)) and len(t) == 2 for t in poly):
                        pts = [(float(px), float(py)) for (px,py) in poly]
                    else:
                        arr = list(poly)
                        for i in range(0, len(arr)-1, 2):
                            pts.append((float(arr[i]), float(arr[i+1])))
                    if pts:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        # Detect normalized and scale
                        if max(x1, y1, x2, y2) <= 1.1:
                            x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
                        return int(max(0, x1)), int(max(0, y1)), int(max(1, x2-x1)), int(max(1, y2-y1))
            except Exception:
                pass
            return None

        # --- Helper: aggregate text content into DataSpec-like 'text' dict ---
        def _aggregate_text(p: dict) -> dict:
            out = {'dialogue': [], 'narration': [], 'sfx': []}
            t = p.get('text')
            def _add(val, bucket='dialogue'):
                if isinstance(val, str) and val.strip():
                    out[bucket].append(val.strip())
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        if isinstance(v, str) and v.strip():
                            out[bucket].append(v.strip())
            if isinstance(t, dict):
                for k in ('dialogue','narration','sfx','caption'):
                    _add(t.get(k), 'dialogue' if k=='dialogue' else ('narration' if k in ('narration','caption') else 'sfx'))
                # include any other non-empty strings
                for k,v in t.items():
                    if k not in ('dialogue','narration','sfx','caption'):
                        _add(v, 'dialogue')
            elif isinstance(t, (list, tuple)):
                _add(t, 'dialogue')
            elif isinstance(t, str):
                _add(t, 'dialogue')
            # Speakers-style schema
            sp = p.get('speakers')
            if isinstance(sp, list):
                for s in sp:
                    if isinstance(s, dict):
                        txt = s.get('dialogue') or s.get('text')
                        st = str(s.get('speech_type') or s.get('type') or '').lower()
                        if txt:
                            if any(tok in st for tok in ('narration','caption','narrator')):
                                _add(txt, 'narration')
                            elif any(tok in st for tok in ('sfx','sound','onomatopoeia','effect')):
                                _add(txt, 'sfx')
                            else:
                                _add(txt, 'dialogue')
            # OCR-like
            for k in ('ocr','ocr_text','ocr_lines','texts'):
                _add(p.get(k), 'dialogue')
            return out

        # --- Helper: character coords (optional) ---
        def _character_coords(p: dict):
            chars = []
            items = p.get('characters') or p.get('faces') or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        b = it.get('bbox') or it.get('box')
                        if isinstance(b, (list, tuple)) and len(b) == 4:
                            bx = _bbox_from_any({'bbox': b}, W, H)
                            if bx:
                                chars.append(bx)
            return chars

        # Normalize panels to ensure 'panel_coords' exists for downstream code
        def _extract_panels(page_dict: dict):
            # Preferred keys first
            preferred_keys = [
                'panels', 'VLM_panels', 'vlm_panels', 'panels_vlm',
                'panel_detections', 'rcnn_panels', 'panel_boxes', 'panel_list',
                'detections', 'boxes', 'segments', 'annotations'
            ]
            for k in preferred_keys:
                v = page_dict.get(k)
                if isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                    return v
            # Fallback: scan any list-valued fields that look like panel dicts
            for k, v in page_dict.items():
                if isinstance(v, list) and v and any(isinstance(x, dict) for x in v):
                    # Check if dicts contain any bbox-like keys
                    if any(any(kk in d for kk in ('panel_coords','bbox','box','rect','polygon','segmentation')) for d in v if isinstance(d, dict)):
                        return v
            return []

        panels_raw = _extract_panels(page)
        norm_panels = []
        for pr in panels_raw:
            if not isinstance(pr, dict):
                continue
            bbox = _bbox_from_any(pr, W, H)
            if not bbox:
                continue
            tx = _aggregate_text(pr)
            ch = _character_coords(pr)
            norm_panels.append({'panel_coords': list(bbox), 'text': tx, 'character_coords': ch})
        panels = norm_panels
        if not panels and self._zero_panel_logs < 5:
            self._zero_panel_logs += 1
            print(f"[Dataset] No panels extracted for {json_path}. Available keys: {list(page.keys())[:10]}")
        # crop panels and build per-panel data
        crops, texts, comps, boxes = [], [], [], []
        for p in panels[:self.max_panels]:
            x,y,w,h = p['panel_coords']
            # Clamp to page bounds to tolerate slight size errors
            x = max(0, min(int(x), W-1))
            y = max(0, min(int(y), H-1))
            w = max(1, int(w)); h = max(1, int(h))
            x2 = max(x+1, min(x+w, W))
            y2 = max(y+1, min(y+h, H))
            crop = img.crop((x, y, x2, y2))
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
            'num_panels': torch.tensor(N, dtype=torch.long),
        }
        # Add original page data for visualization
        batch['original_page'] = page
        # Also add the JSON path, JSON file name and resolved image path for reference
        batch['json_path'] = json_path
        batch['json_file'] = os.path.basename(json_path)
        batch['image_path'] = img_path

        return batch

def collate_pages(batch_list):
    # each item is a page dict
    keys = batch_list[0].keys()
    out = {}
    for k in keys:
        if k in ('original_page', 'json_file', 'json_path', 'image_path'):
            out[k] = [b[k] for b in batch_list]  # Keep as list of dicts/strings
        elif k in ('panel_mask', 'next_idx'):
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N) or (B,N)
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)  # (B,N,...) or (B,N,N)
    return out

def create_dataloader(json_dir, image_root, batch_size=4, max_panels=12, rtl=False, num_workers=2, max_samples=None):
    """Create dataloader from DataSpec JSON files with optional sampling.
    - If json_dir is a directory, recursively loads all *.json under it.
    - If json_dir is a .txt file, treats each non-empty line as a JSON path.
    """
    # Convert Windows paths to WSL paths if needed
    json_dir = _to_wsl_path(json_dir)
    image_root = _to_wsl_path(image_root)

    p = Path(json_dir)
    if p.is_file() and p.suffix.lower() == '.txt':
        print(f"Loading JSON list from: {json_dir}")
        try:
            with open(p, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            all_json_paths = [Path(_to_wsl_path(l)) for l in lines if l.lower().endswith('.json')]
        except Exception as e:
            raise ValueError(f"Failed to read JSON list file '{json_dir}': {e}")
        print(f"List contains {len(all_json_paths)} JSON files")
    else:
        print(f"Looking for JSON files under (recursive): {json_dir}")
        all_json_paths = list(Path(json_dir).rglob("*.json"))
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

def create_dataloader_from_list(json_list_file: str, image_root: str, batch_size: int = 4,
                                num_workers: int = 2, max_samples: int | None = None,
                                max_panels: int = 12, rtl: bool = False,
                                seed: int | None = None, dedupe: bool = True,
                                sample_without_replacement: bool = True):
    """Create a DataLoader from a text file listing JSON paths (one per line).
    Paths are normalized using the same WSL-aware converter used elsewhere.
    """
    json_list_file = _to_wsl_path(json_list_file)
    image_root = _to_wsl_path(image_root)
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    lines: list[str] = []
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(json_list_file, 'r', encoding=enc, errors='strict') as f:
                lines = [l.strip() for l in f if l.strip()]
            break
        except Exception as e:
            last_err = e
            continue
    if not lines:
        raise RuntimeError(f"Failed to read list file '{json_list_file}': {last_err}")
    paths = []
    for l in lines:
        p = _to_wsl_path(l)
        if os.path.exists(p):
            paths.append(p)
        else:
            # try original literal
            if os.path.exists(l):
                paths.append(l)
    # Optional de-duplication by normalized path (case-insensitive on Windows)
    if dedupe and paths:
        normed = []
        seen = set()
        for p in paths:
            key = os.path.normcase(os.path.normpath(p))
            if key not in seen:
                seen.add(key)
                normed.append(p)
        if len(normed) != len(paths):
            try:
                print(f"[create_dataloader_from_list] De-duplicated list: {len(paths)} -> {len(normed)} unique paths")
            except Exception:
                pass
        paths = normed

    # Optional sampling
    if max_samples and max_samples > 0 and len(paths) > max_samples:
        import random
        rng = random.Random(seed) if seed is not None else random
        if sample_without_replacement:
            paths = rng.sample(paths, k=max_samples)
        else:
            # With replacement (rarely needed)
            paths = [rng.choice(paths) for _ in range(max_samples)]
    if not paths:
        raise ValueError(f"No valid JSON paths found in list: {json_list_file}")
    dataset = ComicsPageDataset(paths, image_root, max_panels=max_panels, rtl=rtl)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pages,
        num_workers=num_workers,
        pin_memory=True,
    )

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
