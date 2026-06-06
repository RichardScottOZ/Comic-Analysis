"""
Stage 3 Dataset Loader — VLM-Backed (Manifest-Driven)

Replaces stage3_dataset.py for Stage 3 training using the VLM analysis output
from batch_vlm_analysis_lithops_v2.py (stored locally via sync_vlm_cache.py).

Key differences from Stage3PanelDataset:
- Reads VLM JSONs (description + dialogue) instead of old OCR panel JSONs
- Converts Gemini box_2d [y1,x1,y2,x2] 0-1000 coordinates to pixel crops
- Panel text is description + all dialogue/caption text_content entries
- overall_summary included in metadata for future page-level use
- Direct canonical_id lookup with normalized fallback (no suffix map needed)

Output tensor shapes are identical to Stage3PanelDataset so train_stage3.py
and generate_stage3_embeddings.py need only import swaps.
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

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

NARRATIVE_TYPES = {'narrative', 'story'}


class Stage3PanelDatasetVLM(Dataset):
    """
    Stage 3 training dataset backed by VLM JSON output.

    Args:
        image_map:           Dict mapping canonical_id → absolute_image_path
                             (built from master manifest in train_stage3.py)
        vlm_cache_dir:       Root directory of locally cached VLM JSONs
                             (populated by sync_vlm_cache.py)
        pss_labels_path:     Path to PSS labels JSON (canonical_id → page_type)
        text_model_name:     HuggingFace tokenizer name for panel text
        max_text_length:     Token limit for panel text (truncated if longer)
        image_size:          Panel crop resize target (px)
        only_narrative:      If True, skip non-story/narrative pages
        max_panels_per_page: Maximum panels to use per page (pad/truncate)
        limit:               Cap total number of pages loaded (for smoke tests)
    """

    def __init__(self,
                 image_map: Dict[str, str],
                 vlm_cache_dir: str,
                 pss_labels_path: str,
                 text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_text_length: int = 128,
                 image_size: int = 224,
                 only_narrative: bool = True,
                 max_panels_per_page: int = 16,
                 limit: Optional[int] = None):

        self.image_map = image_map
        self.vlm_cache_dir = Path(vlm_cache_dir)
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

        # Build normalized reverse lookup: normalized_key → (canonical_id, image_path)
        # Used as fallback when PSS label keys differ from manifest canonical_ids
        self._image_map_norm: Dict[str, Tuple[str, str]] = {}
        for cid, path in image_map.items():
            self._image_map_norm[self._normalize_key(cid)] = (cid, path)

        print(f"Loading PSS labels: {pss_labels_path}")
        with open(pss_labels_path, 'r') as f:
            self.pss_labels = json.load(f)

        self.samples = self._build_index(limit)
        print(f"Stage3PanelDatasetVLM ready: {len(self.samples):,} samples.")

    def _scan_vlm_cache(self) -> set:
        """Walk vlm_cache_dir once and return a set of canonical_ids (no .json suffix)."""
        cached = set()
        root = self.vlm_cache_dir
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith('.json'):
                    full = os.path.join(dirpath, fname)
                    # canonical_id = path relative to cache root, without .json
                    rel = os.path.relpath(full, root)
                    canonical_id = rel[:-5]  # strip .json
                    # Normalise to forward slashes regardless of OS
                    canonical_id = canonical_id.replace('\\', '/')
                    cached.add(canonical_id)
        return cached

    @staticmethod
    def _normalize_key(cid: str) -> str:
        """Strip path prefixes, extensions, lowercase, unify separators."""
        for prefix in ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/",
                       "CalibreComics_extracted\\", "amazon/"]:
            if cid.startswith(prefix):
                cid = cid[len(prefix):]
        cid = cid.lower()
        for ext in ('.jpg.png', '.png', '.jpg', '.jpeg'):
            if cid.endswith(ext):
                cid = cid[:-len(ext)]
                break
        return cid.replace('/', '_').replace('\\', '_').strip()

    def _build_index(self, limit: Optional[int] = None) -> List[Dict]:
        samples = []
        skipped_label = skipped_img = skipped_json = added = 0

        # Pre-build set of canonical_ids that have a VLM JSON — one directory
        # scan is vastly faster than 923K individual path.exists() calls.
        print(f"Scanning VLM cache: {self.vlm_cache_dir} ...")
        cached = self._scan_vlm_cache()
        print(f"  {len(cached):,} VLM JSONs found in cache.")

        # image_map comes from the manifest (built from real files), so presence
        # in image_map is sufficient — no need for per-file os.path.exists.
        image_map_norm_keys = set(self._image_map_norm.keys())

        print(f"Building index from {len(self.pss_labels):,} PSS label entries...")

        for cid, page_type in tqdm(self.pss_labels.items(), desc="Building index"):
            if limit and added >= limit:
                break

            if self.only_narrative and page_type not in NARRATIVE_TYPES:
                skipped_label += 1
                continue

            # Resolve canonical_id and image path using in-memory sets (no disk I/O)
            img_path = self.image_map.get(cid)
            canonical_id = cid
            if not img_path:
                norm = self._normalize_key(cid)
                if norm in image_map_norm_keys:
                    canonical_id, img_path = self._image_map_norm[norm]

            if not img_path:
                skipped_img += 1
                continue

            # VLM JSON check against pre-scanned set (no per-file disk I/O)
            if canonical_id not in cached:
                skipped_json += 1
                if skipped_json <= 3:
                    print(f"[DEBUG] VLM JSON not in cache: {canonical_id}.json")
                continue

            samples.append({
                'canonical_id': canonical_id,
                'page_type': page_type,
                'image_path': img_path,
                'json_path': str(self.vlm_cache_dir / f"{canonical_id}.json"),
            })
            added += 1

        print(
            f"\n--- Index Summary ---\n"
            f"  Added            : {added:,}\n"
            f"  Skipped (label)  : {skipped_label:,}\n"
            f"  Skipped (no img) : {skipped_img:,}\n"
            f"  Skipped (no JSON): {skipped_json:,}"
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _panel_text(panel: Dict) -> str:
        """Combine VLM description + all dialogue/caption texts into one string."""
        parts = []
        desc = panel.get('description', '').strip()
        if desc:
            parts.append(desc)
        for tc in panel.get('text_content', []):
            t = tc.get('text', '').strip()
            if t:
                parts.append(t)
        return ' '.join(parts)

    @staticmethod
    def _compute_comp_features(box_2d: List[int], panel_number: int,
                                max_panels: int) -> np.ndarray:
        """
        7-dim compositional feature vector (same shape as Stage3PanelDataset).

        box_2d: [y1, x1, y2, x2] in 0-1000 normalised space.
        """
        y1, x1, y2, x2 = box_2d
        w_norm = (x2 - x1) / 1000.0
        h_norm = (y2 - y1) / 1000.0
        aspect_ratio = w_norm / h_norm if h_norm > 0 else 1.0
        rel_area = w_norm * h_norm
        center_x = (x1 + x2) / 2000.0
        center_y = (y1 + y2) / 2000.0
        panel_num_norm = (panel_number - 1) / max(max_panels - 1, 1)
        return np.array(
            [aspect_ratio, rel_area, panel_num_norm, 0.0, 0.0, center_x, center_y],
            dtype=np.float32
        )

    def _load_page_data(self, json_path: str, image_path: str) -> Dict:
        """Parse VLM JSON + load page image, return panel crops + metadata."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            page_image = Image.open(image_path).convert('RGB')
        except Exception:
            return {'panels': [], 'overall_summary': '', 'page_width': 100, 'page_height': 100}

        pw, ph = page_image.size
        raw_panels = data.get('panels', [])[:self.max_panels_per_page]
        panel_data = []

        for p in raw_panels:
            box_2d = p.get('box_2d')
            if not box_2d or len(box_2d) != 4:
                continue

            # box_2d = [y1, x1, y2, x2] in 0-1000 space → pixel coordinates
            y1, x1, y2, x2 = box_2d
            px1 = max(0, int(x1 / 1000.0 * pw))
            py1 = max(0, int(y1 / 1000.0 * ph))
            px2 = min(pw, int(x2 / 1000.0 * pw))
            py2 = min(ph, int(y2 / 1000.0 * ph))

            if px2 <= px1 or py2 <= py1:
                continue

            panel_num = p.get('panel_number', len(panel_data) + 1)
            panel_data.append({
                'image': page_image.crop((px1, py1, px2, py2)),
                'text': self._panel_text(p),
                'box_2d': box_2d,
                'comp_feats': self._compute_comp_features(
                    box_2d, panel_num, self.max_panels_per_page
                ),
            })

        return {
            'panels': panel_data,
            'overall_summary': data.get('overall_summary', ''),
            'page_width': pw,
            'page_height': ph,
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        page_data = self._load_page_data(sample['json_path'], sample['image_path'])
        panels = page_data['panels']

        images, texts, comp_feats, modality_masks = [], [], [], []

        if not panels:
            images.append(torch.zeros(3, self.image_size, self.image_size))
            texts.append("")
            comp_feats.append(torch.zeros(7))
            modality_masks.append([0.0, 0.0, 0.0])
            num_panels = 1
        else:
            for p in panels:
                images.append(self.transform(p['image']))
                texts.append(p['text'])
                comp_feats.append(torch.from_numpy(p['comp_feats']))
                has_text = 1.0 if p['text'].strip() else 0.0
                modality_masks.append([1.0, has_text, 1.0])
            num_panels = len(panels)

        text_enc = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_text_length, return_tensors='pt'
        )
        images = torch.stack(images)
        input_ids = text_enc['input_ids']
        attention_mask = text_enc['attention_mask']
        comp_feats = torch.stack(comp_feats)
        modality_masks = torch.tensor(modality_masks)
        panel_mask = torch.ones(len(images), dtype=torch.bool)

        # Pad to max_panels_per_page
        n = len(images)
        if n < self.max_panels_per_page:
            pad = self.max_panels_per_page - n
            images = torch.cat([images, torch.zeros(pad, 3, self.image_size, self.image_size)])
            input_ids = torch.cat([input_ids, torch.zeros(pad, self.max_text_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, self.max_text_length, dtype=torch.long)])
            comp_feats = torch.cat([comp_feats, torch.zeros(pad, 7)])
            modality_masks = torch.cat([modality_masks, torch.zeros(pad, 3)])
            panel_mask = torch.cat([panel_mask, torch.zeros(pad, dtype=torch.bool)])

        return {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'comp_feats': comp_feats,
            'panel_mask': panel_mask,
            'modality_mask': modality_masks,
            'metadata': {
                'canonical_id': sample['canonical_id'],
                'num_panels': num_panels,
                'overall_summary': page_data['overall_summary'],
            },
        }


def collate_stage3(batch: List[Dict]) -> Dict:
    """Collate a list of dataset items into a batched dict."""
    return {
        k: torch.stack([s[k] for s in batch]) if torch.is_tensor(batch[0][k])
        else [s[k] for s in batch]
        for k in batch[0].keys()
    }
