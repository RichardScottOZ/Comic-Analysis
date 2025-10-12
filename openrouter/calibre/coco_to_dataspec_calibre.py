"""
Convert CalibreComics COCO detections to DataSpec v0.3 format
Similar to the Amazon version but adapted for CalibreComics structure
"""

import json
import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import re
from typing import Dict, Tuple, Optional, List
import csv
import hashlib

# Global config resolved at runtime
panel_category_ids_global: set[int] = {1}
allow_no_vlm_global: bool = False

def load_coco_data(coco_file):
    """Load COCO detection data"""
    print(f"Loading COCO data from {coco_file}...")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    image_detections = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_detections:
            image_detections[image_id] = []
        image_detections[image_id].append(ann)
    
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = img

    categories = coco_data.get('categories', [])
    cat_id_to_name: Dict[int, str] = {c.get('id'): c.get('name', '') for c in categories if isinstance(c, dict) and 'id' in c}
    cat_name_to_id: Dict[str, int] = {str(c.get('name', '')).lower(): c.get('id') for c in categories if isinstance(c, dict) and 'name' in c and 'id' in c}

    # Build a simple histogram of detections per category_id
    cat_counts: Dict[int, int] = {}
    for ann in coco_data['annotations']:
        cid = ann.get('category_id')
        if cid is not None:
            cat_counts[cid] = cat_counts.get(cid, 0) + 1

    print(f"Loaded {len(image_detections)} images with detections and {len(categories)} categories")
    if cat_counts:
        print("Top categories by count (id:name=count):")
        # Print up to 10 most frequent categories
        for cid, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            nm = cat_id_to_name.get(cid, '')
            print(f"  {cid}:{nm} = {cnt}")
    return image_detections, image_info, cat_id_to_name, cat_name_to_id, cat_counts

def _build_image_indexes(image_root: str):
    """Build filename/stem/normalized/last-number indexes for fast matching."""
    fn_idx: dict[str, list[str]] = {}
    stem_idx: dict[str, list[str]] = {}
    norm_idx: dict[str, list[str]] = {}
    lastnum_idx: dict[str, list[str]] = {}
    for root, _, files in os.walk(image_root):
        for fname in files:
            full = os.path.join(root, fname)
            fl = fname.lower()
            fn_idx.setdefault(fl, []).append(full)
            stem = os.path.splitext(fl)[0]
            stem_idx.setdefault(stem, []).append(full)
            norm = re.sub(r"[^a-z0-9]+", "", stem)
            norm_idx.setdefault(norm, []).append(full)
            # last numeric token
            parts = [p for p in re.split(r"[^a-z0-9]+", stem) if p]
            if parts:
                last = parts[-1]
                if last.isdigit():
                    nz = last.lstrip('0') or '0'
                    lastnum_idx.setdefault(nz, []).append(full)
    return fn_idx, stem_idx, norm_idx, lastnum_idx

def _norm_key(p: str) -> str:
    """Normalize a path-like key for robust cross-platform matching."""
    if not p:
        return ""
    return str(p).replace("\\", "/").lower()

def _sanitize_filename_part(text: Optional[str], max_len: int = 120) -> str:
    """Sanitize for Windows filenames while preserving readability.
    Removes only forbidden characters: <>:"/\\|?* and trims spaces/dots.
    Keeps spaces, #, -, (), etc. Truncates to max_len to avoid long paths.
    """
    if not text:
        return ""
    t = str(text)
    # Remove forbidden characters
    t = re.sub(r'[<>:"/\\|?*]', "", t)
    # Collapse excessive spaces
    t = re.sub(r"\s+", " ", t).strip()
    # Trim trailing dots/spaces (invalid on Windows)
    t = t.rstrip(" .")
    if max_len and len(t) > max_len:
        t = t[:max_len].rstrip(" .")
    return t or "item"

def _extract_last_number(text: str) -> Optional[int]:
    parts = [p for p in re.split(r"[^0-9]+", text) if p]
    if parts:
        try:
            return int(parts[-1].lstrip('0') or '0')
        except Exception:
            return None
    return None

def _wsl_to_windows_path(p: str) -> Optional[str]:
    """Convert /mnt/<drive>/... to Windows style if applicable."""
    if not p:
        return None
    if p.startswith('/mnt/') and len(p) > 6:
        drive = p[5].upper()
        rest = p[6:]
        return f"{drive}:{rest}".replace('/', '\\')
    return None

def _resolve_to_image(path_hint: str, image_root: str, indexes):
    """Try to resolve a path hint (relative/varied) to an actual image under image_root."""
    fn_idx, stem_idx, norm_idx, lastnum_idx = indexes
    if not path_hint:
        return None
    hint = path_hint.replace('\\', '/').lstrip('/')
    # direct join
    cand = os.path.join(image_root, hint)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    # try by basename and compound stems (handle cases like .jpg.png)
    base = os.path.basename(hint).lower()
    # Direct filename hit
    hits = fn_idx.get(base)
    if hits:
        return hits[0]
    # Iteratively strip extensions up to 3 times
    candidates: List[str] = []
    seen: set[str] = set()
    cur = base
    for _ in range(3):
        stem = os.path.splitext(cur)[0]
        if not stem or stem in seen:
            break
        seen.add(stem)
        candidates.append(stem)
        cur = stem
    for s in candidates:
        hits = stem_idx.get(s)
        if hits:
            return hits[0]
        nstem = re.sub(r"[^a-z0-9]+", "", s)
        hits = norm_idx.get(nstem)
        if hits:
            return hits[0]
        parts = [p for p in re.split(r"[^a-z0-9]+", s) if p]
        if parts and parts[-1].isdigit():
            nz = parts[-1].lstrip('0') or '0'
            hits = lastnum_idx.get(nz)
            if hits:
                return hits[0]
    return None

def _safe_page_key(image_id, img_path: str, abs_img: Optional[str] = None) -> str:
    """Create a stable, filesystem-safe key for output filename.
    Uses basename plus short hash of a unique string to avoid collisions.

    Priority for uniqueness:
    - absolute image path if available
    - else include image_id to disambiguate repeated relative names like images/iNNN.jpg
    """
    chosen_path = abs_img or img_path or str(image_id)
    base = os.path.splitext(os.path.basename(str(abs_img or img_path or "page")))[0]
    unique_str = f"{_norm_key(chosen_path)}#{image_id}"
    h = hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:10]
    return f"calibre_{base}_{h}"

def _load_precomputed_map(map_path: Optional[str]) -> Dict[str, str]:
    """Load a precomputed json_path->image_path map (JSON or CSV). Keys are normalized."""
    if not map_path:
        return {}
    mp: Dict[str, str] = {}
    try:
        if map_path.lower().endswith('.json'):
            with open(map_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    mp[_norm_key(k)] = v
            elif isinstance(raw, list):
                for row in raw:
                    if isinstance(row, dict) and 'json_path' in row and 'image_path' in row:
                        mp[_norm_key(row['json_path'])] = row['image_path']
        else:
            import csv
            with open(map_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'json_path' in row and 'image_path' in row:
                        mp[_norm_key(row['json_path'])] = row['image_path']
    except Exception as e:
        print(f"Warning: failed to load precomputed map {map_path}: {e}")
    print(f"Loaded precomputed map entries: {len(mp)}")
    return mp

def _iter_vlm_files(vlm_dirs: list[Path]):
    for d in vlm_dirs:
        # Recursively search for JSON files
        for f in d.rglob("*.json"):
            yield f

def load_vlm_data(vlm_dirs_input, image_root: str, precomputed_map: Optional[Dict[str, str]] = None):
    """Load VLM pages and resolve their image paths under image_root or precomputed map; build robust join indexes.
    vlm_dirs_input can be a string path, a list of strings, or a list of Paths. Multiple dirs supported.
    """
    # Normalize dirs list
    if isinstance(vlm_dirs_input, (str, Path)):
        # Support semicolon/comma-separated list in a single string
        if isinstance(vlm_dirs_input, str) and (";" in vlm_dirs_input or "," in vlm_dirs_input):
            parts = re.split(r"[;,]", vlm_dirs_input)
            vlm_dirs = [Path(p.strip()) for p in parts if p.strip()]
        else:
            vlm_dirs = [Path(vlm_dirs_input)]
    else:
        vlm_dirs = [Path(p) for p in vlm_dirs_input]

    print("Loading VLM data from:")
    for d in vlm_dirs:
        print(f"  - {d}")
    vlm_by_abs_img: dict[str, dict] = {}
    by_basename: dict[str, list[dict]] = {}
    by_stem: dict[str, list[dict]] = {}
    by_norm: dict[str, list[dict]] = {}
    by_lastnum: dict[str, list[dict]] = {}

    vlm_files = list(_iter_vlm_files(vlm_dirs))
    print(f"Found {len(vlm_files)} VLM files (recursive)")

    indexes = _build_image_indexes(image_root)
    pre_map = precomputed_map or {}
    resolved_count = 0
    filename_only_count = 0

    for vlm_file in tqdm(vlm_files, desc="Indexing VLM files"):
        try:
            with open(vlm_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pages = []
            if isinstance(data, dict):
                pages = [data]
            elif isinstance(data, list):
                pages = [p for p in data if isinstance(p, dict)]
            for page in pages:
                hint = page.get('page_image_path') or page.get('image_path') or page.get('image')
                abs_img = None
                # 1) Try precomputed map first (most reliable)
                key_forms = {
                    _norm_key(str(vlm_file)),
                    _norm_key(os.path.abspath(str(vlm_file))),
                }
                for k in key_forms:
                    if not abs_img and k in pre_map:
                        abs_img = pre_map[k]
                        # Translate WSL path if needed
                        if isinstance(abs_img, str) and abs_img.startswith('/mnt/'):
                            w = _wsl_to_windows_path(abs_img)
                            if w:
                                abs_img = w
                # 2) Try explicit hint in page
                if not abs_img and hint:
                    abs_img = _resolve_to_image(hint, image_root, indexes)
                # 3) Fallback: infer from VLM json filename (stem)
                if not abs_img:
                    stem = os.path.splitext(os.path.basename(str(vlm_file)))[0]
                    abs_img = _resolve_to_image(stem, image_root, indexes)
                # Align path to current image_root if pre_map path is WSL or different drive
                if abs_img and not os.path.exists(abs_img):
                    # Try to re-resolve by basename/stem/norm under image_root
                    base_try = os.path.basename(abs_img)
                    alt = _resolve_to_image(base_try, image_root, indexes)
                    if alt and os.path.exists(alt):
                        abs_img = alt
                if abs_img:
                    key = abs_img.replace('\\', '/').lower()
                    vlm_by_abs_img[key] = page
                    base = os.path.basename(abs_img).lower()
                    s = os.path.splitext(base)[0]
                    n = re.sub(r"[^a-z0-9]+", "", s)
                    by_basename.setdefault(base, []).append(page)
                    by_stem.setdefault(s, []).append(page)
                    by_norm.setdefault(n, []).append(page)
                    # last numeric token
                    parts = [p for p in re.split(r"[^a-z0-9]+", s) if p]
                    nz = None
                    if parts and parts[-1].isdigit():
                        nz = parts[-1].lstrip('0') or '0'
                    else:
                        m = re.search(r"(\d+)$", s)
                        if m:
                            nz = m.group(1).lstrip('0') or '0'
                    if nz is not None:
                        by_lastnum.setdefault(nz, []).append(page)
                    resolved_count += 1
                else:
                    # Index by VLM JSON filename and by any available hint basename/stem
                    # 1) JSON filename
                    base_json = os.path.basename(str(vlm_file)).lower()
                    s_json = os.path.splitext(base_json)[0]
                    n_json = re.sub(r"[^a-z0-9]+", "", s_json)
                    by_basename.setdefault(base_json, []).append(page)
                    by_stem.setdefault(s_json, []).append(page)
                    by_norm.setdefault(n_json, []).append(page)
                    parts_json = [p for p in re.split(r"[^a-z0-9]+", s_json) if p]
                    nz = None
                    if parts_json and parts_json[-1].isdigit():
                        nz = parts_json[-1].lstrip('0') or '0'
                    else:
                        m = re.search(r"(\d+)$", s_json)
                        if m:
                            nz = m.group(1).lstrip('0') or '0'
                    if nz is not None:
                        by_lastnum.setdefault(nz, []).append(page)
                    # 2) Hint-based indexing (e.g., 'images/i249.jpg' or 'images_i249.jpg')
                    if hint and isinstance(hint, str):
                        base_hint = os.path.basename(hint.replace('\\', '/')).lower()
                        s_hint = os.path.splitext(base_hint)[0]
                        n_hint = re.sub(r"[^a-z0-9]+", "", s_hint)
                        by_basename.setdefault(base_hint, []).append(page)
                        by_stem.setdefault(s_hint, []).append(page)
                        by_norm.setdefault(n_hint, []).append(page)
                        parts_hint = [p for p in re.split(r"[^a-z0-9]+", s_hint) if p]
                        nz_h = None
                        if parts_hint and parts_hint[-1].isdigit():
                            nz_h = parts_hint[-1].lstrip('0') or '0'
                        else:
                            m2 = re.search(r"(\d+)$", s_hint)
                            if m2:
                                nz_h = m2.group(1).lstrip('0') or '0'
                        if nz_h is not None:
                            by_lastnum.setdefault(nz_h, []).append(page)
                    filename_only_count += 1
        except Exception as e:
            print(f"Error loading {vlm_file}: {e}")
            continue

    print(f"Loaded VLM pages: {resolved_count} resolved to images; {filename_only_count} indexed by filename only")
    return vlm_by_abs_img, (by_basename, by_stem, by_norm, by_lastnum), indexes

def convert_detection_to_dataspec(image_id, detections, image_info, joiners, image_root_indexes):
    """Convert a single image's detections to DataSpec format.
    Returns: (dataspec_page_or_none, failure_info_or_none)
    failure_info example (no_vlm_match):
        {
            'reason': 'no_vlm_match', 'image_id': ..., 'img_file_name': ..., 'resolved_abs_img': ..., 'basename': ..., 'stem': ..., 'normstem': ..., 'lastnum_key': ...
        }
    """

    # Get image info
    img_info = image_info[image_id]
    img_path = img_info['file_name']
    img_width = img_info['width']
    img_height = img_info['height']

    vlm_by_abs_img, (by_basename, by_stem, by_norm, by_lastnum), indexes = joiners
    fn_idx, stem_idx, norm_idx, lastnum_idx = image_root_indexes

    # Resolve COCO image file_name to absolute under image_root indexes
    abs_img = _resolve_to_image(img_path, args_image_root_global, (fn_idx, stem_idx, norm_idx, lastnum_idx))
    vlm_match = None
    base_for_lookup = None
    if abs_img:
        k = abs_img.replace('\\', '/').lower()
        vlm_match = vlm_by_abs_img.get(k)
        base_for_lookup = os.path.basename(k).lower()
    else:
        # Fall back to using the file_name string directly for index-based lookups
        base_for_lookup = os.path.basename(str(img_path)).lower()
    if not vlm_match and base_for_lookup:
        s = os.path.splitext(base_for_lookup)[0]
        # Also consider compound extensions by stripping multiple times
        stems: List[str] = []
        cur = base_for_lookup
        seen: set[str] = set()
        for _ in range(3):
            st = os.path.splitext(cur)[0]
            if not st or st in seen:
                break
            seen.add(st)
            stems.append(st)
            cur = st
        found = None
        for st in stems:
            n = re.sub(r"[^a-z0-9]+", "", st)
            parts = [p for p in re.split(r"[^a-z0-9]+", st) if p]
            # Extract trailing numeric token even if stem contains a leading letter (e.g., i249 -> 249)
            lastnum_key = None
            if parts and parts[-1].isdigit():
                lastnum_key = parts[-1].lstrip('0') or '0'
            else:
                m = re.search(r"(\d+)$", st)
                if m:
                    lastnum_key = m.group(1).lstrip('0') or '0'
            for cand in (
                by_stem.get(st, []),
                by_norm.get(n, []),
                by_lastnum.get(lastnum_key, []) if lastnum_key is not None else []
            ):
                if cand:
                    found = cand[0]
                    break
            if found:
                break
        if not found:
            # Lastly try exact basename
            found = (by_basename.get(base_for_lookup, []) or [None])[0]
        vlm_match = found

    if not vlm_match and not allow_no_vlm_global:
        # Build failure info for logging
        s_for = base_for_lookup or ''
        s = os.path.splitext(s_for)[0] if s_for else ''
        n = re.sub(r"[^a-z0-9]+", "", s) if s else ''
        parts = [p for p in re.split(r"[^a-z0-9]+", s) if p] if s else []
        if parts and parts[-1].isdigit():
            lastnum_key = (parts[-1].lstrip('0') or '0')
        else:
            m = re.search(r"(\d+)$", s) if s else None
            lastnum_key = (m.group(1).lstrip('0') or '0') if m else ''
        failure = {
            'reason': 'no_vlm_match',
            'image_id': image_id,
            'img_file_name': img_path,
            'resolved_abs_img': abs_img or '',
            'basename': base_for_lookup or '',
            'stem': s,
            'normstem': n,
            'lastnum_key': lastnum_key,
        }
        print(f"[UNMATCHED] No VLM data found for {img_path}")
        return None, failure

    # Separate detections by category
    panels = []
    characters = []
    faces = []
    texts = []

    for det in detections:
        category_id = det['category_id']
        bbox = det['bbox']  # [x, y, width, height]
        confidence = det.get('score', 1.0)

        # Convert COCO bbox to our format
        x, y, w, h = bbox
        coords = [int(x), int(y), int(w), int(h)]

        # Use configured/auto-detected panel category ids
        if category_id in panel_category_ids_global:  # Panel
            panels.append({
                'panel_coords': coords,
                'confidence': confidence
            })
        elif category_id == 2:  # Character
            characters.append({
                'character_coords': coords,
                'confidence': confidence
            })
        elif category_id == 3:  # Face
            faces.append({
                'face_coords': coords,
                'confidence': confidence
            })
        elif category_id == 4:  # Text
            texts.append({
                'text_coords': coords,
                'confidence': confidence
            })

    # Create panels with VLM text data
    panels_with_text = []
    for i, panel in enumerate(panels):
        # Find matching VLM panel (by position overlap)
        panel_coords = panel['panel_coords']
        best_vlm_panel = None
        best_overlap = 0

        for vlm_panel in (vlm_match.get('panels', []) if vlm_match else []):
            vlm_coords = vlm_panel.get('panel_coords', [])
            if len(vlm_coords) == 4:
                # Calculate overlap
                x1, y1, w1, h1 = panel_coords
                x2, y2, w2, h2 = vlm_coords

                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    union = w1 * h1 + w2 * h2 - intersection
                    overlap = intersection / union if union > 0 else 0

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_vlm_panel = vlm_panel

        # Create panel with text data
        panel_data = {
            'panel_coords': panel_coords,
            'confidence': panel['confidence']
        }

        if best_vlm_panel:
            # Add text data from VLM
            vlm_text = best_vlm_panel.get('text', {})
            panel_data['text'] = {
                'dialogue': vlm_text.get('dialogue', []),
                'narration': vlm_text.get('narration', []),
                'sfx': vlm_text.get('sfx', [])
            }
        else:
            # No matching VLM panel, add empty text
            panel_data['text'] = {
                'dialogue': [],
                'narration': [],
                'sfx': []
            }

        panels_with_text.append(panel_data)

    # Create reading order (simple left-to-right, top-to-bottom)
    reading_order = []
    if len(panels_with_text) > 1:
        # Sort panels by position (top to bottom, left to right)
        sorted_panels = sorted(panels_with_text, key=lambda p: (p['panel_coords'][1], p['panel_coords'][0]))
        for i in range(len(sorted_panels) - 1):
            reading_order.append(f"{i}→{i+1}")

    # Derive human-friendly provenance (book directory and relative path)
    source_book = None
    source_rel_image_path = None
    try:
        # Prefer absolute image path for deriving provenance
        basis = abs_img if abs_img else img_path
        if basis:
            # If basis is absolute under image_root, compute relative
            if abs_img and os.path.commonpath([os.path.abspath(abs_img), os.path.abspath(args_image_root_global)]) == os.path.abspath(args_image_root_global):
                rel_full = os.path.relpath(abs_img, args_image_root_global)
            else:
                # Fall back to using the provided file_name; normalize separators
                rel_full = str(basis).replace('\\', '/').lstrip('/')
            parts = rel_full.replace('\\', '/').split('/')
            if parts:
                source_book = parts[0]
                source_rel_image_path = '/'.join(parts[1:]) if len(parts) > 1 else ''
    except Exception:
        # Non-fatal; leave provenance fields as None if any path issues
        pass

    # Create DataSpec page
    safe_key = _safe_page_key(image_id, img_path, abs_img)
    dataspec_page = {
        'page_id': safe_key,
        # Prefer absolute path if resolved to ensure consumers can find the image and to make keys unique
        'page_image_path': abs_img if abs_img else img_path,
        # Provenance fields for human-friendly context (like Amazon series/comic/page)
        'source_book': source_book,
        'source_rel_image_path': source_rel_image_path,
        'page_size': [img_width, img_height],
        'panels': panels_with_text,
        'reading_order': reading_order,
        'characters': characters,
        'faces': faces,
        'texts': texts
    }

    return dataspec_page, None

def convert_coco_to_dataspec(coco_file, vlm_dir, output_dir, image_root, max_samples=None, precomputed_map_path: Optional[str] = None,
                             log_unmatched: Optional[str] = None, log_errors: Optional[str] = None,
                             fail_fast_total: Optional[int] = None, fail_fast_consecutive: Optional[int] = None,
                             friendly_filenames: bool = True,
                             panel_category_ids_override: Optional[List[int]] = None,
                             panel_category_names_override: Optional[str] = None,
                             allow_no_vlm: bool = False):
    """Convert COCO detections to DataSpec format"""
    
    print("🔄 Converting CalibreComics COCO to DataSpec v0.3")
    print("=" * 60)
    
    # Load data
    image_detections, image_info, cat_id_to_name, cat_name_to_id, cat_counts = load_coco_data(coco_file)
    pre_map = _load_precomputed_map(precomputed_map_path)
    vlm_joiners = load_vlm_data(vlm_dir, image_root, pre_map)
    image_root_indexes = _build_image_indexes(image_root)
    
    # Resolve which category ids are panels
    def _parse_names(s: Optional[str]) -> List[str]:
        if not s:
            return []
        parts = [p.strip().lower() for p in re.split(r"[;,]", s) if p.strip()]
        return parts

    def _select_panel_ids(
        cat_id_to_name: Dict[int, str],
        cat_name_to_id: Dict[str, int],
        cat_counts: Dict[int, int],
        override_ids: Optional[List[int]],
        override_names: Optional[str],
    ) -> set[int]:
        # 1) explicit id overrides
        if override_ids:
            return set(int(x) for x in override_ids)
        # 2) explicit names overrides
        names = _parse_names(override_names)
        selected: set[int] = set()
        for nm in names:
            cid = cat_name_to_id.get(nm)
            if cid is not None:
                selected.add(cid)
        if selected:
            return selected
        # 3) heuristic based on common names
        candidates = {"panel", "panels", "frame", "frames", "comic_panel", "panel_box"}
        for cid, nm in cat_id_to_name.items():
            if isinstance(nm, str) and nm.lower() in candidates:
                selected.add(cid)
        if selected:
            return selected
        # 4) fallback to id=1 with a warning
        print("Warning: Could not auto-detect panel category ids by name; falling back to {1}.")
        return {1}

    global panel_category_ids_global, allow_no_vlm_global
    panel_category_ids_global = _select_panel_ids(
        cat_id_to_name, cat_name_to_id, cat_counts, panel_category_ids_override, panel_category_names_override
    )
    allow_no_vlm_global = bool(allow_no_vlm)

    print("Selected panel category id(s):", sorted(panel_category_ids_global))
    # Warn if selected ids have zero detections
    for cid in sorted(panel_category_ids_global):
        cnt = cat_counts.get(cid, 0)
        if cnt == 0:
            nm = cat_id_to_name.get(cid, '')
            print(f"  Note: selected panel id {cid} ('{nm}') has 0 detections in this COCO file.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare log writers
    unmatched_writer = None
    errors_writer = None
    unmatched_fh = None
    errors_fh = None
    try:
        if log_unmatched:
            if os.path.dirname(log_unmatched):
                os.makedirs(os.path.dirname(log_unmatched), exist_ok=True)
            unmatched_fh = open(log_unmatched, 'w', encoding='utf-8', newline='')
            unmatched_writer = csv.DictWriter(unmatched_fh, fieldnames=['reason','image_id','img_file_name','resolved_abs_img','basename','stem','normstem','lastnum_key'])
            unmatched_writer.writeheader()
        if log_errors:
            if os.path.dirname(log_errors):
                os.makedirs(os.path.dirname(log_errors), exist_ok=True)
            errors_fh = open(log_errors, 'w', encoding='utf-8', newline='')
            errors_writer = csv.DictWriter(errors_fh, fieldnames=['reason','image_id','img_file_name','exception'])
            errors_writer.writeheader()
    except Exception as e:
        print(f"Warning: could not open log files: {e}")

    # Process images
    processed_count = 0
    skipped_count = 0
    total_failures = 0
    consecutive_failures = 0
    
    print(f"Processing {len(image_detections)} images...")
    
    for image_id, detections in tqdm(image_detections.items(), desc="Converting images"):
        if max_samples and processed_count >= max_samples:
            break
            
        try:
            dataspec_page, failure = convert_detection_to_dataspec(
                image_id, detections, image_info, vlm_joiners, image_root_indexes
            )
            # Handle failure (unmatched)
            if failure:
                skipped_count += 1
                total_failures += 1
                consecutive_failures += 1
                if unmatched_writer and failure.get('reason') == 'no_vlm_match':
                    try:
                        unmatched_writer.writerow(failure)
                    except Exception as e:
                        print(f"Warning: failed to write unmatched log row: {e}")
                # Fail-fast thresholds
                if fail_fast_total and total_failures >= fail_fast_total:
                    print(f"Fail-fast: total failures reached {total_failures} (threshold {fail_fast_total}). Aborting.")
                    break
                if fail_fast_consecutive and consecutive_failures >= fail_fast_consecutive:
                    print(f"Fail-fast: consecutive failures reached {consecutive_failures} (threshold {fail_fast_consecutive}). Aborting.")
                    break
                continue

            if dataspec_page:
                # Choose output filename aligned to Amazon/Calibre pattern: Book_Book - pNNN.json
                out_base = dataspec_page['page_id']
                if friendly_filenames:
                    book_part = _sanitize_filename_part(dataspec_page.get('source_book'))
                    img_basename = os.path.splitext(os.path.basename(str(dataspec_page.get('page_image_path', 'page'))))[0]
                    num = _extract_last_number(img_basename)
                    page_label = f"p{num:03d}" if num is not None else "p000"
                    if book_part:
                        out_base = f"{book_part}_{book_part} - {page_label}"
                    else:
                        # Fallback to using the image stem as the second component
                        stem_part = _sanitize_filename_part(img_basename)
                        out_base = f"{stem_part}_{stem_part} - {page_label}"
                    # Reflect friendly id in page_id for readability
                    dataspec_page['page_id'] = out_base
                output_file = os.path.join(output_dir, f"{out_base}.json")
                # Check for duplicate output filename; treat as error and skip to avoid overwriting
                if os.path.exists(output_file):
                    # Adjust by appending a small numeric suffix to keep both without hashes
                    suffix = 1
                    adjusted = output_file
                    while os.path.exists(adjusted) and suffix <= 99:
                        adjusted = os.path.join(output_dir, f"{out_base}__dup{suffix}.json")
                        suffix += 1
                    if adjusted == output_file:
                        msg = f"duplicate output file: {output_file}"
                        print(f"[ERROR] {msg} (image_id={image_id}, img_file_name={dataspec_page.get('page_image_path','')})")
                        if errors_writer:
                            try:
                                img_info_local = image_info.get(image_id, {})
                                img_file_name = img_info_local.get('file_name', '') if isinstance(img_info_local, dict) else ''
                                errors_writer.writerow({'reason': 'duplicate_output_file', 'image_id': image_id, 'img_file_name': img_file_name, 'exception': msg})
                            except Exception as e2:
                                print(f"Warning: failed to write error log row: {e2}")
                        skipped_count += 1
                        total_failures += 1
                        consecutive_failures += 1
                        if fail_fast_total and total_failures >= fail_fast_total:
                            print(f"Fail-fast: total failures reached {total_failures} (threshold {fail_fast_total}). Aborting.")
                            break
                        if fail_fast_consecutive and consecutive_failures >= fail_fast_consecutive:
                            print(f"Fail-fast: consecutive failures reached {consecutive_failures} (threshold {fail_fast_consecutive}). Aborting.")
                            break
                        continue
                    else:
                        if errors_writer:
                            try:
                                errors_writer.writerow({'reason': 'duplicate_output_file_adjusted', 'image_id': image_id, 'img_file_name': dataspec_page.get('page_image_path',''), 'exception': f"renamed to {os.path.basename(adjusted)}"})
                            except Exception:
                                pass
                        output_file = adjusted
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataspec_page, f, indent=2)
                processed_count += 1
                consecutive_failures = 0
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"[ERROR] processing image {image_id}: {e}")
            if errors_writer:
                try:
                    img_info_local = image_info.get(image_id, {})
                    img_file_name = img_info_local.get('file_name', '') if isinstance(img_info_local, dict) else ''
                    errors_writer.writerow({'reason': 'exception', 'image_id': image_id, 'img_file_name': img_file_name, 'exception': str(e)})
                except Exception as e2:
                    print(f"Warning: failed to write error log row: {e2}")
            skipped_count += 1
            continue
    
    print(f"✅ Conversion complete!")
    print(f"  Processed: {processed_count} pages")
    print(f"  Skipped: {skipped_count} pages")
    print(f"  Output directory: {output_dir}")
    if unmatched_fh:
        unmatched_fh.close()
        print(f"  Unmatched log: {log_unmatched}")
    if errors_fh:
        errors_fh.close()
        print(f"  Errors log: {log_errors}")

def main():
    parser = argparse.ArgumentParser(description='Convert CalibreComics COCO to DataSpec v0.3')
    parser.add_argument('--coco', required=True, 
                       help='Path to COCO detection JSON file')
    parser.add_argument('--vlm_dir', required=True, 
                       help='Directory or directories containing VLM JSON files (use ; or , to separate)')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for DataSpec JSON files')
    parser.add_argument('--image_root', required=True,
                       help='Root directory for extracted images (for robust VLM matching)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--precomputed_map', type=str, default=None,
                        help='Optional json_path->image_path map (JSON or CSV) from build_precomputed_map.py')
    parser.add_argument('--log_unmatched', type=str, default=None,
                        help='CSV file to write unmatched (no VLM) pages with details')
    parser.add_argument('--log_errors', type=str, default=None,
                        help='CSV file to write exceptions and processing errors')
    parser.add_argument('--fail_fast_total', type=int, default=None,
                        help='Abort after this many total failures (unmatched or errors)')
    parser.add_argument('--fail_fast_consecutive', type=int, default=None,
                        help='Abort after this many consecutive failures')
    parser.add_argument('--no_friendly_filenames', action='store_true',
                        help='Disable friendly filenames; use hash-based unique page_id only')
    parser.add_argument('--allow_no_vlm', action='store_true',
                        help='If set, pages without VLM matches are still written using COCO panels with empty text')
    parser.add_argument('--panel_category_id', type=int, action='append', default=None,
                        help='Panel category id(s) to treat as panels; can be repeated. Overrides auto-detection.')
    parser.add_argument('--panel_category_names', type=str, default=None,
                        help='Comma/semicolon-separated category names to treat as panels (e.g., panel,frame). Overrides auto-detection if found.')
    
    args = parser.parse_args()

    global args_image_root_global
    args_image_root_global = args.image_root

    convert_coco_to_dataspec(
        args.coco,
        args.vlm_dir,
        args.output_dir,
        args.image_root,
        args.max_samples,
        args.precomputed_map,
        args.log_unmatched,
        args.log_errors,
        args.fail_fast_total,
        args.fail_fast_consecutive,
        friendly_filenames=not args.no_friendly_filenames,
        panel_category_ids_override=args.panel_category_id,
        panel_category_names_override=args.panel_category_names,
        allow_no_vlm=args.allow_no_vlm,
    )

if __name__ == "__main__":
    main()


