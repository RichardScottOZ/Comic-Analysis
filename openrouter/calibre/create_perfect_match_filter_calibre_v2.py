import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# --------------------------------------
# Shared normalization helpers
# --------------------------------------
def normalize_folder_name(name: str) -> str:
    import re
    return re.sub(r'[\s_-]+', '_', name.strip()).lower()

def normalize_image_key_from_img_path(img_path: str) -> str:
    """Normalize an image path into a key compatible with VLM keys.
    Handles generic folders (e.g., calibrecomics_extracted, jpg4cbz), derives
    a series-like folder from filename when needed, and extracts a page code.
    """
    import os, re
    # Normalize slashes and case
    img_path = img_path.replace('\\', '/').lower()
    folder_path = os.path.dirname(img_path)
    parent_path = os.path.dirname(folder_path) if folder_path else ''
    filename = os.path.basename(img_path)

    # Handle case where img_path doesn't have a directory
    if not folder_path:
        parts = img_path.split('/')
        folder_path = parts[-2] if len(parts) > 1 else ''

    # Remove all chained extensions (e.g., .jpg.png)
    base_name = re.sub(r'(\.[a-z0-9]+)+$', '', filename.lower())

    # Handle flattened paths that include generic prefixes like calibrecomics_extracted_...
    # If no directory separators in the entire path except drive colon, treat the whole name as flattened
    flattened = ('/' not in img_path and '\\' not in img_path)
    generic_prefixes = (
        'calibrecomics_extracted_',
        'calibrecomicsanalysis_',
        'calibrecomics_analysis_',
        'calibrecomics_',
        'images_',
        'extracted_',
        'jpg4cbz_',
        'pages_',
        'page_'
    )
    if flattened:
        # Try to strip any known prefix once
        for pref in generic_prefixes:
            if base_name.startswith(pref):
                base_name = base_name[len(pref):]
                break

    # Clean or derive folder name
    folder_raw = os.path.basename(folder_path) if folder_path else ''
    folder = normalize_folder_name(folder_raw)
    generic_markers = {"calibrecomics_extracted", "calibrecomics", "images", "extracted", "jpg4cbz", "pages", "page", "scans"}
    if folder in generic_markers or "extracted" in folder:
        # Derive a title from filename prefix (strip trailing page tokens)
        title = base_name
        # Drop trailing jpg4cbz/page/number suffixes
        title = re.sub(r'_(?:jpg4cbz_)?\d{1,6}$', '', title)
        title = re.sub(r'_?page[_\-]?\d{1,6}$', '', title)
        # Replace separators and trim
        title = re.sub(r'[\s\-]+', '_', title).strip('_')
        if title:
            folder = normalize_folder_name(title)
        # Prefer parent-of-parent folder if available and non-generic
        if parent_path and not flattened:
            pp = normalize_folder_name(os.path.basename(parent_path))
            if pp and pp not in generic_markers and len(pp) > 2:
                folder = pp

    # Try multiple page identification strategies
    m = re.search(r'(jpg4cbz_\d+)', base_name)
    if m:
        page_code = m.group(1)
        return f"{folder}_{page_code}"

    m = re.search(r'page[_\-\s]?(\d+)', base_name)
    if m:
        num = m.group(1)
        return f"{folder}_{num}"

    if base_name.isdigit():
        return f"{folder}_{base_name}"

    # Prefer a trailing number if present (e.g., *_0001). Otherwise use last number in the string.
    m = re.search(r'_(\d{1,6})$', base_name)
    if m:
        num = m.group(1)
        return f"{folder}_{num}"
    nums = re.findall(r'(\d{1,6})', base_name)
    if nums:
        num = nums[-1]
        return f"{folder}_{num}"

    # Fallback: whole cleaned filename
    clean_name = re.sub(r'[^a-z0-9]', '_', base_name).strip('_')
    return f"{folder}_{clean_name}"

# Globals for worker processes (to avoid per-task pickling on Windows)
_VLM_DATA = None
_VLM_INDEX = None
_VLM_EAGER = False
# Global threshold for text coverage used in per-image scoring (set by main analyzer)
_MIN_TEXT_COVERAGE = 0.8

def _init_worker(vlm_data, vlm_index, eager, min_text_cov):
    global _VLM_DATA, _VLM_INDEX, _VLM_EAGER, _MIN_TEXT_COVERAGE
    _VLM_DATA = vlm_data
    _VLM_INDEX = vlm_index
    _VLM_EAGER = eager
    _MIN_TEXT_COVERAGE = float(min_text_cov) if min_text_cov is not None else 0.8

def load_coco_data(coco_file):
    """Load COCO detection data"""
    print("Loading COCO data...")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
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
    print(f"Loaded {len(image_detections)} images with R-CNN data")
    return image_detections, image_info, categories

def load_vlm_data(vlm_dir, eager=False, vlm_map: str = None):
    """Load VLM data from directory with comprehensive key creation focused on general patterns
    Returns:
        vlm_data: dict of key -> json data
        vlm_index: dict with fast lookup structures to avoid O(N) fuzzy scans
            - by_folder_num: {(folder, num_variant): key}
            - by_num: {num_variant: set(keys)}
            - by_folder: {folder: set(keys)}
            - tokens_to_keys: {token: set(keys)} (tokens length>=3)
    """
    print("Loading VLM data...")
    vlm_data = {}  # only populated when eager=True
    # Fast lookup indexes
    by_folder_num = {}
    by_num = {}
    by_folder = {}
    tokens_to_keys = {}
    key_to_path = {}
    
    # Helper: register a single json file path with multiple key variants
    import re
    import re as _re
    
    def register_vlm_json(vlm_file_path: Path, data_for_eager=None):
        nonlocal vlm_data, key_to_path, by_folder, by_folder_num, by_num, tokens_to_keys
        # Build variations from filename and parent dir
        vlm_filename = os.path.basename(vlm_file_path)
        parent_dir = os.path.dirname(vlm_file_path)
        
        def normalize_vlm_key(filename, parent_dir=None):
            base = filename.replace('.json', '')
            normalized = re.sub(r'[\s_-]+', '_', base.strip()).lower()
            keys = [normalized]
            if parent_dir:
                parent = os.path.basename(parent_dir).lower()
                parent_norm = re.sub(r'[\s_-]+', '_', parent.strip())
                keys.append(f"{parent_norm}_{normalized}")
            # Prefer a trailing number when present; else use the last number anywhere
            m_trail = re.search(r'_(\d{1,6})$', normalized)
            if m_trail:
                num = m_trail.group(1)
            else:
                nums_any = re.findall(r'(\d{1,6})', normalized)
                num = nums_any[-1] if nums_any else None
            if num:
                if parent_dir:
                    parent = re.sub(r'[\s_-]+', '_', os.path.basename(parent_dir).lower())
                    keys.append(f"{parent}_{num}")
                    keys.append(f"{parent}_{num.zfill(3)}")
                keys.append(num)
                keys.append(num.zfill(3))
                if "page" in normalized:
                    page_match = re.search(r'page[_\-\s]*(\d+)', normalized)
                    if page_match and parent_dir:
                        parent = re.sub(r'[\s_-]+', '_', os.path.basename(parent_dir).lower())
                        keys.append(f"{parent}_{page_match.group(1)}")
            if parent_dir:
                rel_path = f"{os.path.basename(parent_dir)}/{base}"
                clean_path = re.sub(r'[\s_-]+', '_', rel_path.lower())
                keys.append(clean_path)
            return keys
        
        key_variations = normalize_vlm_key(vlm_filename, parent_dir)
        # Relative path variation
        rel_path = os.path.relpath(vlm_file_path, vlm_dir).replace('\\', '/') if os.path.isdir(vlm_dir) else os.path.basename(vlm_file_path)
        rel_key = rel_path.replace('.json', '').lower()
        rel_key = re.sub(r'[\s_-]+', '_', rel_key)
        key_variations.append(rel_key)

        main_key = key_variations[0]
        if eager and data_for_eager is not None:
            vlm_data[main_key] = data_for_eager
        key_to_path[main_key] = str(vlm_file_path)
        for key in key_variations[1:]:
            if eager and data_for_eager is not None and key not in vlm_data:
                vlm_data[key] = data_for_eager
            key_to_path.setdefault(key, str(vlm_file_path))
        
        # Index informative key shapes, including trailing _pNNN variants
        def _index_from_key(key_str, primary_key_ref: str):
            folder = None
            num = None
            hash_case = False
            # jpg4cbz folder_num variant
            if "jpg4cbz_" in key_str:
                parts = key_str.split("_jpg4cbz_")
                if len(parts) == 2:
                    folder = parts[0]
                    # Sanitize numeric part to digits only (handle cases like 066-)
                    import re as __re
                    mnum = __re.search(r"(\d{1,6})", parts[1])
                    num = mnum.group(1) if mnum else None
            if folder is None:
                # Allow optional trailing punctuation after digits
                m = _re.search(r"^(.*)_(\d{1,6})(?:[^a-z0-9].*)?$", key_str)
                if m:
                    folder = m.group(1)
                    num = m.group(2)
            if folder is None:
                # Allow optional trailing punctuation after digits
                m = _re.search(r"^(.*)_page[_\-]?(\d{1,6})(?:[^a-z0-9].*)?$", key_str)
                if m:
                    folder = m.group(1)
                    num = m.group(2)
            if folder is None:
                # NEW: support hash-delimited numbers, e.g., MockingDeadVol1#0001 (with optional trailing chars)
                m = _re.search(r"^(.*)#(\d{1,6})(?:[^a-z0-9].*)?$", key_str)
                if m:
                    folder = m.group(1)
                    num = m.group(2)
                    hash_case = True
            if folder is None:
                # NEW: match trailing _pNNN pattern common in perfect_match_jsons (with optional trailing chars)
                m = _re.search(r"^(.*)_p(\d{1,6})(?:[^a-z0-9].*)?$", key_str)
                if m:
                    folder = m.group(1)
                    num = m.group(2)
            if folder:
                # Broaden normalization: collapse any non-alphanumeric to underscores
                folder_norm = _re.sub(r"[^a-z0-9]+", "_", folder.strip().lower()).strip("_")
                # Build folder variants to improve recall:
                #  - original folder
                #  - collapsed duplicated suffix (e.g., x_y_x_y -> x_y)
                #  - last-half only if suffix repeats
                #  - parent+last-half composite if present in key
                variants_folders = [folder_norm]
                toks = [t for t in folder_norm.split('_') if t]
                collapsed = None
                just_last = None
                if len(toks) >= 2:
                    # Detect repeated suffix halves of any length
                    for half_len in range(1, len(toks)//2 + 1):
                        if toks[-2*half_len:-half_len] == toks[-half_len:]:
                            # Remove one repetition
                            collapsed = toks[:-half_len]
                            just_last = toks[-half_len:]
                            break
                if collapsed:
                    collapsed_name = '_'.join(collapsed)
                    if collapsed_name and collapsed_name not in variants_folders:
                        variants_folders.append(collapsed_name)
                if just_last:
                    last_name = '_'.join(just_last)
                    if last_name and last_name not in variants_folders:
                        variants_folders.append(last_name)
                # For hash-case keys (e.g., ..._mockingdeadvol1#0001), also index the last token alone
                if hash_case and len(toks) >= 1:
                    last_tok = toks[-1]
                    if last_tok and last_tok not in variants_folders:
                        variants_folders.append(last_tok)
                # Also index parent+last if the key_str starts with a composite and we detected a last-half
                if just_last:
                    # Heuristic parent is the prefix tokens before the repeated last-half (if any)
                    parent_tokens = toks[:-2*len(just_last)] if len(toks) >= 2*len(just_last) else toks[:-len(just_last)]
                    if parent_tokens:
                        combo = '_'.join(parent_tokens + just_last)
                        if combo and combo not in variants_folders:
                            variants_folders.append(combo)
                # Index variants
                for fvar in variants_folders:
                    by_folder.setdefault(fvar, set()).add(primary_key_ref)
                    for tok in fvar.split("_"):
                        if len(tok) >= 3:
                            tokens_to_keys.setdefault(tok, set()).add(primary_key_ref)
                    if num:
                        variants = set([num, num.lstrip('0') or '0'])
                        for z in (3, 4, 5):
                            variants.add(num.zfill(z))
                        for v in variants:
                            by_folder_num[(fvar, v)] = primary_key_ref
                            by_num.setdefault(v, set()).add(primary_key_ref)
            else:
                # Broad rescue: any number anywhere in key
                m2 = _re.search(r"(\d{1,6})", key_str)
                if m2:
                    num = m2.group(1)
                    variants = set([num, num.lstrip('0') or '0'])
                    for z in (3, 4, 5):
                        variants.add(num.zfill(z))
                    for v in variants:
                        by_num.setdefault(v, set()).add(primary_key_ref)
        _index_from_key(main_key, main_key)
        # Index all generated variations to capture container/relative forms
        for kv in key_variations:
            _index_from_key(kv, main_key)

    # Branch 1: load via mapping JSON (preferred for Calibre)
    if vlm_map and os.path.isfile(vlm_map):
        print(f"Using VLM mapping file: {vlm_map}")
        try:
            with open(vlm_map, 'r', encoding='utf-8') as f:
                mapping_obj = json.load(f)
        except Exception as e:
            print(f"Failed to read VLM map {vlm_map}: {e}")
            mapping_obj = None
        count = 0
        if isinstance(mapping_obj, dict):
            # Supported shapes:
            #  A) {json_path: image_path}  [produced by build_precomputed_map.py]
            #  B) {image_path: json_path}
            #  C) {key: json_path}
            #  D) {image_path: {json_path: ..}}
            for k, v in mapping_obj.items():
                img_path = None
                json_path = None
                # Case A: key is json, value is image
                if isinstance(k, str) and k.lower().endswith('.json') and isinstance(v, str):
                    json_path = k
                    img_path = v
                # Case B/C: value is json path, key is image path or pre-normalized key
                elif isinstance(v, str) and v.lower().endswith('.json'):
                    json_path = v
                    if ('/' in k) or ('\\' in k):
                        img_path = k
                    else:
                        # Treat k as a pre-normalized key
                        key_to_path[k] = json_path
                        register_vlm_json(Path(json_path))
                        count += 1
                        continue
                # Case D: dict object
                elif isinstance(v, dict):
                    json_path = v.get('json_path') or v.get('path') or v.get('vlm')
                    img_path = v.get('image_path') or v.get('image') or v.get('img') or k

                if json_path:
                    # Always register the JSON into the index so by_folder_num/by_num work
                    try:
                        register_vlm_json(Path(json_path))
                    except Exception:
                        pass
                if json_path and img_path:
                    # Map normalized image key -> json path for direct lookup
                    key = normalize_image_key_from_img_path(img_path)
                    key_to_path[key] = json_path
                    count += 1
        elif isinstance(mapping_obj, list):
            # List of json paths or objects with image/json fields
            for item in mapping_obj:
                if isinstance(item, str) and item.lower().endswith('.json'):
                    register_vlm_json(Path(item))
                    count += 1
                elif isinstance(item, dict):
                    json_path = item.get('json_path') or item.get('path') or item.get('vlm')
                    img_path = item.get('image_path') or item.get('image') or item.get('img')
                    if json_path:
                        if img_path:
                            key = normalize_image_key_from_img_path(img_path)
                            key_to_path[key] = json_path
                        register_vlm_json(Path(json_path))
                        count += 1
        print(f"Registered {count} VLM entries from map")
        vlm_files = []
    else:
        # Branch 2: scan directory (original behavior)
        # Search in both the main directory and recursively in subdirectories
        vlm_files = list(Path(vlm_dir).glob("**/*.json"))  # Include subdirectories
        print(f"Found {len(vlm_files)} VLM files (including subdirectories)")
    
    debug_printed = 0
    for vlm_file in tqdm(vlm_files, desc="Loading VLM files"):
        try:
            data = None
            if eager:
                with open(vlm_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            if debug_printed < 3:
                print(f"\n--- DEBUG VLM FILE [{vlm_file}] ---")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                elif isinstance(data, list):
                    print(f"List length: {len(data)}")
                    if len(data) > 0 and isinstance(data[0], dict):
                        print(f"First item keys: {list(data[0].keys())}")
                else:
                    print(f"Type: {type(data)}")
                debug_printed += 1

            register_vlm_json(Path(vlm_file), data)
        except Exception as e:
            print(f"Error loading {vlm_file}: {e}")
            continue

    loaded_count = len(vlm_files) if vlm_files else len(key_to_path)
    print(f"Loaded {loaded_count} VLM entries into {len(key_to_path)} lookup keys")
    
    # Debug: print sample keys to understand key format (from index)
    keys_sample = list(key_to_path.keys())[:20]
    print("\nSample of VLM data keys:")
    for i, k in enumerate(keys_sample):
        print(f"{i+1}. {k}")
        
    # Analyze key patterns to understand common formats
    key_patterns = {
        "numeric_only": 0,
        "folder_number": 0,
        "page_pattern": 0,
        "jpg4cbz_pattern": 0,
        "other": 0
    }
    
    for key in key_to_path.keys():
        if key.isdigit() or re.match(r'^\d+$', key):
            key_patterns["numeric_only"] += 1
        elif re.search(r'_\d+$', key):
            key_patterns["folder_number"] += 1
        elif "page" in key:
            key_patterns["page_pattern"] += 1
        elif "jpg4cbz" in key:
            key_patterns["jpg4cbz_pattern"] += 1
        else:
            key_patterns["other"] += 1
    
    print("\nVLM key pattern distribution:")
    for pattern, count in key_patterns.items():
        print(f"  {pattern}: {count} keys ({count/max(1,len(key_to_path))*100:.1f}%)")
    
    # Initialize a log file for successful matches to analyze patterns
    with open("key_matches.log", "w", encoding="utf-8") as f:
        f.write("# Successful key matches log\n")
        f.write("original_path,normalized_key,match_type\n")
        
    # Build and return indexes
    vlm_index = {
        'by_folder_num': by_folder_num,
        'by_num': by_num,
        'by_folder': by_folder,
        'tokens_to_keys': tokens_to_keys,
        'key_to_path': key_to_path,
        'eager': eager,
    }
    return vlm_data, vlm_index

def analyze_image_alignment(args):
    # Normalization function - EXACTLY matching the one in load_vlm_data
    import re, os
    
    def normalize_image_key(img_path):
        return normalize_image_key_from_img_path(img_path)
    """Analyze alignment for a single image (for multiprocessing)"""
    # args: (image_id, detections, image_info, allow_partial_match, fast_only, fuzzy_max_candidates, verbose_fuzzy, log_misses, panel_category_ids, keep_unmatched, first_failure, image_roots, require_image_exists)
    image_id, detections, image_info, allow_partial_match, fast_only, fuzzy_max_candidates, verbose_fuzzy, log_misses, panel_category_ids, keep_unmatched, first_failure, image_roots, require_image_exists = args
    try:
        # Safely resolve the image path from COCO image_info or fallbacks
        img_path = None
        img_info = None
        if isinstance(image_info, dict) and image_id in image_info and isinstance(image_info[image_id], dict):
            img_info = image_info[image_id]
            img_path = img_info.get('file_name')
        # Some COCO exports use string image_id equal to the file path/name
        if not img_path and isinstance(image_id, str):
            img_path = image_id
        if not img_path:
            raise ValueError("Could not resolve image file_name for image_id")
        import re
        def normalize_folder_name(name):
            # Remove trailing spaces, unify dashes/underscores, lowercase
            return re.sub(r'[\s_-]+', '_', name.strip()).lower()

        def normalize_image_key(img_path):
            """Create a key from image path using general comic naming patterns.
            If the parent folder is a generic container (e.g., calibrecomics_extracted), derive the series folder from the filename prefix.
            """
            import os, re

            # Normalize slashes and case
            img_path = img_path.replace('\\', '/').lower()
            folder_path = os.path.dirname(img_path)
            parent_path = os.path.dirname(folder_path) if folder_path else ''
            filename = os.path.basename(img_path)

            # Handle case where img_path doesn't have a directory
            if not folder_path:
                parts = img_path.split('/')
                folder_path = parts[-2] if len(parts) > 1 else ''

            # Remove all chained extensions (e.g., .jpg.png)
            base_name = re.sub(r'(\.[a-z0-9]+)+$', '', filename.lower())

            # NEW: Support flattened file_names (no directory separators) that include generic prefixes
            flattened = ('/' not in img_path and '\\' not in img_path)
            if flattened:
                for pref in ('calibrecomics_extracted_', 'calibrecomicsanalysis_', 'calibrecomics_analysis_', 'calibrecomics_', 'images_', 'extracted_', 'jpg4cbz_', 'pages_', 'page_'):
                    if base_name.startswith(pref):
                        base_name = base_name[len(pref):]
                        break

            # Clean or derive folder name
            folder_raw = os.path.basename(folder_path) if folder_path else ''
            folder = normalize_folder_name(folder_raw)
            generic_markers = {"calibrecomics_extracted", "calibrecomics", "images", "extracted", "jpg4cbz", "pages", "page", "scans"}
            if folder in generic_markers or "extracted" in folder:
                # Derive a title from filename prefix (strip trailing page tokens)
                title = base_name
                # Drop trailing jpg4cbz/page/number suffixes
                title = re.sub(r'_(?:jpg4cbz_)?\d{1,6}$', '', title)
                title = re.sub(r'_?page[_\-]?\d{1,6}$', '', title)
                # Replace separators and trim
                title = re.sub(r'[\s\-]+', '_', title).strip('_')
                if title:
                    folder = normalize_folder_name(title)
                # Prefer parent-of-parent folder if available and non-generic
                if parent_path and not flattened:
                    pp = normalize_folder_name(os.path.basename(parent_path))
                    if pp and pp not in generic_markers and len(pp) > 2:
                        folder = pp

            # Try multiple page identification strategies
            m = re.search(r'(jpg4cbz_\d+)', base_name)
            if m:
                page_code = m.group(1)
                return f"{folder}_{page_code}"

            m = re.search(r'page[_\-\s]?(\d+)', base_name)
            if m:
                num = m.group(1)
                # keep raw; index lookup will try multiple zero-fill variants
                return f"{folder}_{num}"

            if base_name.isdigit():
                return f"{folder}_{base_name}"

            # Prefer trailing number; else last number
            m = re.search(r'_(\d{1,6})$', base_name)
            if m:
                num = m.group(1)
                return f"{folder}_{num}"
            nums = re.findall(r'(\d{1,6})', base_name)
            if nums:
                num = nums[-1]
                return f"{folder}_{num}"

            # Fallback: whole cleaned filename
            clean_name = re.sub(r'[^a-z0-9]', '_', base_name).strip('_')
            return f"{folder}_{clean_name}"
        # Build a better path for normalization when file_name is under a generic folder (e.g., JPG4CBZ\0001.jpg)
        def _norm(s):
            return re.sub(r'[\s_-]+','_', s.strip()).lower()
        id_parent_folder = None
        try:
            id_path_norm = str(image_id).replace('\\','/').lower()
            id_parent_folder = os.path.basename(os.path.dirname(id_path_norm)) if '/' in id_path_norm else None
        except Exception:
            id_parent_folder = None
        generic_folders_local = {"jpg4cbz", "pages", "page", "images", "scans", "calibrecomics_extracted", "calibrecomics", "extracted"}
        img_path_norm = img_path.replace('\\','/').lower()
        img_dirname = os.path.dirname(img_path_norm)
        img_dirbase = os.path.basename(img_dirname) if img_dirname else ''
        img_dirbase_norm = _norm(img_dirbase) if img_dirbase else ''
        # If we only have a generic container or no folder, synthesize a path with the parent folder from image_id
        if (not img_dirbase_norm) or (img_dirbase_norm in generic_folders_local):
            if id_parent_folder:
                synth = f"{id_parent_folder}/{os.path.basename(img_path_norm)}"
                normalized_path = normalize_image_key(synth)
            else:
                normalized_path = normalize_image_key(img_path)
        else:
            normalized_path = normalize_image_key(img_path)

        # Attempt to resolve an actual image file on disk for training (optional)
        def resolve_image_path(img_path_str: str, id_parent: str, roots: list):
            try:
                # If absolute and exists, return as-is
                if isinstance(img_path_str, str) and os.path.isabs(img_path_str) and os.path.exists(img_path_str):
                    return img_path_str
                candidates = []
                # Normalized forms
                img_path_norm2 = (img_path_str or '').replace('\\','/')
                filename_only = os.path.basename(img_path_norm2)
                dirname_only = os.path.dirname(img_path_norm2)
                # Extract numeric for common page patterns
                base_no_ext = re.sub(r'(\.[a-z0-9]+)+$', '', filename_only.lower())
                mnum = re.search(r'(\d{1,6})', base_no_ext)
                num = mnum.group(1) if mnum else None
                # Build variations of names to try
                def name_variants(n):
                    if not n:
                        return []
                    v = {n, n.lstrip('0') or '0', n.zfill(3), n.zfill(4)}
                    return list(v)
                num_variants = name_variants(num)
                # Possible subfolders
                subfolders = []
                dn = os.path.basename(dirname_only).lower()
                if dn:
                    subfolders.append(dn)
                # common containers
                for sf in ['JPG4CBZ','jpg4cbz','pages','Pages']:
                    if sf.lower() not in [s.lower() for s in subfolders]:
                        subfolders.append(sf)
                # Derive container (grandparent of image_id) if present
                container_name = None
                try:
                    idp = str(image_id).replace('\\','/')
                    gp = os.path.basename(os.path.dirname(os.path.dirname(idp))) if '/' in idp else None
                    if gp:
                        container_name = re.sub(r'[\s_-]+','_', gp.strip()).lower()
                except Exception:
                    container_name = None
                # Try with provided roots
                for root in (roots or []):
                    # 1) root + id_parent + original relative path if any
                    if id_parent:
                        if dirname_only:
                            candidates.append(os.path.join(root, id_parent, dirname_only, filename_only))
                        candidates.append(os.path.join(root, id_parent, filename_only))
                    # 2) root + just original relative path
                    if dirname_only:
                        candidates.append(os.path.join(root, dirname_only, filename_only))
                    candidates.append(os.path.join(root, filename_only))
                    # 2b) root + container + id_parent + path variants
                    if container_name and id_parent:
                        if dirname_only:
                            candidates.append(os.path.join(root, container_name, id_parent, dirname_only, filename_only))
                        candidates.append(os.path.join(root, container_name, id_parent, filename_only))
                    if container_name and dirname_only:
                        candidates.append(os.path.join(root, container_name, dirname_only, filename_only))
                    if container_name:
                        candidates.append(os.path.join(root, container_name, filename_only))
                    # 3) Heuristic: root + id_parent + subfolder + number with common extensions
                    if id_parent and num_variants:
                        for sf in subfolders:
                            for nv in num_variants:
                                for ext in ['.jpg', '.jpeg', '.png']:
                                    candidates.append(os.path.join(root, id_parent, sf, nv + ext))
                                    # Overlay variants occasionally present
                                    candidates.append(os.path.join(root, id_parent, sf, nv + ext + '.png'))
                                    candidates.append(os.path.join(root, id_parent, sf, nv + ext + '.g.png'))
                                    if container_name:
                                        candidates.append(os.path.join(root, container_name, id_parent, sf, nv + ext))
                                        candidates.append(os.path.join(root, container_name, id_parent, sf, nv + ext + '.png'))
                                        candidates.append(os.path.join(root, container_name, id_parent, sf, nv + ext + '.g.png'))
                for c in candidates:
                    if os.path.exists(c):
                        return c
            except Exception:
                pass
            return None

        resolved_image_path = resolve_image_path(img_path, id_parent_folder, image_roots)

        # Helper to resolve a key into JSON data (lazy or eager) and remember the key used
        vlm_key_used = None
        def resolve_vlm_by_key(key):
            nonlocal vlm_key_used
            if key is None:
                return None
            # Eager path: data in memory
            if _VLM_DATA is not None and key in _VLM_DATA:
                vlm_key_used = key
                return _VLM_DATA[key]
            # Lazy path: load from file if mapped
            path = _VLM_INDEX['key_to_path'].get(key)
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    vlm_key_used = key
                    return data
                except Exception:
                    return None
            return None

        # Try exact key match (on either data or path map)
        vlm_match = None
        if (_VLM_DATA is not None and normalized_path in _VLM_DATA) or (normalized_path in _VLM_INDEX['key_to_path']):
            vlm_match = resolve_vlm_by_key(normalized_path)
        partial_vlm_key = None

        # Fast tiered matching using indexes
        if not vlm_match:
            dirname = os.path.dirname(img_path)
            folder_name = os.path.basename(dirname).lower()
            # Use broad normalization (match VLM index): collapse non-alphanumerics
            def _norm_folder_token(s: str) -> str:
                return re.sub(r'[^a-z0-9]+', '_', (s or '').strip().lower()).strip('_')
            folder_norm = _norm_folder_token(folder_name)
            parent_dir = os.path.basename(os.path.dirname(dirname)).lower() if dirname else ''
            parent_norm = _norm_folder_token(parent_dir) if parent_dir else ''
            generic_folders = {"jpg4cbz", "pages", "page", "images", "scans", "calibrecomics_extracted", "calibrecomics", "extracted"}
            # Also consider the parent folder derived from image_id when file_name is generic/flattened
            id_parent_norm = _norm_folder_token(id_parent_folder) if id_parent_folder else None
            # Choose preferred series folder: if current folder is generic, use parent; else use current folder
            preferred_series = None
            if folder_norm and folder_norm not in generic_folders:
                preferred_series = folder_norm
            elif parent_norm and parent_norm not in generic_folders:
                preferred_series = parent_norm
            # Prepare folder candidates with preferred series first
            folder_candidates = []
            # Highest priority: id_parent_norm when available (helps when file_name path is generic like JPG4CBZ/page)
            if id_parent_norm and id_parent_norm not in generic_folders and id_parent_norm not in folder_candidates:
                folder_candidates.append(id_parent_norm)
            if preferred_series:
                folder_candidates.append(preferred_series)
            # Add immediate folder if meaningful and not already included
            if folder_norm and folder_norm not in folder_candidates and folder_norm not in generic_folders:
                folder_candidates.append(folder_norm)
            # Add non-generic parent as fallback
            if parent_norm and parent_norm not in folder_candidates and parent_norm not in generic_folders:
                folder_candidates.append(parent_norm)
            # Composite: parent (from image_id) + preferred series (from file_name) for cases like 'revival_vol1_update_unknown_revival_vol1'
            if id_parent_norm and preferred_series and id_parent_norm != preferred_series:
                combo = f"{id_parent_norm}_{preferred_series}"
                if combo not in folder_candidates:
                    folder_candidates.append(combo)
            # Add container-prefixed candidates from image_id grandparent (e.g., bundle_extracted)
            id_path_for_container = str(image_id).replace('\\','/').lower()
            container_folder = os.path.basename(os.path.dirname(os.path.dirname(id_path_for_container))) if '/' in id_path_for_container else ''
            container_norm = _norm_folder_token(container_folder) if container_folder else ''
            if container_norm and container_norm not in generic_folders:
                base_candidates = list(folder_candidates) or []
                for bc in base_candidates:
                    combo = f"{container_norm}_{bc}"
                    if combo not in folder_candidates:
                        folder_candidates.append(combo)
            # Derive title from filename for generic containers and add as candidate
            if folder_norm in generic_folders:
                base_filename_full = os.path.basename(img_path).lower()
                base_filename = re.sub(r'(\.[a-z0-9]+)+$', '', base_filename_full)
                # Drop trailing jpg4cbz/page/number suffixes
                title = re.sub(r'_(?:jpg4cbz_)?\d{1,6}$', '', base_filename)
                title = re.sub(r'_?page[_\-]?\d{1,6}$', '', title)
                # Normalize title broadly to align with index
                title = re.sub(r'[^a-z0-9]+', '_', title).strip('_')
                # Compress duplicated halves like 'x_y_x_y' -> 'x_y'
                parts = [p for p in title.split('_') if p]
                if len(parts) % 2 == 0 and len(parts) > 0:
                    half = len(parts) // 2
                    if parts[:half] == parts[half:]:
                        title = '_'.join(parts[:half])
                if title and title not in folder_candidates:
                    folder_candidates.insert(0, title)
            # Also collapse duplicated trailing halves of non-generic folder itself (e.g., foo_bar_foo_bar -> foo_bar)
            if preferred_series and '_' in preferred_series:
                toks = [t for t in preferred_series.split('_') if t]
                for half_len in range(1, len(toks)//2 + 1):
                    if toks[-2*half_len:-half_len] == toks[-half_len:]:
                        collapsed = '_'.join(toks[:-half_len])
                        last_half = '_'.join(toks[-half_len:])
                        if collapsed and collapsed not in folder_candidates:
                            folder_candidates.append(collapsed)
                        if last_half and last_half not in folder_candidates:
                            folder_candidates.append(last_half)
                        break
            # Add filename prefix-before-number as an additional candidate (helps for names like X_Y_0001)
            base_filename_full = os.path.basename(img_path).lower()
            base_filename = re.sub(r'(\.[a-z0-9]+)+$', '', base_filename_full)
            # Detect prefix before trailing number with underscore, hyphen, or space separator
            mpref = re.match(r'(.+?)[_\-\s](\d{1,6})$', base_filename)
            if mpref:
                # Normalize prefix broadly (remove punctuation like commas)
                prefix = re.sub(r'[^a-z0-9]+', '_', mpref.group(1).strip().lower()).strip('_')
                if prefix:
                    # Compress duplicated halves like 'x_y_x_y' -> 'x_y'
                    parts = [p for p in prefix.split('_') if p]
                    collapsed = None
                    if len(parts) % 2 == 0 and len(parts) > 0:
                        half = len(parts) // 2
                        if parts[:half] == parts[half:]:
                            collapsed = '_'.join(parts[:half])
                    if prefix not in folder_candidates:
                        folder_candidates.append(prefix)
                    if collapsed and collapsed not in folder_candidates:
                        folder_candidates.append(collapsed)
                # Also consider combining the series folder (from file_name) with the prefix
                if preferred_series:
                    combo = f"{preferred_series}_{prefix}"
                    if combo not in folder_candidates:
                        folder_candidates.append(combo)
                    if collapsed:
                        combo2 = f"{preferred_series}_{collapsed}"
                        if combo2 not in folder_candidates:
                            folder_candidates.append(combo2)
                # Combine parent series (from image_id) with prefix and with (series+prefix)
                if id_parent_norm:
                    pcombo = f"{id_parent_norm}_{prefix}"
                    if pcombo not in folder_candidates:
                        folder_candidates.append(pcombo)
                    if collapsed:
                        pcombo2 = f"{id_parent_norm}_{collapsed}"
                        if pcombo2 not in folder_candidates:
                            folder_candidates.append(pcombo2)
                    # Some keys duplicate the parent token: parent_parent_prefix
                    ppdup = f"{id_parent_norm}_{id_parent_norm}_{prefix}"
                    if ppdup not in folder_candidates:
                        folder_candidates.append(ppdup)
                    if collapsed:
                        ppdup2 = f"{id_parent_norm}_{id_parent_norm}_{collapsed}"
                        if ppdup2 not in folder_candidates:
                            folder_candidates.append(ppdup2)
                    if preferred_series:
                        ppcombo = f"{id_parent_norm}_{preferred_series}_{prefix}"
                        if ppcombo not in folder_candidates:
                            folder_candidates.append(ppcombo)
                        if collapsed:
                            ppcombo2 = f"{id_parent_norm}_{preferred_series}_{collapsed}"
                            if ppcombo2 not in folder_candidates:
                                folder_candidates.append(ppcombo2)
            else:
                # Fallback: hash-delimited number (e.g., MockingDeadVol1#0001)
                mh = re.match(r'(.+?)#(\d{1,6})$', base_filename)
                if mh:
                    hpref = re.sub(r'[^a-z0-9]+', '_', mh.group(1).strip().lower()).strip('_')
                    if hpref and hpref not in folder_candidates:
                        folder_candidates.append(hpref)
                    if preferred_series:
                        combo = f"{preferred_series}_{hpref}"
                        if combo not in folder_candidates:
                            folder_candidates.append(combo)
                    if id_parent_norm:
                        pcombo = f"{id_parent_norm}_{hpref}"
                        if pcombo not in folder_candidates:
                            folder_candidates.append(pcombo)
                        # Duplicate parent variant
                        ppdup = f"{id_parent_norm}_{id_parent_norm}_{hpref}"
                        if ppdup not in folder_candidates:
                            folder_candidates.append(ppdup)
            # After all candidates are assembled, prepend container to every candidate if available
            if container_norm:
                existing = list(folder_candidates)
                for bc in existing:
                    combo = f"{container_norm}_{bc}"
                    if combo not in folder_candidates:
                        folder_candidates.append(combo)
            filename = os.path.basename(img_path)
            base_filename = os.path.splitext(filename)[0].lower()
            # Prefer trailing number; else last number anywhere
            m_trail = re.search(r'_(\d+)$', base_filename)
            if m_trail:
                number = m_trail.group(1)
            else:
                nums_any = re.findall(r'(\d+)', base_filename)
                number = nums_any[-1] if nums_any else None
            # Try folder+number direct (with potential off-by-one adjustment)
            if number:
                variants = [number, number.lstrip('0') or '0', number.zfill(3), number.zfill(4), number.zfill(5)]
                found = False
                for cand_folder in folder_candidates:
                    for v in variants:
                        key = _VLM_INDEX['by_folder_num'].get((cand_folder, v))
                        if key:
                            vlm_match = resolve_vlm_by_key(key)
                            found = True
                            if verbose_fuzzy:
                                print(f"Indexed match (folder+num): {img_path} → {key}")
                            break
                        # Off-by-one: some datasets use p000 vs jpg4cbz_0001
                        try:
                            v_int = int(v)
                            if v_int > 0:
                                v_minus = str(v_int - 1)
                                key = _VLM_INDEX['by_folder_num'].get((cand_folder, v_minus))
                                if key:
                                    vlm_match = resolve_vlm_by_key(key)
                                    found = True
                                    if verbose_fuzzy:
                                        print(f"Indexed match (folder+num, -1): {img_path} → {key}")
                                    break
                        except ValueError:
                            pass
                    if found:
                        break
            # Try jpg4cbz pattern
            if not vlm_match:
                m = re.search(r'(jpg4cbz_\d+)', base_filename)
                if m:
                    page_num = m.group(1).split('_')[-1]
                    for cand_folder in folder_candidates:
                        # Try exact number
                        key = _VLM_INDEX['by_folder_num'].get((cand_folder, page_num))
                        if key:
                            vlm_match = resolve_vlm_by_key(key)
                            if verbose_fuzzy:
                                print(f"Indexed match (jpg4cbz): {img_path} → {key}")
                            break
                        # Try -1 variant for p000 vs 0001 differences
                        try:
                            v_int = int(page_num)
                            if v_int > 0:
                                v_minus = str(v_int - 1)
                                key = _VLM_INDEX['by_folder_num'].get((cand_folder, v_minus))
                                if key:
                                    vlm_match = resolve_vlm_by_key(key)
                                    if verbose_fuzzy:
                                        print(f"Indexed match (jpg4cbz -1): {img_path} → {key}")
                                    break
                        except ValueError:
                            pass
                # If the path is under JPG4CBZ/NNNN but filename itself lacks the jpg4cbz_ prefix, infer number from dirname
                if not vlm_match and folder_norm in generic_folders:
                    try:
                        dir_parts = img_path.replace('\\','/').split('/')
                        # Look for a numeric file in a JPG4CBZ folder
                        if len(dir_parts) >= 2 and dir_parts[-2].lower() in { 'jpg4cbz', 'pages', 'page' }:
                            base_no_ext = re.sub(r'(\.[a-z0-9]+)+$', '', os.path.basename(img_path).lower())
                            mnum = re.search(r'(\d{1,6})', base_no_ext)
                            if mnum:
                                page_num = mnum.group(1)
                                for cand_folder in folder_candidates:
                                    key = _VLM_INDEX['by_folder_num'].get((cand_folder, page_num))
                                    if key:
                                        vlm_match = resolve_vlm_by_key(key)
                                        if verbose_fuzzy:
                                            print(f"Indexed match (jpg4cbz inferred): {img_path} → {key}")
                                        break
                                    try:
                                        v_int = int(page_num)
                                        if v_int > 0:
                                            v_minus = str(v_int - 1)
                                            key = _VLM_INDEX['by_folder_num'].get((cand_folder, v_minus))
                                            if key:
                                                vlm_match = resolve_vlm_by_key(key)
                                                if verbose_fuzzy:
                                                    print(f"Indexed match (jpg4cbz inferred -1): {img_path} → {key}")
                                                break
                                    except ValueError:
                                        pass
                    except Exception:
                        pass
            # Try composite candidate: for any candidate that contains an underscore with a subprefix (e.g., series_2000ad_regened)
            if not vlm_match and number:
                for cand_folder in folder_candidates:
                    if '_' in cand_folder:
                        key = _VLM_INDEX['by_folder_num'].get((cand_folder, number))
                        if key:
                            vlm_match = resolve_vlm_by_key(key)
                            if verbose_fuzzy:
                                print(f"Indexed match (composite folder+num): {img_path} → {key}")
                            break
                # also try -1 off-by-one with composites
                if not vlm_match:
                    try:
                        v_int = int(number)
                        if v_int > 0:
                            v_minus = str(v_int - 1)
                            for cand_folder in folder_candidates:
                                if '_' in cand_folder:
                                    key = _VLM_INDEX['by_folder_num'].get((cand_folder, v_minus))
                                    if key:
                                        vlm_match = resolve_vlm_by_key(key)
                                        if verbose_fuzzy:
                                            print(f"Indexed match (composite folder+num, -1): {img_path} → {key}")
                                        break
                    except ValueError:
                        pass

            # Targeted fallback for hash-number filenames like 'MockingDeadVol1#0001.jpg'
            if not vlm_match:
                try:
                    mh = re.match(r'(.+?)#(\d{1,6})$', base_filename)
                    if mh:
                        hpref = re.sub(r'[^a-z0-9]+', '_', mh.group(1).strip().lower()).strip('_')
                        hnum = mh.group(2)
                        hvariants = [hnum, hnum.lstrip('0') or '0', hnum.zfill(3), hnum.zfill(4), hnum.zfill(5)]
                        # Build a small pool of token sets to query
                        tok_pool = set([t for t in [hpref, id_parent_norm, preferred_series, container_norm] if t])
                        # Also include all candidate folder tokens to maximize recall
                        for cf in (folder_candidates or []):
                            toks = [t for t in re.sub(r"[^a-z0-9]+", "_", cf).split('_') if len(t) >= 3]
                            tok_pool.update(toks)
                        picked_key = None
                        for tok in tok_pool:
                            cand_keys = _VLM_INDEX['tokens_to_keys'].get(tok, set())
                            if not cand_keys:
                                continue
                            for v in hvariants:
                                # Prefer exact '#<v>' suffix, else any occurrence
                                for k in cand_keys:
                                    if k.endswith(f"#{v}") or f"#{v}" in k:
                                        picked_key = k
                                        break
                                if picked_key:
                                    break
                            if picked_key:
                                break
                        if picked_key:
                            vlm_match = resolve_vlm_by_key(picked_key)
                            if verbose_fuzzy:
                                print(f"Hash fallback match: {img_path} → {picked_key}")
                except Exception:
                    pass

            # Try by number candidates filtered by tokens from all folder candidates + parent/container
            if not vlm_match and number and not fast_only:
                variants = [number, number.lstrip('0') or '0', number.zfill(3), number.zfill(4), number.zfill(5)]
                # Aggregate tokens from every useful source
                def _tokens(s):
                    return [t for t in re.sub(r"[^a-z0-9]+", "_", (s or '').lower()).split('_') if len(t) >= 3]
                agg_tokens = set(_tokens(folder_norm))
                agg_tokens.update(_tokens(parent_norm))
                agg_tokens.update(_tokens(id_parent_norm))
                agg_tokens.update(_tokens(container_norm))
                for cf in (folder_candidates or []):
                    agg_tokens.update(_tokens(cf))
                candidate_keys = set()
                for v in variants:
                    candidate_keys.update(_VLM_INDEX['by_num'].get(v, set()))
                # Intersect with aggregated tokens to reduce
                if agg_tokens and candidate_keys:
                    filtered = set()
                    for tok in agg_tokens:
                        tok_keys = _VLM_INDEX['tokens_to_keys'].get(tok, set())
                        if not tok_keys:
                            continue
                        inter = candidate_keys & tok_keys
                        if inter:
                            filtered.update(inter)
                    if filtered:
                        candidate_keys = filtered
                # Bound candidates
                if len(candidate_keys) > fuzzy_max_candidates:
                    candidate_keys = set(list(candidate_keys)[:fuzzy_max_candidates])
                # Choose the best candidate: prefer ones containing folder_norm
                best_key = None
                for k in candidate_keys:
                    if folder_norm and folder_norm in k:
                        best_key = k
                        break
                if not best_key and candidate_keys:
                    best_key = next(iter(candidate_keys))
                if best_key:
                    vlm_match = resolve_vlm_by_key(best_key)
                    if verbose_fuzzy:
                        print(f"Fuzzy indexed match: {img_path} → {best_key}")
        
        # Debug for problematic match cases
        if not vlm_match and os.environ.get("DEBUG_MATCHING", "0") == "1":
            print(f"\n[DEBUG] Looking for key: {normalized_path}")
            # Find similar keys to help troubleshoot based on patterns, not specific series
            similar_keys = []
            
            # Look for keys with similar patterns - using digits and folder name parts
            if number:
                for vlm_path in _VLM_INDEX['key_to_path'].keys():
                    if number in vlm_path or number.lstrip('0') in vlm_path:
                        similar_keys.append(vlm_path)
            
            # Add keys with folder name parts
            folder_parts = dirname.lower().split('/')
            for part in folder_parts:
                if len(part) >= 4:  # Only consider meaningful parts
                    for vlm_path in _VLM_INDEX['key_to_path'].keys():
                        if part in vlm_path.lower() and vlm_path not in similar_keys:
                            similar_keys.append(vlm_path)
            
            if similar_keys:
                print(f"[DEBUG] Found {len(similar_keys)} similar VLM keys:")
                for i, key in enumerate(sorted(similar_keys[:5], key=len)):  # Show up to 5, sorted by length
                    print(f"  {i+1}. {key}")
        
        if not vlm_match and allow_partial_match and not fast_only:
            # Last-resort bounded partial match via token hits
            folder_tokens = [t for t in folder_norm.split('_') if len(t) >= 4]
            candidates = set()
            for tok in folder_tokens:
                candidates.update(_VLM_INDEX['tokens_to_keys'].get(tok, set()))
            # Bound
            if len(candidates) > fuzzy_max_candidates:
                candidates = set(list(candidates)[:fuzzy_max_candidates])
            # Pick the shortest key that also contains any number from filename
            number_match = re.search(r'(\d+)', base_filename)
            number = number_match.group(1) if number_match else None
            picked = None
            for k in sorted(candidates, key=len):
                if number and (number in k or (number.lstrip('0') in k)):
                    picked = k
                    break
            if picked:
                vlm_match = resolve_vlm_by_key(picked)
                partial_vlm_key = picked
                if verbose_fuzzy:
                    print(f"Partial indexed match: COCO={normalized_path} VLM={picked}")
        if not vlm_match:
            # Log the normalized COCO path and the exact VLM key attempted
            if log_misses:
                log_path = os.path.abspath('calibre_match_failures.log')
                with open(log_path, 'a', encoding='utf-8') as logf:
                    logf.write(f"COCO image: {img_path} | Normalized: {normalized_path}\n---\n")
            # Print only when verbose to reduce console overhead
            if verbose_fuzzy:
                comic_name = os.path.basename(os.path.dirname(img_path))
                print(f"No VLM match for: {comic_name}/{os.path.basename(img_path)} → {normalized_path}")
            if first_failure:
                # Emit rich diagnostics and signal early termination
                print("\n===== FIRST FAILURE DIAGNOSTIC =====")
                try:
                    img_meta = image_info.get(image_id, {})
                    print(f"Image ID: {image_id}")
                    print(f"COCO image: {img_meta}")
                except Exception:
                    print(f"Image ID: {image_id}")
                    print(f"COCO path: {img_path}")
                print(f"Normalized key: {normalized_path}")
                print("Folder candidates tried (priority order):")
                try:
                    print("  " + ", ".join(folder_candidates))
                except Exception:
                    pass
                print(f"Resolved image path: {resolved_image_path if resolved_image_path else '<not found>'}")
                # Recompute number variants for reporting
                number = None
                try:
                    filename = os.path.basename(img_path)
                    base_filename = os.path.splitext(filename)[0].lower()
                    m_trail = re.search(r'_(\d+)$', base_filename)
                    if m_trail:
                        number = m_trail.group(1)
                    else:
                        nums_any = re.findall(r'(\d+)', base_filename)
                        number = nums_any[-1] if nums_any else None
                except Exception:
                    number = None
                if number:
                    variants = [number, number.lstrip('0') or '0', number.zfill(3), number.zfill(4), number.zfill(5)]
                    print(f"Numeric variants considered: {variants}")
                # Report direct map/index lookups
                key_exists = normalized_path in _VLM_INDEX.get('key_to_path', {})
                print(f"Direct key_to_path lookup exists: {key_exists}")
                # Show a few similar keys for the same folder token
                try:
                    folder_token = folder_candidates[0] if folder_candidates else None
                    similar = []
                    if folder_token:
                        for k in _VLM_INDEX.get('key_to_path', {}).keys():
                            if folder_token in k:
                                similar.append(k)
                                if len(similar) >= 10:
                                    break
                    if similar:
                        print("Similar keys (sample up to 10):")
                        for sk in similar:
                            print(f"  {sk}")
                except Exception:
                    pass
                print("===== END FIRST FAILURE DIAGNOSTIC =====\n")
                return {'_first_failure': True}
            if keep_unmatched:
                # Return a row indicating no VLM data, with rcnn_panels filled
                if panel_category_ids:
                    rcnn_panels = sum(1 for d in detections if d.get('category_id') in panel_category_ids)
                else:
                    rcnn_panels = len([d for d in detections if d.get('category_id') == 1])
                row = {
                    'image_id': image_id,
                    'image_path': img_path,
                    'rcnn_panels': rcnn_panels,
                    'vlm_panels': 0,
                    'panel_count_ratio': 0.0,
                    'text_coverage': 0.0,
                    'quality_score': 0.0,
                    'is_perfect_match': False,
                    'has_vlm_data': False,
                    'resolved_image_path': resolved_image_path,
                    'image_exists': bool(resolved_image_path and os.path.exists(resolved_image_path)),
                }
                if require_image_exists and not row['image_exists']:
                    # Keep row but it will be marked image_exists False
                    return row
                return row
            return None
        if panel_category_ids:
            rcnn_panels = sum(1 for d in detections if d.get('category_id') in panel_category_ids)
        else:
            rcnn_panels = len([d for d in detections if d.get('category_id') == 1])

        # Robustly extract panels from VLM JSON which may be dict or list or nested
        def _extract_panels(v):
            try:
                # Direct dict with panels
                if isinstance(v, dict):
                    if isinstance(v.get('panels'), list):
                        return v.get('panels')
                    # Common nesting patterns
                    for k in ['result', 'page', 'data']:
                        sub = v.get(k)
                        if isinstance(sub, dict) and isinstance(sub.get('panels'), list):
                            return sub.get('panels')
                        if isinstance(sub, list) and sub and isinstance(sub[0], dict) and isinstance(sub[0].get('panels'), list):
                            return sub[0].get('panels')
                    # Dict that looks like a single panel
                    if any(k in v for k in ['bbox', 'text', 'mask', 'polygon']):
                        return [v]
                    return []
                # List-based structures
                if isinstance(v, list):
                    # Single-item wrapper containing a dict with panels
                    if len(v) == 1 and isinstance(v[0], dict):
                        if isinstance(v[0].get('panels'), list):
                            return v[0].get('panels')
                    # List of panels directly
                    if v and isinstance(v[0], dict) and any(k in v[0] for k in ['bbox', 'text', 'mask', 'polygon']):
                        return v
                return []
            except Exception:
                return []

        panels_list = _extract_panels(vlm_match)
        vlm_panels = len(panels_list)
        panel_ratio = (rcnn_panels / vlm_panels) if vlm_panels > 0 else 0.0
        # Compute text coverage across varied panel schemas (speakers/caption/dialogue/narration/sfx/OCR)
        text_coverage = 0.0
        vlm_panels_with_text = 0
        vlm_text_dialogue_panels = 0
        vlm_text_narration_panels = 0
        vlm_text_sfx_panels = 0
        if vlm_panels > 0:
            import re as _re2
            def _nonempty_str(x):
                return isinstance(x, str) and bool(x.strip())
            def _any_nonempty_list(xs):
                return isinstance(xs, list) and any(_nonempty_str(x) or (isinstance(x, dict) and any(_nonempty_str(v) for v in x.values())) for x in xs)
            for panel in panels_list:
                has_any = False
                has_d = False
                has_n = False
                has_s = False
                if not isinstance(panel, dict):
                    # If panel is a bare string/list, treat any content as generic text
                    if _nonempty_str(panel) or (isinstance(panel, list) and any(_nonempty_str(x) for x in panel)):
                        has_any = True
                else:
                    # 1) Common "text" aggregate field
                    ptext = panel.get('text')
                    if isinstance(ptext, dict):
                        if _nonempty_str(ptext.get('dialogue')):
                            has_any = has_d = True
                        if _nonempty_str(ptext.get('narration')):
                            has_any = has_n = True or has_any
                        if _nonempty_str(ptext.get('sfx')):
                            has_any = has_s = True or has_any
                        # Also if any other non-empty string values exist
                        if not has_any and any(_nonempty_str(v) for v in ptext.values()):
                            has_any = True
                    elif isinstance(ptext, list):
                        if _any_nonempty_list(ptext):
                            has_any = True
                    elif _nonempty_str(ptext):
                        has_any = True

                    # 2) Speakers list with dialogue + speech_type
                    speakers = panel.get('speakers')
                    if isinstance(speakers, list):
                        for sp in speakers:
                            if not isinstance(sp, dict):
                                continue
                            dlg = sp.get('dialogue')
                            stype = str(sp.get('speech_type') or sp.get('type') or '').strip().lower()
                            if _nonempty_str(dlg):
                                has_any = True
                                # Heuristic categorization
                                if any(tok in stype for tok in ['dialogue', 'speech', 'thought']):
                                    has_d = True
                                elif any(tok in stype for tok in ['narration', 'caption', 'narrator', 'monologue']):
                                    has_n = True
                                elif any(tok in stype for tok in ['sfx', 'sound', 'onomatopoeia', 'effect']):
                                    has_s = True
                                else:
                                    # Unknown type: treat as dialogue by default
                                    has_d = True

                    # 3) Direct fields commonly present
                    if _nonempty_str(panel.get('dialogue')):
                        has_any = has_d = True
                    if _nonempty_str(panel.get('narration')):
                        has_any = has_n = True or has_any
                    # Treat caption as narration-style text
                    if _nonempty_str(panel.get('caption')):
                        has_any = has_n = True or has_any
                    # SFX direct
                    sfx_field = panel.get('sfx') or panel.get('sound_effects')
                    if _nonempty_str(sfx_field) or (isinstance(sfx_field, list) and any(_nonempty_str(x) for x in sfx_field)):
                        has_any = has_s = True or has_any

                    # 4) OCR-like fields
                    for ocr_key in ['ocr', 'ocr_text', 'ocr_lines', 'texts']:
                        ocr_val = panel.get(ocr_key)
                        if _nonempty_str(ocr_val) or (isinstance(ocr_val, list) and any(_nonempty_str(x) for x in ocr_val)):
                            has_any = True
                            has_d = True  # bucket OCR under dialogue-like text

                if has_any:
                    vlm_panels_with_text += 1
                if has_d:
                    vlm_text_dialogue_panels += 1
                if has_n:
                    vlm_text_narration_panels += 1
                if has_s:
                    vlm_text_sfx_panels += 1
            text_coverage = vlm_panels_with_text / vlm_panels if vlm_panels else 0.0
        quality_score = 0
        if 0.9 <= panel_ratio <= 1.1:
            quality_score += 2
        elif 0.8 <= panel_ratio <= 1.2:
            quality_score += 1
        if text_coverage >= _MIN_TEXT_COVERAGE:
            quality_score += 1
        elif text_coverage >= max(0.0, _MIN_TEXT_COVERAGE - 0.2):
            quality_score += 0.5
        if 1 <= vlm_panels <= 15:
            quality_score += 1
        elif 1 <= vlm_panels <= 20:
            quality_score += 0.5
        is_perfect_match = (panel_ratio == 1.0)
        vlm_json_path = _VLM_INDEX['key_to_path'].get(vlm_key_used) if 'key_to_path' in _VLM_INDEX else None
        row = {
            'image_id': image_id,
            'image_path': img_path,
            'rcnn_panels': rcnn_panels,
            'vlm_panels': vlm_panels,
            'panel_count_ratio': panel_ratio,
            'text_coverage': text_coverage,
            'vlm_panels_with_text': vlm_panels_with_text,
            'vlm_text_dialogue_panels': vlm_text_dialogue_panels,
            'vlm_text_narration_panels': vlm_text_narration_panels,
            'vlm_text_sfx_panels': vlm_text_sfx_panels,
            'quality_score': quality_score,
            'is_perfect_match': is_perfect_match,
            'vlm_data': vlm_match,
            'vlm_key': vlm_key_used,
            'vlm_json_path': vlm_json_path,
            'has_vlm_data': True,
            'resolved_image_path': resolved_image_path,
            'image_exists': bool(resolved_image_path and os.path.exists(resolved_image_path)),
        }
        if require_image_exists and not row['image_exists']:
            # Treat as matched but note missing image; training can filter by this flag
            return row
        return row
    except Exception as e:
        print(f"Error analyzing image {image_id}: {e}")
        if 'first_failure' in locals() and first_failure:
            # Emit minimal diagnostic if structural failure before normalization
            try:
                print("\n===== FIRST FAILURE STRUCTURAL DIAGNOSTIC =====")
                print(f"Image ID: {image_id}")
                if isinstance(image_info, dict) and image_id in image_info:
                    print(f"COCO image dict: {image_info.get(image_id)}")
                else:
                    print("COCO image dict: <not found>")
                print(f"Exception: {repr(e)}")
                print("===== END DIAGNOSTIC =====\n")
            except Exception:
                pass
            return {'_first_failure': True}
        return None

def test_normalization():
    """Test normalization logic with comprehensive examples to help debug key matching"""
    import os, re, glob, json
    from collections import defaultdict
    
    # Test examples covering various file naming patterns commonly found in calibre/comics
    examples = [
        # Simple numeric page names
        {"folder": "Comic Series Volume 1", "img": "001.jpeg"},
        {"folder": "Comic Series/Volume 1", "img": "002.png"},
        {"folder": "Series", "img": "Volume 1/003.jpg"},
        
        # Page prefix patterns
        {"folder": "Comic Series", "img": "page_001.jpeg"},
        {"folder": "Comic Series", "img": "page-002.png"},
        {"folder": "Comic Series", "img": "page003.jpg"},
        
        # Calibre specific patterns
        {"folder": "Comic - Volume 1 - Unknown", "img": "jpg4cbz_001.jpeg"},
        {"folder": "Comic Volume 2 - Issue 5", "img": "page_005.png"},
        {"folder": "Manga Series", "img": "Chapter 10/page_014.jpg"},
        
        # Mixed path formats
        {"folder": "Comic Series", "img": "Issue 101/003.jpg"},
        {"folder": "Series/Volume 5", "img": "001.jpg"},
        {"folder": "Digital Comics/Series", "img": "Chapter 056/page_014.jpg"},
    ]
    
    debug_folder = r"E:/CalibreComics_analysis"
    
    # Test both normalization functions
    print("\n[DEBUG] TESTING NORMALIZATION FUNCTIONS")
    print("=" * 60)
    
    # Import the normalization functions from our code
    import re
    
    # Define all key normalization strategies
    def foldername_number_key(path):
        import os, re
        folder = re.sub(r'[\s_-]+', '_', os.path.basename(os.path.dirname(path)).lower())
        filename = os.path.basename(path)
        match = re.search(r'(\d+)', filename)
        if match:
            num = match.group(1)
            return f"{folder}_{num}"
        else:
            base = os.path.splitext(filename)[0].lower()
            return f"{folder}_{base}"
    strategies = {
        "Basic": lambda path: re.sub(r'[\s_-]+', '_', os.path.splitext(os.path.basename(path))[0].lower()),
        "Folder_Filename": lambda path: f"{os.path.basename(os.path.dirname(path)).lower()}_{os.path.splitext(os.path.basename(path))[0].lower()}",
        "Extract_Number": lambda path: re.search(r'(\d+)', os.path.basename(path)).group(1) if re.search(r'(\d+)', os.path.basename(path)) else os.path.splitext(os.path.basename(path))[0].lower(),
        "FolderName_Number": lambda path: foldername_number_key(path),
        "Clean_Path": lambda path: re.sub(r'[\s_-]+', '_', path.replace('\\', '/').lower().replace(os.path.splitext(path)[1], '')),
    }
    
    # Add our actual implementation
    def normalize_image_key_impl(img_path):
        """Current implementation from code"""
        img_path = img_path.replace('\\', '/')
        folder = os.path.dirname(img_path)
        if not folder:
            parts = img_path.split('/')
            folder = parts[-2] if len(parts) > 1 else ''
        base = re.sub(r'[\s_-]+', '_', folder.strip()).lower()
        filename = os.path.basename(img_path)
        fname = filename.lower()
        fname = re.sub(r'(\.[a-z0-9]+)+$', '', fname)
        match = re.search(r'(jpg4cbz_\d+)', fname)
        if match:
            page_code = match.group(1)
        else:
            page_code = fname.split('_')[-1]
        return f"{base}_{page_code}"
    
    strategies["Current_Implementation"] = normalize_image_key_impl
    
    # Define new implementation
    def normalize_image_key_new(img_path):
        """Create a key from image path that handles multiple comic naming patterns"""
        import os, re
        
        # Normalize slashes and get path components
        img_path = img_path.replace('\\', '/').lower()
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        
        # Handle case where img_path doesn't have a directory
        if not folder_path:
            parts = img_path.split('/')
            folder_path = parts[-2] if len(parts) > 1 else ''
        
        # Clean folder name
        folder = os.path.basename(folder_path)
        folder = re.sub(r'[\s_-]+', '_', folder.strip()).lower()
        
        # Remove file extension and clean filename
        base_name = os.path.splitext(filename)[0].lower()
        
        # Try multiple page identification strategies
        
        # Strategy 1: Look for jpg4cbz pattern (appears in many calibre files)
        match = re.search(r'(jpg4cbz_\d+)', base_name)
        if match:
            page_code = match.group(1)
            return f"{folder}_{page_code}"
        
        # Strategy 2: Look for page_NNN pattern
        match = re.search(r'page_(\d+)', base_name)
        if match:
            page_num = match.group(1).zfill(3)
            return f"{folder}_{page_num}"
        
        # Strategy 3: If filename is just digits
        if base_name.isdigit():
            page_num = base_name.zfill(3)
            return f"{folder}_{page_num}"
            
        # Strategy 4: Extract any number from the filename
        match = re.search(r'(\d+)', base_name)
        if match:
            page_num = match.group(1).zfill(3)
            return f"{folder}_{page_num}"
        
        # Fallback: use the last part of the filename after any underscore
        parts = base_name.split('_')
        page_code = parts[-1] if len(parts) > 1 else base_name
        
        return f"{folder}_{page_code}"
    
    strategies["New_Implementation"] = normalize_image_key_new
    
    # Try all strategies on all examples
    results = defaultdict(dict)
    for i, example in enumerate(examples):
        folder = example['folder']
        img = example['img']
        img_path = f"{folder}/{img}"
        
        print(f"\n[Example {i+1}] {img_path}")
        print("-" * 60)
        
        for name, strategy in strategies.items():
            try:
                key = strategy(img_path)
                print(f"{name}: {key}")
                results[i][name] = key
            except Exception as e:
                print(f"{name}: ERROR - {str(e)}")
                results[i][name] = f"ERROR: {str(e)}"
    
    # Try to find VLM files matching our examples
    print("\n[DEBUG] CHECKING FOR MATCHING VLM FILES")
    print("=" * 60)
    
    # Collect JSON files for testing
    json_files = []
    json_keys = {}
    
    # First check if the debug folder exists
    if os.path.exists(debug_folder):
        for root, dirs, files in os.walk(debug_folder):
            for file in files:
                if file.lower().endswith('.json'):
                    full_path = os.path.join(root, file)
                    json_files.append(full_path)
                    # Generate keys for each JSON file
                    base_name = os.path.splitext(os.path.basename(full_path))[0]
                    key = re.sub(r'[\s_-]+', '_', base_name.lower())
                    json_keys[key] = full_path
                    # Also add a variant with numbers extracted
                    number_match = re.search(r'(\d+)', base_name)
                    if number_match:
                        folder_part = re.sub(r'[^\w]+', '_', os.path.basename(os.path.dirname(full_path)).lower())
                        num = number_match.group(1)
                        num_key = f"{folder_part}_{num}"
                        json_keys[num_key] = full_path
        
        # Print found JSON files
        print(f"Found {len(json_files)} JSON files in {debug_folder}")
        print(f"Generated {len(json_keys)} lookup keys")
        
        # Check for each example if we can find a match
        for i, example in enumerate(examples):
            print(f"\n[Example {i+1} Matches]")
            
            found_match = False
            for strategy_name, key in results[i].items():
                if isinstance(key, str) and not key.startswith("ERROR:"):
                    if key in json_keys:
                        found_match = True
                        match_file = os.path.basename(json_keys[key])
                        print(f"✓ Match found with {strategy_name}: {key} → {match_file}")
                        
                        # Try to peek inside the JSON file
                        try:
                            with open(json_keys[key], 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    print(f"  JSON structure: {list(data.keys())[:5]}...")
                                else:
                                    print(f"  JSON type: {type(data)}")
                        except Exception as e:
                            print(f"  Error loading JSON: {e}")
                            
            if not found_match:
                print("✗ No match found with any strategy")
    else:
        print(f"Debug folder {debug_folder} does not exist - skipping JSON matching tests")
        
    print("\n[DEBUG] NORMALIZATION TESTING COMPLETE")
    print("=" * 60)

def analyze_calibre_alignment(coco_file, vlm_dir, output_csv, num_workers=8, limit=None, allow_partial_match=False, fast_only=False, fuzzy_max_candidates=200, verbose_fuzzy=False, eager_vlm_load=False, log_misses=False, vlm_map: str = None, panel_category_ids=None, panel_category_names=None, keep_unmatched=False, first_failure=False, image_roots=None, require_image_exists=False, min_text_coverage: float = 0.8, emit_json_list=None, emit_all_json_lists: bool = False, json_list_out: str = None, json_require_exists: bool = False, emit_dataspec_for=None, emit_dataspec_all: bool = False, emit_dataspec_everything: bool = False, dataspec_out_dir: str = None, dataspec_require_equal_counts: bool = False, dataspec_min_det_score: float = 0.0, dataspec_list_out: str = None, dataspec_limit: int = None):
    """Analyze alignment between COCO detections and VLM data for Calibre comics"""
    # Create fuzzy matches log file
    with open("fuzzy_matches.log", "w", encoding="utf-8") as f:
        f.write("# Fuzzy matches log\n")
        f.write("original_path,normalized_key,vlm_key\n")
        
    # Skip test_normalization as we're using a completely different approach now
    print("🔍 Analyzing CalibreComics R-CNN vs VLM Alignment (v2)")
    print("=" * 60)
    image_detections, image_info, categories = load_coco_data(coco_file)
    # Determine panel category IDs
    cat_name_to_id = {c.get('name','').lower(): c.get('id') for c in categories}
    selected_ids = set(panel_category_ids or [])
    if panel_category_names:
        # split on commas/semicolons
        raw = panel_category_names.replace(';', ',')
        for name in [s.strip().lower() for s in raw.split(',') if s.strip()]:
            if name in cat_name_to_id:
                selected_ids.add(cat_name_to_id[name])
    if not selected_ids:
        # Auto-detect common names
        common = {'panel','frame','panel_frame','panel_border','comic_panel'}
        for n, cid in cat_name_to_id.items():
            if n in common:
                selected_ids.add(cid)
    print(f"Selected panel category id(s): {sorted(selected_ids) if selected_ids else '[default id==1]'}")
    vlm_data, vlm_index = load_vlm_data(vlm_dir, eager=eager_vlm_load, vlm_map=vlm_map)
    # Ensure globals are populated for single-threaded execution paths
    global _VLM_DATA, _VLM_INDEX, _VLM_EAGER, _MIN_TEXT_COVERAGE
    _VLM_DATA = vlm_data
    _VLM_INDEX = vlm_index
    _VLM_EAGER = eager_vlm_load
    _MIN_TEXT_COVERAGE = float(min_text_coverage) if min_text_coverage is not None else 0.8
    # Debug: print comprehensive diagnostics about key normalization
    print("\n--- DEBUG: Key Normalization Diagnostics ---")
    import re
    
    # Import the normalize_image_key function from analyze_image_alignment
    # to ensure we're using the exact same function
    def normalize_image_key(img_path):
        """Create a key from image path using general comic naming patterns"""
        import os, re
        
        # Normalize slashes and get path components
        img_path = img_path.replace('\\', '/').lower()
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        
        # Handle case where img_path doesn't have a directory
        if not folder_path:
            parts = img_path.split('/')
            folder_path = parts[-2] if len(parts) > 1 else ''
        
        # Clean folder name - use last folder component only
        folder = os.path.basename(folder_path)
        folder = re.sub(r'[\s_-]+', '_', folder.strip()).lower()
        
        # Remove file extension and clean filename
        base_name = os.path.splitext(filename)[0].lower()
        
        # Try multiple page identification strategies in order of specificity
        
        # Strategy 1: Calibre-specific jpg4cbz pattern
        match = re.search(r'(jpg4cbz_\d+)', base_name)
        if match:
            page_code = match.group(1)
            return f"{folder}_{page_code}"
        
        # Strategy 2: Common page_NNN or page-NNN patterns
        match = re.search(r'page[_\-\s]*(\d+)', base_name)
        if match:
            page_num = match.group(1).zfill(3)
            return f"{folder}_{page_num}"
        
        # Strategy 3: If filename is just digits (common in comics)
        if base_name.isdigit():
            page_num = base_name.zfill(3)
            return f"{folder}_{page_num}"
            
        # Strategy 4: Extract any number from the filename
        match = re.search(r'(\d+)', base_name)
        if match:
            page_num = match.group(1).zfill(3)
            return f"{folder}_{page_num}"
        
        # Fallback: use the whole cleaned filename
        # (we prefer the entire name rather than just the last part for better matching)
        clean_name = re.sub(r'[^a-z0-9]', '_', base_name)
        
        return f"{folder}_{clean_name}"
    
    # Sample images from different comic types to test key generation
    sample_imgs = []
    comics_by_type = {}
    
    # Get a sample of images for testing
    for img_id, img_data in image_info.items():
        img_path = img_data['file_name']
        
        # Categorize by file pattern
        if "jpg4cbz" in img_path:
            category = "jpg4cbz"
        elif "page_" in img_path:
            category = "page_NNN"
        elif re.search(r'/\d+\.', img_path):  # Files like 001.jpg
            category = "numeric"
        else:
            category = "other"
            
        if category not in comics_by_type:
            comics_by_type[category] = []
        
        comics_by_type[category].append(img_path)
        
        # Keep total sample size reasonable
        if len(sample_imgs) < 15 and len(comics_by_type[category]) <= 3:
            sample_imgs.append(img_path)
    
    # Print sample paths and their normalized versions by category
    print("Sample normalized paths by category:")
    for category, paths in comics_by_type.items():
        print(f"\n  {category.upper()} format examples:")
        for i, path in enumerate(paths[:3]):  # Show up to 3 per category
            try:
                norm_key = normalize_image_key(path)
                key_in_vlm = "✓ KEY FOUND" if (norm_key in vlm_data or norm_key in vlm_index['key_to_path']) else "✗ KEY NOT FOUND"
                print(f"    Original: {path}")
                print(f"    Normalized: {norm_key} - {key_in_vlm}")
            except Exception as e:
                print(f"    Error normalizing {path}: {e}")
    
    # Print distribution of VLM key patterns
    print("\nVLM key pattern distribution:")
    vlm_key_types = {
        "folder_number": 0,
        "contains_page": 0,
        "pure_numeric": 0,
        "jpg4cbz": 0,
        "other": 0
    }
    
    for key in vlm_data.keys():
        if "jpg4cbz" in key:
            vlm_key_types["jpg4cbz"] += 1
        elif "page" in key:
            vlm_key_types["contains_page"] += 1
        elif re.search(r'_\d+$', key):  # Ends with _number
            vlm_key_types["folder_number"] += 1
        elif re.match(r'^\d+$', key):  # Pure number
            vlm_key_types["pure_numeric"] += 1
        else:
            vlm_key_types["other"] += 1
    
    total_keys = len(vlm_index['key_to_path']) if 'key_to_path' in vlm_index else len(vlm_data)
    for key_type, count in vlm_key_types.items():
        print(f"  {key_type}: {count} keys ({count/max(1,total_keys)*100:.1f}%)")
        
    # Sample of VLM keys by type
    print("\nSample VLM keys by pattern:")
    for key_type in vlm_key_types.keys():
        print(f"  {key_type.upper()} examples:")
        examples = []
        for key in vlm_index['key_to_path'].keys():
            if (key_type == "jpg4cbz" and "jpg4cbz" in key) or \
               (key_type == "contains_page" and "page" in key) or \
               (key_type == "folder_number" and re.search(r'_\d+$', key)) or \
               (key_type == "pure_numeric" and re.match(r'^\d+$', key)) or \
               (key_type == "other" and not any([
                   "jpg4cbz" in key,
                   "page" in key,
                   re.search(r'_\d+$', key),
                   re.match(r'^\d+$', key)
               ])):
                examples.append(key)
                if len(examples) >= 3:
                    break
        for ex in examples:
            print(f"    {ex}")
    
    print(f"\nTotal VLM keys loaded: {len(vlm_index['key_to_path'])}")
    print("--- END DEBUG ---\n")
    # Parse image roots (semicolon or comma separated)
    roots_list = None
    if image_roots:
        parts = image_roots.replace(';', ',').split(',')
        roots_list = [p.strip().strip('"') for p in parts if p.strip()]
    analysis_args = []
    for image_id, detections in image_detections.items():
        analysis_args.append((image_id, detections, image_info, allow_partial_match, fast_only, fuzzy_max_candidates, verbose_fuzzy, log_misses, selected_ids, keep_unmatched, first_failure, roots_list, require_image_exists))
    if limit:
        analysis_args = analysis_args[:limit]
        print(f"Limited analysis to {limit} images")
    print(f"Analyzing {len(analysis_args)} images...")
    results = []
    if first_failure:
        # Force single-threaded for deterministic early-exit diagnostics
        for args in tqdm(analysis_args, desc="Analyzing images"):
            r = analyze_image_alignment(args)
            if isinstance(r, dict) and r.get('_first_failure'):
                print("Stopping after first failure as requested (--first_failure).")
                return
            if r is not None:
                results.append(r)
    elif num_workers > 1:
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(vlm_data, vlm_index, eager_vlm_load, min_text_coverage)) as pool:
            results = list(tqdm(
                pool.imap(analyze_image_alignment, analysis_args),
                total=len(analysis_args),
                desc="Analyzing images"
            ))
    else:
        # Single-threaded execution also needs globals set (already set above)
        results = [analyze_image_alignment(args) for args in tqdm(analysis_args, desc="Analyzing images")]
    results = [r for r in results if r is not None]
    attempted = len(analysis_args)
    matched = sum(1 for r in results if r.get('has_vlm_data'))
    if not results:
        print("No valid results found!")
        return
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"📊 Analysis Complete! Results saved to: {output_csv}")
    print(f"📈 Summary Statistics:")
    print(f"  Total inputs (attempted): {attempted}")
    print(f"  Rows in CSV: {len(df)}")
    print(f"  Images with VLM data (matched): {matched} ({(matched/max(1,attempted))*100:.1f}%)")
    print(f"  Good panel alignment: {len(df[df['panel_count_ratio'].between(0.9, 1.1)])} ({len(df[df['panel_count_ratio'].between(0.9, 1.1)])/len(df)*100:.1f}%)")
    print(f"  Good text coverage: {len(df[df['text_coverage'] >= min_text_coverage])} ({len(df[df['text_coverage'] >= min_text_coverage])/len(df)*100:.1f}%)")
    print(f"  Good VLM quality: {len(df[df['vlm_panels'].between(1, 15)])} ({len(df[df['vlm_panels'].between(1, 15)])/len(df)*100:.1f}%)")
    print(f"  Recommended for training: {len(df[df['quality_score'] >= 4])} ({len(df[df['quality_score'] >= 4])/len(df)*100:.1f}%)")
    print(f"🎯 Quality Score Distribution:")
    for score in sorted(df['quality_score'].unique()):
        count = len(df[df['quality_score'] == score])
        print(f"  Score {score}: {count} images ({count/len(df)*100:.1f}%)")
    print(f"📊 Panel Count Statistics:")
    print(f"  R-CNN panels - Avg: {df['rcnn_panels'].mean():.1f}, Min: {df['rcnn_panels'].min()}, Max: {df['rcnn_panels'].max()}")
    print(f"  VLM panels - Avg: {df['vlm_panels'].mean():.1f}, Min: {df['vlm_panels'].min()}, Max: {df['vlm_panels'].max()}")
    perfect_matches = df[df['is_perfect_match']]
    near_perfect = df[df['panel_count_ratio'].between(0.9, 1.1) & ~df['is_perfect_match']]
    print(f"🎯 Perfect Match Analysis:")
    print(f"  Perfect matches (1.0 ratio): {len(perfect_matches)} images ({len(perfect_matches)/len(df)*100:.1f}%)")
    print(f"  Near-perfect (0.9-1.1 ratio): {len(near_perfect)} images ({len(near_perfect)/len(df)*100:.1f}%)")
    print(f"  Perfect + Near-perfect: {len(perfect_matches) + len(near_perfect)} images ({(len(perfect_matches) + len(near_perfect))/len(df)*100:.1f}%)")
    high_quality = df[df['quality_score'] >= 4]
    medium_quality = df[df['quality_score'] == 3]
    low_quality = df[df['quality_score'] < 3]
    print(f"🎯 Training Dataset Recommendations:")
    print(f"  High quality (score 4-5): {len(high_quality)} images - Use for training")
    print(f"  Medium quality (score 3): {len(medium_quality)} images - Use with caution")
    print(f"  Low quality (score 0-2): {len(low_quality)} images - Exclude from training")
    perfect_matches.to_csv(output_csv.replace('.csv', '_perfect_matches.txt'), columns=['image_path'], header=False, index=False)
    near_perfect.to_csv(output_csv.replace('.csv', '_near_perfect.txt'), columns=['image_path'], header=False, index=False)
    high_quality.to_csv(output_csv.replace('.csv', '_high_quality.txt'), columns=['image_path'], header=False, index=False)
    medium_quality.to_csv(output_csv.replace('.csv', '_medium_quality.txt'), columns=['image_path'], header=False, index=False)
    # Additional list: perfect matches with any VLM text present
    if 'vlm_panels_with_text' in df.columns:
        perfect_with_text = perfect_matches[perfect_matches['vlm_panels_with_text'] > 0]
        perfect_with_text.to_csv(output_csv.replace('.csv', '_perfect_with_text.txt'), columns=['image_path'], header=False, index=False)
    print(f"💾 Filtered lists saved:")
    print(f"  Perfect matches: {output_csv.replace('.csv', '_perfect_matches.txt')}")
    print(f"  Near-perfect: {output_csv.replace('.csv', '_near_perfect.txt')}")
    print(f"  High quality: {output_csv.replace('.csv', '_high_quality.txt')}")
    print(f"  Medium quality: {output_csv.replace('.csv', '_medium_quality.txt')}")
    if 'vlm_panels_with_text' in df.columns:
        print(f"  Perfect + Text: {output_csv.replace('.csv', '_perfect_with_text.txt')}")

    # Emit a compact training manifest CSV for perfect matches
    try:
        manifest_cols = ['image_path', 'resolved_image_path', 'image_exists', 'vlm_key', 'vlm_json_path', 'rcnn_panels', 'vlm_panels', 'vlm_panels_with_text', 'vlm_text_dialogue_panels', 'vlm_text_narration_panels', 'vlm_text_sfx_panels']
        train_manifest = perfect_matches.copy()
        # If require_image_exists was set upstream, we may want to keep only existing images
        if 'image_exists' in train_manifest.columns:
            train_manifest = train_manifest[train_manifest['image_exists'] == True]
        train_manifest[manifest_cols].to_csv(output_csv.replace('.csv', '_perfect_match_training_manifest.csv'), index=False)
        print(f"📝 Training manifest saved: {output_csv.replace('.csv', '_perfect_match_training_manifest.csv')}")
    except Exception as e:
        print(f"Warning: could not write training manifest: {e}")

    # Print VLM text metrics summary for quick inspection
    try:
        if {'vlm_panels_with_text','vlm_text_dialogue_panels','vlm_text_narration_panels','vlm_text_sfx_panels','vlm_panels','text_coverage'}.issubset(df.columns):
            total_vlm_panels = int(df['vlm_panels'].sum())
            total_with_text_panels = int(df['vlm_panels_with_text'].sum())
            total_dialogue_panels = int(df['vlm_text_dialogue_panels'].sum())
            total_narration_panels = int(df['vlm_text_narration_panels'].sum())
            total_sfx_panels = int(df['vlm_text_sfx_panels'].sum())
            pages_with_any_text = int((df['vlm_panels_with_text'] > 0).sum())
            avg_text_cov_all = float(df['text_coverage'].mean())
            pw = df[df['is_perfect_match']]
            pw_pages_with_text = int((pw['vlm_panels_with_text'] > 0).sum()) if not pw.empty else 0
            avg_text_cov_perfect = float(pw['text_coverage'].mean()) if not pw.empty else 0.0
            print("\n🗣️ VLM Text Summary:")
            print(f"  Total VLM panels: {total_vlm_panels:,}")
            print(f"  Panels with any text: {total_with_text_panels:,} ({(total_with_text_panels/max(1,total_vlm_panels))*100:.1f}%)")
            print(f"   • Dialogue panels: {total_dialogue_panels:,}")
            print(f"   • Narration panels: {total_narration_panels:,}")
            print(f"   • SFX panels: {total_sfx_panels:,}")
            print(f"  Pages with any text: {pages_with_any_text:,} ({(pages_with_any_text/len(df))*100:.1f}%)")
            print(f"  Avg text coverage (all pages): {avg_text_cov_all:.3f}")
            print(f"  Perfect pages with any text: {pw_pages_with_text:,} ({(pw_pages_with_text/max(1,len(pw)))*100:.1f}% of perfect)")
            print(f"  Avg text coverage (perfect pages): {avg_text_cov_perfect:.3f}")
    except Exception as e:
        print(f"Warning: text summary error: {e}")

    # ----------------------------------------------------
    # Optional: Emit DataSpec JSON list(s) in one pass
    # ----------------------------------------------------
    try:
        import os as _os
        def _collect_jsons(sub_df):
            paths = []
            if 'vlm_json_path' not in sub_df.columns:
                return paths
            for p in sub_df['vlm_json_path']:
                if isinstance(p, str) and p.lower().endswith('.json'):
                    if (not json_require_exists) or _os.path.exists(p):
                        paths.append(p)
            # de-dupe and sort for determinism
            return sorted(list(dict.fromkeys(paths)))

        def _write_list(paths, out_path):
            if not paths:
                print(f"Note: No JSON paths to write for {out_path}")
                # still create an empty file for pipeline consistency
                with open(out_path, 'w', encoding='utf-8') as f:
                    pass
                return
            with open(out_path, 'w', encoding='utf-8') as f:
                for p in paths:
                    f.write(str(p).strip() + "\n")
            print(f"🧾 JSON list written: {out_path} ({len(paths)} items)")

        # Optionally filter to only images that exist when upstream required
        base_filter = df
        if 'image_exists' in df.columns and require_image_exists:
            base_filter = df[df['image_exists'] == True]

        # Build common subsets
        subsets = {
            'perfect': base_filter[base_filter['is_perfect_match'] == True],
            'near_perfect': base_filter[base_filter['panel_count_ratio'].between(0.9, 1.1) & ~base_filter['is_perfect_match']],
            'high_quality': base_filter[base_filter['quality_score'] >= 4],
            'medium_quality': base_filter[base_filter['quality_score'] == 3],
        }
        if 'vlm_panels_with_text' in base_filter.columns:
            subsets['perfect_with_text'] = base_filter[(base_filter['is_perfect_match'] == True) & (base_filter['vlm_panels_with_text'] > 0)]

        def _default_out(name: str) -> str:
            return output_csv.replace('.csv', f'_{name}_jsons.txt')

        # Emit per user request
        if emit_all_json_lists:
            for name, sdf in subsets.items():
                paths = _collect_jsons(sdf)
                _write_list(paths, _default_out(name))
        elif emit_json_list:
            # If a single category provided and a custom output specified, honor it
            if isinstance(emit_json_list, (list, tuple)):
                reqs = list(emit_json_list)
            else:
                reqs = [emit_json_list]
            for i, name in enumerate(reqs):
                sdf = subsets.get(name)
                if sdf is None:
                    print(f"Warning: unknown --emit_json_list category '{name}'. Allowed: {', '.join(subsets.keys())}")
                    continue
                paths = _collect_jsons(sdf)
                out_path = json_list_out if (json_list_out and len(reqs) == 1 and i == 0) else _default_out(name)
                _write_list(paths, out_path)
    except Exception as e:
        print(f"Warning: failed to emit JSON list(s): {e}")

    # ----------------------------------------------------
    # Optional: Emit DataSpec JSON files for training in one pass
    # ----------------------------------------------------
    try:
        # Quick exit if not requested
        do_emit = bool(emit_dataspec_all) or bool(emit_dataspec_everything) or (emit_dataspec_for is not None and len(emit_dataspec_for) > 0)
        if not do_emit:
            return
        if not dataspec_out_dir:
            print("Warning: --dataspec_out_dir is required to emit DataSpec JSONs; skipping.")
            return
        os.makedirs(dataspec_out_dir, exist_ok=True)

        # Reuse the same subset logic as above
        base_filter = df
        if 'image_exists' in df.columns and require_image_exists:
            base_filter = df[df['image_exists'] == True]
        subsets = {
            'perfect': base_filter[base_filter['is_perfect_match'] == True],
            'near_perfect': base_filter[base_filter['panel_count_ratio'].between(0.9, 1.1) & ~base_filter['is_perfect_match']],
            'high_quality': base_filter[base_filter['quality_score'] >= 4],
            'medium_quality': base_filter[base_filter['quality_score'] == 3],
        }
        if 'vlm_panels_with_text' in base_filter.columns:
            subsets['perfect_with_text'] = base_filter[(base_filter['is_perfect_match'] == True) & (base_filter['vlm_panels_with_text'] > 0)]
        # Everything = any matched VLM page regardless of quality
        if 'has_vlm_data' in base_filter.columns:
            subsets['everything'] = base_filter[base_filter['has_vlm_data'] == True]

        # Determine which set(s) to emit
        if emit_dataspec_everything:
            wanted_sets = ['everything']
        else:
            wanted_sets = list(subsets.keys()) if emit_dataspec_all else (list(emit_dataspec_for) if isinstance(emit_dataspec_for, (list, tuple)) else [emit_dataspec_for])
        wanted_sets = [w for w in wanted_sets if w in subsets]
        if not wanted_sets:
            # Exclude the synthetic 'everything' from the guidance list to avoid confusion
            allowed_names = [k for k in subsets.keys() if k != 'everything']
            print("Warning: no valid --emit_dataspec_for subset(s) selected; allowed: " + ', '.join(allowed_names))
            return

        # Helper to aggregate text from a VLM panel dict
        def _aggregate_text(panel_obj: dict) -> dict:
            out = {'dialogue': [], 'narration': [], 'sfx': []}
            def _add(val, bucket='dialogue'):
                if isinstance(val, str) and val.strip():
                    out[bucket].append(val.strip())
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        if isinstance(v, str) and v.strip():
                            out[bucket].append(v.strip())
            t = panel_obj.get('text')
            if isinstance(t, dict):
                for k in ('dialogue','narration','sfx','caption'):
                    _add(t.get(k), 'dialogue' if k=='dialogue' else ('narration' if k in ('narration','caption') else 'sfx'))
                for k,v in t.items():
                    if k not in ('dialogue','narration','sfx','caption'):
                        _add(v, 'dialogue')
            elif isinstance(t, (list, tuple)):
                _add(t, 'dialogue')
            elif isinstance(t, str):
                _add(t, 'dialogue')
            _add(panel_obj.get('caption'), 'narration')
            _add(panel_obj.get('description'), 'narration')
            _add(panel_obj.get('title') or panel_obj.get('alt'), 'narration')
            _add(panel_obj.get('key_elements'), 'narration')
            _add(panel_obj.get('actions'), 'narration')
            sp = panel_obj.get('speakers')
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
            for k in ('ocr','ocr_text','ocr_lines','texts'):
                _add(panel_obj.get(k), 'dialogue')
            return out

        # Helper to extract VLM panels list from the stored row data
        def _extract_panels_from_vlm(v):
            try:
                if isinstance(v, dict):
                    if isinstance(v.get('panels'), list):
                        return v.get('panels')
                    for k in ['result', 'page', 'data']:
                        sub = v.get(k)
                        if isinstance(sub, dict) and isinstance(sub.get('panels'), list):
                            return sub.get('panels')
                        if isinstance(sub, list) and sub and isinstance(sub[0], dict) and isinstance(sub[0].get('panels'), list):
                            return sub[0].get('panels')
                    if any(k in v for k in ['bbox', 'text', 'mask', 'polygon']):
                        return [v]
                    return []
                if isinstance(v, list):
                    if len(v) == 1 and isinstance(v[0], dict):
                        if isinstance(v[0].get('panels'), list):
                            return v[0].get('panels')
                    if v and isinstance(v[0], dict) and any(k in v[0] for k in ['bbox', 'text', 'mask', 'polygon']):
                        return v
                return []
            except Exception:
                return []

        # Build a reverse map from image_id to its detections filtered by score and category
        def _dets_for_image_id(img_id):
            dets = image_detections.get(img_id, [])
            boxes = []
            for d in dets:
                try:
                    if dataspec_min_det_score and float(d.get('score', 1.0)) < float(dataspec_min_det_score):
                        continue
                    if selected_ids:
                        if d.get('category_id') not in selected_ids:
                            continue
                    bbox = d.get('bbox') or d.get('box')
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x,y,w,h = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                        boxes.append([x,y,w,h])
                except Exception:
                    continue
            # Sort top-to-bottom, then left-to-right
            boxes.sort(key=lambda b: (b[1], b[0]))
            return boxes

        total_written = 0
        written_paths = []
        for subset_name in wanted_sets:
            sdf = subsets.get(subset_name)
            if sdf is None or sdf.empty:
                print(f"Note: subset '{subset_name}' is empty; skipping.")
                continue
            # Optional limit for quick smoke tests
            rows_iter = sdf.itertuples(index=False)
            if dataspec_limit is not None:
                rows_iter = list(rows_iter)[:int(dataspec_limit)]
            for row in tqdm(rows_iter, desc=f"Emitting DataSpec ({subset_name})"):
                try:
                    # Pandas namedtuple fields access
                    row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._asdict())
                except Exception:
                    # Fallback to Series
                    try:
                        row_dict = row._asdict()
                    except Exception:
                        row_dict = dict(row._asdict()) if hasattr(row, '_asdict') else {}
                try:
                    img_id = row_dict.get('image_id')
                    vlm_json_path = row_dict.get('vlm_json_path')
                    vlm_obj = row_dict.get('vlm_data')
                    page_image_path = row_dict.get('resolved_image_path') or row_dict.get('image_path')
                    if not page_image_path or (require_image_exists and not (isinstance(page_image_path, str) and os.path.exists(page_image_path))):
                        continue
                    det_boxes = _dets_for_image_id(img_id)
                    if not det_boxes:
                        continue
                    panels_vlm = _extract_panels_from_vlm(vlm_obj)
                    texts = [_aggregate_text(p) for p in panels_vlm if isinstance(p, dict)]
                    if dataspec_require_equal_counts and texts and (len(det_boxes) != len(texts)):
                        continue
                    K = min(len(det_boxes), len(texts) if texts else len(det_boxes))
                    if K == 0:
                        continue
                    panels_out = []
                    for i in range(K):
                        x,y,w,h = det_boxes[i]
                        t = texts[i] if i < len(texts) else {'dialogue': [], 'narration': [], 'sfx': []}
                        panels_out.append({
                            'panel_coords': [int(x), int(y), max(1,int(w)), max(1,int(h))],
                            'text': t
                        })
                    spec_obj = {
                        'page_image_path': page_image_path,
                        'panels': panels_out
                    }
                    # Derive output filename primarily from VLM JSON path when available
                    if isinstance(vlm_json_path, str) and vlm_json_path.lower().endswith('.json'):
                        base_out = os.path.splitext(os.path.basename(vlm_json_path))[0] + '.json'
                    else:
                        # fallback to image filename
                        base_out = os.path.splitext(os.path.basename(str(page_image_path)))[0] + '.json'
                    out_path = os.path.join(dataspec_out_dir, base_out)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(spec_obj, f, ensure_ascii=False)
                    total_written += 1
                    written_paths.append(out_path)
                except Exception:
                    continue
        print(f"Wrote {total_written} DataSpec JSONs to {dataspec_out_dir}")
        if dataspec_list_out:
            try:
                with open(dataspec_list_out, 'w', encoding='utf-8') as f:
                    for p in written_paths:
                        f.write(p + "\n")
                print(f"Saved DataSpec list: {dataspec_list_out}")
            except Exception as e:
                print(f"Warning: failed to write dataspec_list_out: {e}")
    except Exception as e:
        print(f"Warning: failed to emit DataSpec JSONs: {e}")

def main():
    parser = argparse.ArgumentParser(description='Create perfect match filter for CalibreComics dataset (v2)')
    parser.add_argument('--coco', help='Path to COCO detection JSON file')
    parser.add_argument('--vlm_dir', help='Directory containing VLM JSON files')
    parser.add_argument('--output_csv', default='calibre_rcnn_vlm_analysis_v2.csv', help='Output CSV file for analysis results')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to analyze (for testing)')
    parser.add_argument('--allow_partial_match', action='store_true', help='Enable bounded partial match analysis (uses indexed tokens)')
    parser.add_argument('--fast_only', action='store_true', help='Disable fuzzy/indexed fallback; use exact and folder+num only')
    parser.add_argument('--fuzzy_max_candidates', type=int, default=200, help='Max candidate keys to consider in fuzzy/indexed fallback')
    parser.add_argument('--verbose_fuzzy', action='store_true', help='Verbose logs for fuzzy/indexed matches')
    parser.add_argument('--eager_vlm_load', action='store_true', help='Eagerly load all VLM JSON files into memory (default is lazy on-demand)')
    parser.add_argument('--log_misses', action='store_true', help='Write unmatched cases to calibre_match_failures.log (default off for speed)')
    parser.add_argument('--test-normalization', action='store_true', help='Run tests on key normalization functions')
    parser.add_argument('--vlm_map', help='Optional JSON mapping file: image_path->json_path or list of jsons; speeds up and aligns keying')
    parser.add_argument('--panel_category_id', type=int, action='append', help='Panel category id to count as panel (can repeat)')
    parser.add_argument('--panel_category_names', type=str, help='Comma/semicolon-separated category names to include as panels')
    parser.add_argument('--keep_unmatched', action='store_true', help='Keep unmatched images as rows with has_vlm_data=False')
    parser.add_argument('--first_failure', action='store_true', help='Stop on first unmatched image and print detailed diagnostics')
    parser.add_argument('--image_roots', type=str, help='One or more base directories to resolve image files for training (comma or semicolon separated)')
    parser.add_argument('--require_image_exists', action='store_true', help='Mark rows with image_exists False if no file found under image_roots')
    parser.add_argument('--min_text_coverage', type=float, default=0.8, help='Minimum fraction of panels with text to award full text point (default 0.8). Half point if >= min-0.2')
    # One-pass DataSpec JSON list emission
    parser.add_argument('--emit_json_list', action='append', choices=['perfect','perfect_with_text','near_perfect','high_quality','medium_quality'], help='Emit a DataSpec JSON list for the selected subset (can repeat).')
    parser.add_argument('--emit_all_json_lists', action='store_true', help='Emit JSON lists for all standard subsets: perfect, perfect_with_text, near_perfect, high_quality, medium_quality.')
    parser.add_argument('--json_list_out', type=str, help='When emitting a single JSON list via --emit_json_list, write to this path instead of the default.')
    parser.add_argument('--json_require_exists', action='store_true', help='Only include JSON paths that exist on disk when building the list(s).')
    # One-pass DataSpec JSON emission
    parser.add_argument('--emit_dataspec_for', action='append', choices=['perfect','perfect_with_text','near_perfect','high_quality','medium_quality'], help='Emit DataSpec JSONs for the selected subset (can repeat).')
    parser.add_argument('--emit_dataspec_all', action='store_true', help='Emit DataSpec JSONs for all standard subsets.')
    parser.add_argument('--emit_dataspec_everything', action='store_true', help='Emit DataSpec JSONs for all pages with a VLM match (ignores quality tiers).')
    parser.add_argument('--dataspec_out_dir', type=str, help='Directory where DataSpec JSONs will be written.')
    parser.add_argument('--dataspec_require_equal_counts', action='store_true', help='Only emit DataSpec when RCNN count == VLM panel count.')
    parser.add_argument('--dataspec_min_det_score', type=float, default=0.0, help='Minimum detection score for panels used in DataSpec.')
    parser.add_argument('--dataspec_list_out', type=str, help='Optional path to write list of generated DataSpec JSONs.')
    parser.add_argument('--dataspec_limit', type=int, help='Optional limit when emitting DataSpec for quick smoke tests.')
    args = parser.parse_args()
    
    if args.test_normalization:
        print("Running normalization tests...")
        test_normalization()
    else:
        if not args.coco or not args.vlm_dir:
            parser.error("--coco and --vlm_dir are required unless --test-normalization is specified")
        analyze_calibre_alignment(
            args.coco,
            args.vlm_dir,
            args.output_csv,
            args.num_workers,
            args.limit,
            allow_partial_match=args.allow_partial_match,
            fast_only=args.fast_only,
            fuzzy_max_candidates=args.fuzzy_max_candidates,
            verbose_fuzzy=args.verbose_fuzzy,
            eager_vlm_load=args.eager_vlm_load,
            log_misses=args.log_misses,
            vlm_map=args.vlm_map,
            panel_category_ids=args.panel_category_id,
            panel_category_names=args.panel_category_names,
            keep_unmatched=args.keep_unmatched,
            first_failure=args.first_failure,
            image_roots=args.image_roots,
            require_image_exists=args.require_image_exists,
            min_text_coverage=args.min_text_coverage,
            emit_json_list=args.emit_json_list,
            emit_all_json_lists=args.emit_all_json_lists,
            json_list_out=args.json_list_out,
            json_require_exists=args.json_require_exists,
            emit_dataspec_for=args.emit_dataspec_for,
            emit_dataspec_all=args.emit_dataspec_all,
            emit_dataspec_everything=args.emit_dataspec_everything,
            dataspec_out_dir=args.dataspec_out_dir,
            dataspec_require_equal_counts=args.dataspec_require_equal_counts,
            dataspec_min_det_score=args.dataspec_min_det_score,
            dataspec_list_out=args.dataspec_list_out,
            dataspec_limit=args.dataspec_limit,
        )

if __name__ == "__main__":
    main()
