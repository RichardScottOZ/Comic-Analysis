"""
Enhanced VLM precomputed map builder (v2)

This script builds a richer mapping between VLM JSON files and image files / normalized keys.
It attempts to reproduce the analyzer's key-variation logic to avoid many fuzzy misses (covers, OEBPS layouts,
jpg4cbz patterns, calibre prefixes, etc.). Output includes:
 - key_to_jsons.json: mapping from normalized key variant -> list of JSON paths
 - json_to_image.json (optional): mapping from JSON path -> resolved absolute image path (if image_root provided)
 - image_to_jsons.json (optional): mapping from resolved absolute image path -> list of JSON paths

Usage:
  python build_vlm_precomputed_map_v2.py --vlm_dir E:\CalibreComics_analysis --image_root "E:\CalibreComics_extracted" --out_dir E:\Maps

Notes:
 - This is more conservative than the older map builder and emits multiple variants per JSON.
 - Downstream analyzer can be passed --vlm_map <json> where <json> can be the json produced here (json_to_image or key_to_jsons depending on needs).
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Set
from tqdm import tqdm


def normalize_folder_name(name: str) -> str:
    return re.sub(r'[\s_-]+', '_', name.strip()).lower()


def normalize_image_key_from_img_path(img_path: str) -> str:
    img_path = img_path.replace('\\', '/').lower()
    folder_path = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    base_name = re.sub(r'(\.[a-z0-9]+)+$', '', filename.lower())
    flattened = ('/' not in img_path and '\\' not in img_path)
    generic_prefixes = (
        'calibrecomics_extracted_', 'calibrecomicsanalysis_', 'calibrecomics_analysis_', 'calibrecomics_', 'images_', 'extracted_', 'jpg4cbz_', 'pages_', 'page_'
    )
    if flattened:
        for pref in generic_prefixes:
            if base_name.startswith(pref):
                base_name = base_name[len(pref):]
                break
    folder_raw = os.path.basename(folder_path) if folder_path else ''
    folder = normalize_folder_name(folder_raw)
    # Handle generic markers
    generic_markers = {"calibrecomics_extracted", "calibrecomics", "images", "extracted", "jpg4cbz", "pages", "page", "scans"}
    if folder in generic_markers or any(g in folder for g in ('extracted','calibrecomics')):
        title = base_name
        title = re.sub(r'_(?:jpg4cbz_)?\d{1,6}$', '', title)
        title = re.sub(r'_?page[_\-]?\d{1,6}$', '', title)
        title = re.sub(r'[\s\-]+', '_', title).strip('_')
        if title:
            folder = normalize_folder_name(title)
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
    m = re.search(r'_(\d{1,6})$', base_name)
    if m:
        num = m.group(1)
        return f"{folder}_{num}"
    nums = re.findall(r'(\d{1,6})', base_name)
    if nums:
        num = nums[-1]
        return f"{folder}_{num}"
    clean_name = re.sub(r'[^a-z0-9]', '_', base_name).strip('_')
    return f"{folder}_{clean_name}"


def generate_key_variants_from_jsonpath(vlm_path: Path) -> List[str]:
    # Similar to register_vlm_json's normalize_vlm_key behavior in the analyzer
    filename = vlm_path.name
    parent = vlm_path.parent.name.lower() if vlm_path.parent else ''
    base = filename.replace('.json','')
    normalized = re.sub(r'[\s_-]+','_', base.strip()).lower()
    keys = [normalized]
    if parent:
        parent_norm = re.sub(r'[\s_-]+', '_', parent.strip())
        keys.append(f"{parent_norm}_{normalized}")
    # numeric variants
    m_trail = re.search(r'_(\d{1,6})$', normalized)
    if m_trail:
        num = m_trail.group(1)
    else:
        nums_any = re.findall(r'(\d{1,6})', normalized)
        num = nums_any[-1] if nums_any else None
    if num:
        if parent:
            parent_norm = re.sub(r'[\s_-]+', '_', parent.strip().lower())
            keys.append(f"{parent_norm}_{num}")
            keys.append(f"{parent_norm}_{num.zfill(3)}")
        keys.append(num)
        keys.append(num.zfill(3))
    # relative path variant
    rel = str(vlm_path).replace('\\','/').lower()
    # try to get rel from last two components
    parts = rel.split('/')
    if len(parts) >= 2:
        rel_key = re.sub(r'[\s_-]+','_', f"{parts[-2]}_{parts[-1].replace('.json','')}" )
        keys.append(rel_key)
    # unique and return
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def build_image_index(image_root: Path):
    fn_idx = {}
    dir_idx = {}
    for root, dirs, files in os.walk(image_root):
        # record directory normalized keys
        try:
            dname = os.path.basename(root).lower()
            dnorm = re.sub(r'[^a-z0-9]+', '', dname)
            dir_idx.setdefault(dnorm, []).append(root)
        except Exception:
            pass
        for f in files:
            full = os.path.join(root, f)
            fn = f.lower()
            fn_idx.setdefault(fn, []).append(full)
    return fn_idx, dir_idx


def try_resolve_hint(hint: str, image_root: Path, fn_idx, dir_idx) -> str | None:
    if not hint:
        return None
    # direct join
    cand = os.path.join(str(image_root), hint.replace('/','\\'))
    if os.path.exists(cand):
        return os.path.abspath(cand)
    # basename match
    b = os.path.basename(hint).lower()
    hits = fn_idx.get(b)
    if hits:
        return hits[0]
    # Heuristic fallbacks for common calibre/epub extraction patterns
    stem = os.path.splitext(b)[0]
    # 1) cover.* variants
    if 'cover' in stem:
        for ext in ('.jpg', '.jpeg', '.png'):
            k = f'cover{ext}'
            if k in fn_idx:
                return fn_idx[k][0]
    # 2) numeric-based pages: try unpadded/padded numeric variants and common prefixes
    num_match = re.search(r'(\d{1,6})$', stem)
    if num_match:
        num = num_match.group(1)
        candidates = []
        candidates.append(f"{num}.jpg")
        candidates.append(f"{num}.jpeg")
        candidates.append(f"{num}.png")
        candidates.append(f"{num.zfill(3)}.jpg")
        candidates.append(f"{num.zfill(4)}.jpg")
        candidates.append(f"image-{num}.jpg")
        candidates.append(f"image_{num}.jpg")
        candidates.append(f"i{num}.jpg")
        candidates.append(f"image-{num.zfill(3)}.jpg")
        # try candidates in fn_idx
        for c in candidates:
            if c in fn_idx:
                return fn_idx[c][0]
    # 3) common epub-layouts: try OEBPS/images and images folders under any parent by scanning fn_idx keys
    # This is more expensive but bounded to a few checks (we check by suffix patterns)
    suffix_variants = [f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"]
    for sv in suffix_variants:
        if sv in fn_idx:
            return fn_idx[sv][0]
    # 4) last-resort: scan fn_idx for keys that endwith the stem or contain the numeric token
    # (avoid full scan unless necessary)
    short_tokens = [t for t in re.split(r'[^a-z0-9]+', stem) if t]
    if short_tokens:
        token = short_tokens[-1]
        # try matching any basename that contains the token
        for k in fn_idx.keys():
            if token in k:
                return fn_idx[k][0]

    # 5) directory-name heuristics: try to find a directory whose normalized name matches
    # the hint's parent or stem when extraction added suffixes like '._1'
    try:
        # if hint contains a path component, prefer the parent part
        if '/' in hint or '\\' in hint:
            parent_part = os.path.dirname(hint).replace('\\','/').split('/')[-1]
        else:
            # derive a parent-like token from stem by removing trailing numeric tokens
            parent_part = re.sub(r'[\._-]?\d{1,6}$', '', stem)
        parent_norm = re.sub(r'[^a-z0-9]+', '', parent_part.lower())
        # direct normalized dir match
        if parent_norm in dir_idx:
            # attempt to find a matching file inside the directory using stem or numeric
            for dpath in dir_idx[parent_norm]:
                # try file equal to stem + common extensions
                for ext in ('.jpg', '.jpeg', '.png'):
                    candf = os.path.join(dpath, stem + ext)
                    if os.path.exists(candf):
                        return candf
                # try numeric variants if stem ends with digits
                mnum = re.search(r'(\d{1,6})$', stem)
                if mnum:
                    num = mnum.group(1)
                    for fn in (f"{num}.jpg", f"{num.zfill(3)}.jpg", f"{num.zfill(4)}.jpg"):
                        candf = os.path.join(dpath, fn)
                        if os.path.exists(candf):
                            return candf
                # fallback: return first image file in directory
                try:
                    for entry in os.listdir(dpath):
                        if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                            return os.path.join(dpath, entry)
                except Exception:
                    pass
        # try looser matching: find any dir whose normalized name contains parent_norm
        for dn, paths in dir_idx.items():
            if parent_norm and parent_norm in dn:
                for dpath in paths:
                    for entry in os.listdir(dpath):
                        if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                            return os.path.join(dpath, entry)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vlm_dir', required=True, help='VLM JSON dir (semicolon/commas allowed)')
    ap.add_argument('--image_root', required=False, help='Optional image root to resolve image_paths')
    ap.add_argument('--out_dir', required=True, help='Output directory to write maps')
    ap.add_argument('--max', type=int, default=None, help='Limit for quick tests')
    ap.add_argument('--fail_on_unresolved', action='store_true', help='Immediately fail (exit non-zero) on first unresolved VLM JSON')
    args = ap.parse_args()

    vlm_dirs = [Path(p.strip()) for p in re.split(r'[;,]', args.vlm_dir) if p.strip()]
    vlm_files = []
    for d in vlm_dirs:
        for f in d.rglob('*.json'):
            vlm_files.append(f)
    if args.max:
        vlm_files = vlm_files[:args.max]
    print(f"Found {len(vlm_files)} VLM JSON files")

    image_root = Path(args.image_root) if args.image_root else None
    if image_root and image_root.exists():
        fn_idx, dir_idx = build_image_index(image_root)
    else:
        fn_idx, dir_idx = {}, {}

    key_to_jsons: Dict[str, List[str]] = {}
    json_to_image: Dict[str, str] = {}
    image_to_jsons: Dict[str, List[str]] = {}
    unresolved: List[tuple] = []

    for vf in tqdm(vlm_files, desc='Indexing VLM files'):
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        pages = [data] if isinstance(data, dict) else [p for p in data if isinstance(p, dict)]
        if not pages:
            continue
        # extract hint fields
        hint = None
        for k in ('page_image_path','image_path','image','IMAGE_PATH','page_image'):
            v = pages[0].get(k) if isinstance(pages[0], dict) else None
            if isinstance(v, str) and v.strip():
                hint = v
                break
        # record normalized key variants from path and json filename
        variants = generate_key_variants_from_jsonpath(vf)
        for kv in variants:
            key_to_jsons.setdefault(kv, []).append(str(vf))
        # attempt to resolve to image if root provided
        resolved = None
        if image_root:
            resolved = try_resolve_hint(hint or os.path.splitext(os.path.basename(str(vf)))[0], image_root, fn_idx, dir_idx)
            if resolved:
                json_to_image[str(vf)] = resolved
                image_to_jsons.setdefault(resolved, []).append(str(vf))
            else:
                # record unresolved; optionally fail fast
                unresolved.append((str(vf), 'no_match'))
                if args.fail_on_unresolved:
                    print(f"First unresolved VLM JSON: {vf} (no matching image under {image_root})")
                    # write partial outputs for debugging
                    outd = Path(args.out_dir)
                    outd.mkdir(parents=True, exist_ok=True)
                    with open(outd / 'json_to_image.partial.json', 'w', encoding='utf-8') as f:
                        json.dump(json_to_image, f, indent=2)
                    diag_path = outd / 'first_unresolved_vlm.txt'
                    try:
                        with open(diag_path, 'w', encoding='utf-8') as d:
                            d.write(f"{vf}\t{hint or ''}\n")
                    except Exception:
                        pass
                    print(f"Wrote diagnostic: {diag_path}")
                    raise SystemExit(3)
    # write outputs
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    with open(outd / 'key_to_jsons.json', 'w', encoding='utf-8') as f:
        json.dump(key_to_jsons, f, indent=2)
    with open(outd / 'json_to_image.json', 'w', encoding='utf-8') as f:
        json.dump(json_to_image, f, indent=2)
    with open(outd / 'image_to_jsons.json', 'w', encoding='utf-8') as f:
        json.dump(image_to_jsons, f, indent=2)
    print(f"Wrote maps to {outd}")


if __name__ == '__main__':
    main()
