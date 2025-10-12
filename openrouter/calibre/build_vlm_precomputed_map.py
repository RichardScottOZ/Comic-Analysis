"""
Build a precomputed json_path -> image_path map for VLM JSON files.

This helps the COCO->DataSpec converter resolve VLM pages to absolute image paths
even when the JSON hints are inconsistent (different roots, WSL paths, etc.).

Usage:
  python build_vlm_precomputed_map.py --vlm_dir E:\CalibreVLM --image_root E:\CalibreImages --out E:\Maps\vlm_json_to_image.csv
  # Multiple VLM dirs: --vlm_dir "E:\VLM1;E:\VLM2"
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm


def _build_image_indexes(image_root: str):
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
            parts = [p for p in re.split(r"[^a-z0-9]+", stem) if p]
            if parts and parts[-1].isdigit():
                nz = parts[-1].lstrip('0') or '0'
                lastnum_idx.setdefault(nz, []).append(full)
    return fn_idx, stem_idx, norm_idx, lastnum_idx


def _resolve_to_image(path_hint: str, image_root: str, indexes) -> Optional[str]:
    fn_idx, stem_idx, norm_idx, lastnum_idx = indexes
    if not path_hint:
        return None
    hint = path_hint.replace('\\', '/').lstrip('/')
    cand = os.path.join(image_root, hint)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    base = os.path.basename(hint).lower()
    hits = fn_idx.get(base)
    if hits:
        return hits[0]
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


def _iter_vlm_files(vlm_dirs: list[Path]):
    for d in vlm_dirs:
        for f in d.rglob("*.json"):
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vlm_dir', required=True, help='Directory or ;/,-separated list of VLM JSON roots')
    ap.add_argument('--image_root', required=True, help='Root of extracted images')
    ap.add_argument('--out', required=True, help='Output file (.json or .csv)')
    ap.add_argument('--max', type=int, default=None, help='Optional cap for quick tests')
    args = ap.parse_args()

    # Normalize dirs
    dirs = []
    parts = re.split(r"[;,]", args.vlm_dir)
    for p in parts:
        p = p.strip()
        if p:
            dirs.append(Path(p))
    vlm_files = list(_iter_vlm_files(dirs))
    if args.max:
        vlm_files = vlm_files[:args.max]
    print(f"Scanning {len(vlm_files)} VLM JSON files ...")

    indexes = _build_image_indexes(args.image_root)
    results: Dict[str, str] = {}
    unresolved = []

    for vf in tqdm(vlm_files, desc='Resolving VLM', unit='json'):
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pages = [data] if isinstance(data, dict) else [p for p in data if isinstance(p, dict)]
            if not pages:
                unresolved.append((str(vf), 'no_pages'))
                continue
            img = None
            # Prefer explicit fields
            for k in ('page_image_path', 'image_path', 'image', 'IMAGE_PATH'):
                v = pages[0].get(k)
                if isinstance(v, str) and v.strip():
                    img = v
                    break
            abs_img = _resolve_to_image(img, args.image_root, indexes) if img else None
            if not abs_img:
                # Fallback to JSON filename stem
                stem = os.path.splitext(os.path.basename(str(vf)))[0]
                abs_img = _resolve_to_image(stem, args.image_root, indexes)
            if abs_img and os.path.exists(abs_img):
                results[str(vf)] = abs_img
            else:
                unresolved.append((str(vf), 'no_match'))
        except Exception as e:
            unresolved.append((str(vf), f'error:{e!r}'))

    print(f"Resolved {len(results)} of {len(vlm_files)} VLM JSONs. Missing: {len(unresolved)}")

    # Write outputs
    out = args.out
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if out.lower().endswith('.json'):
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f)
    else:
        import csv
        with open(out, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['json_path', 'image_path'])
            w.writeheader()
            for jp, ip in results.items():
                w.writerow({'json_path': jp, 'image_path': ip})

    miss_path = f"{out}.missing.txt"
    miss_dir = os.path.dirname(miss_path)
    if miss_dir:
        os.makedirs(miss_dir, exist_ok=True)
    with open(miss_path, 'w', encoding='utf-8') as f:
        for jp, why in unresolved:
            f.write(f"{jp}\t{why}\n")
    print(f"Wrote map to {out} and missing list to {miss_path}")


if __name__ == '__main__':
    main()
