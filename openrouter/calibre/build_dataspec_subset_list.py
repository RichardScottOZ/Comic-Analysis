#!/usr/bin/env python3
"""
Build a DataSpec subset list (e.g., perfect matches) from:
  - dataspec_all_list.txt (absolute paths to emitted DataSpec JSONs), and
  - a subset list produced by the analyzer (e.g., perfect matches or perfect+text)

We robustly match by:
  - page_image_path basename inside each DataSpec JSON (preferred), and
  - DataSpec JSON filename stem as a fallback.

This avoids re-running the long analyzer step and produces a ready-to-train list
that points into the single canonical DataSpec directory you already generated.

Usage (PowerShell/WSL similar):
  python build_dataspec_subset_list.py \
    --dataspec_list C:\path\to\dataspec_all_list.txt \
    --subset_list   C:\path\to\calibre_rcnn_vlm_analysis_v2_perfect_matches.txt \
    --out_list      C:\path\to\dataspec_perfect_list.txt
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Set, List


GENERIC_FOLDERS = {
    'jpg4cbz','pages','page','images','image','scans','scan',
    'calibrecomics_extracted','calibrecomics','extracted'
}

def _normalize_key(s: str) -> str:
    """Lowercase alnum-only key for robust matching.
    E.g., "Foo Bar_Page_001.jpg" -> "foobarpage001".
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # unify path separators
    s = s.replace('\\', '/').split('/')[-1]
    # drop extension if any
    if '.' in s:
        s = s.rsplit('.', 1)[0]
    # remove non-alnum
    return re.sub(r"[^a-z0-9]+", "", s)

def _series_token_from_path(p: str) -> str:
    """Extract a robust 'series' token from a file path by picking the nearest
    non-generic folder; collapse duplicate halves (foo_bar_foo_bar -> foo_bar).
    """
    try:
        parts = [t for t in p.replace('\\','/').split('/') if t]
        # Walk from tail-2 upwards looking for first non-generic
        for i in range(len(parts)-2, -1, -1):
            cand = re.sub(r"[^a-z0-9]+","_", parts[i].strip().lower()).strip('_')
            if cand and cand not in GENERIC_FOLDERS:
                toks = [t for t in cand.split('_') if t]
                # collapse duplicated trailing halves
                for half_len in range(1, len(toks)//2 + 1):
                    if toks[-2*half_len:-half_len] == toks[-half_len:]:
                        cand = '_'.join(toks[:-half_len])
                        break
                return cand
    except Exception:
        pass
    return ''

def _extract_number_variants(name: str) -> List[str]:
    """Return numeric variants found in a basename: jpg4cbz_NNNN, page_NNN,
    trailing _NNN, hash #NNN, or any number. Provide multiple zero-fill variants.
    """
    name = name.lower()
    # Remove extensions
    base = re.sub(r'(\.[a-z0-9]+)+$', '', name)
    nums: List[str] = []
    m = re.search(r'jpg4cbz[_\-\s]?(\d{1,6})', base)
    if m: nums.append(m.group(1))
    m = re.search(r'page[_\-\s]?(\d{1,6})', base)
    if m: nums.append(m.group(1))
    m = re.search(r'_(\d{1,6})$', base)
    if m: nums.append(m.group(1))
    m = re.search(r'#(\d{1,6})$', base)
    if m: nums.append(m.group(1))
    if not nums:
        alln = re.findall(r'(\d{1,6})', base)
        if alln: nums.append(alln[-1])
    out: List[str] = []
    for n in nums:
        v = {n, n.lstrip('0') or '0', n.zfill(3), n.zfill(4), n.zfill(5)}
        out.extend(sorted(v))
    # unique preserving order
    seen = set(); res = []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res


def _load_json_with_fallbacks(p: str):
    encs = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),
        ("cp1252", "strict"),
        ("latin-1", "strict"),
        ("utf-8", "replace"),
    ]
    last = None
    for enc, err in encs:
        try:
            with open(p, 'r', encoding=enc, errors=err) as f:
                return json.load(f)
        except Exception as e:
            last = e
            continue
    raise last if last else RuntimeError(f"Failed to read JSON: {p}")


def build_index(dataspec_list_file: Path) -> Dict[str, List[str]]:
    """Return a multimap from several key variants to one or more DataSpec paths.
    Keys include:
      - base-only key (image basename)
      - json-stem key
      - series__base key (series token + basename)
      - series__NNN variants (series + page number variants including jpg4cbz/page/hash)
      - numeric-only keys (NNN variants) for numeric basenames
    """
    index: Dict[str, List[str]] = {}
    paths: Set[str] = set()
    with open(dataspec_list_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if not os.path.exists(p):
                # try raw path without normalization
                continue
            paths.add(p)

    for jp in paths:
        try:
            data = _load_json_with_fallbacks(jp)
        except Exception:
            data = None
        # Accept dict or list-of-dicts
        if isinstance(data, dict):
            page = data
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            page = data[0]
        else:
            page = {}

        # Prefer page_image_path basename + series + number variants
        imgp = (page.get('page_image_path') or page.get('image_path') or '')
        base = _normalize_key(os.path.basename(str(imgp)))
        series = _series_token_from_path(str(imgp))
        nums = _extract_number_variants(os.path.basename(str(imgp)))
        # Fallback: JSON stem
        stem = Path(jp).stem
        stem_key = _normalize_key(stem)

        def put(k: str, v: str):
            if not k:
                return
            index.setdefault(k, []).append(v)

        put(base, jp)
        put(stem_key, jp)
        # Always index numeric-only variants as standalone keys to catch cases
        # where DataSpec basenames are numeric (e.g., 001.jpg) or page_001 etc.
        for n in nums:
            put(n, jp)

        if series:
            put(f"{series}__{base}", jp)
            for n in nums:
                put(f"{series}__{n}", jp)

    return index


def load_subset_key_variants(subset_list_file: Path) -> List[Dict[str, str]]:
    """Load subset entries and compute multiple key variants per line.
    Returns a list of dicts with {'raw': line, 'keys': [k1,k2,...], 'series': series}.
    """
    out = []
    with open(subset_list_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            base_full = s.replace('\\','/')
            base = base_full.split('/')[-1]
            bkey = _normalize_key(base)
            series = _series_token_from_path(base_full)
            nums = _extract_number_variants(base)
            keys = [bkey]
            # Always include numeric-only variants to match DataSpec numeric basenames
            for n in nums:
                keys.append(n)

            if series:
                keys.append(f"{series}__{bkey}")
                for n in nums:
                    keys.append(f"{series}__{n}")
            # unique preserve order
            seen = set(); klist = []
            for k in keys:
                if k and k not in seen:
                    seen.add(k); klist.append(k)
            out.append({'raw': s, 'keys': klist, 'series': series})
    return out


def main():
    ap = argparse.ArgumentParser(description='Build a DataSpec subset list from analyzer outputs.')
    ap.add_argument('--dataspec_list', required=True, help='Path to dataspec_all_list.txt')
    ap.add_argument('--subset_list', required=True, help='Analyzer subset list (e.g., perfect matches .txt)')
    ap.add_argument('--out_list', required=True, help='Output list file for the subset (DataSpec JSON paths)')
    args = ap.parse_args()

    ds_list = Path(args.dataspec_list)
    sub_list = Path(args.subset_list)
    out_list = Path(args.out_list)

    print(f"Indexing DataSpec from: {ds_list}")
    index = build_index(ds_list)
    total_keys = sum(len(v) for v in index.values())
    print(f"Indexed {len(index)} key buckets covering {total_keys} path entries")

    print(f"Loading subset keys from: {sub_list}")
    want = load_subset_key_variants(sub_list)
    print(f"Subset has {len(want)} entries")

    matched_paths: List[str] = []
    missing: List[str] = []
    for entry in want:
        found = None
        for k in entry['keys']:
            cand = index.get(k)
            if cand and len(cand) == 1:
                found = cand[0]
                break
            elif cand and len(cand) > 1:
                # Disambiguate by preferring paths whose series token matches
                ser = entry.get('series') or ''
                if ser:
                    filtered = [p for p in cand if _series_token_from_path(p) == ser]
                    if filtered:
                        found = filtered[0]
                        break
                # else pick the first deterministically
                found = cand[0]
                break
        if found:
            matched_paths.append(found)
        else:
            missing.append(entry['raw'])

    out_list.parent.mkdir(parents=True, exist_ok=True)
    with open(out_list, 'w', encoding='utf-8') as f:
        for p in matched_paths:
            f.write(p + "\n")

    print(f"Wrote {len(matched_paths)} paths to: {out_list}")
    if missing:
        # Show a small sample of missing to guide normalization tweaks if needed
        print(f"Warning: {len(missing)} subset keys had no match in DataSpec index. Sample:")
        for s in list(missing)[:10]:
            print(f"  - {s}")


if __name__ == '__main__':
    main()
