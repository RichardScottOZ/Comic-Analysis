import argparse
import hashlib
import os
from typing import Optional, Tuple
import pandas as pd

try:
    from PIL import Image
    import imagehash  # type: ignore
    _HAS_PHASH = True
except Exception:
    _HAS_PHASH = False


def file_md5(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except Exception:
        return None


def file_phash(path: str) -> Optional[int]:
    if not _HAS_PHASH or not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            ph = imagehash.phash(im)
            return int(str(ph), 16)
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def tag_duplicates(input_csv: str, output_csv: Optional[str] = None, use_phash: bool = False, phash_thresh: int = 6):
    df = pd.read_csv(input_csv)
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_duplicates.csv')

    # Prefer resolved image path for hashing
    img_col = 'resolved_image_path' if 'resolved_image_path' in df.columns else 'image_path'

    # Exact duplicate grouping by MD5
    md5s = []
    for p in df.get(img_col, []):
        md5s.append(file_md5(str(p)) if isinstance(p, str) else None)
    df['dup_md5'] = md5s

    # Assign group ids for exact duplicates
    group_id = {}
    next_gid = 1
    dup_group_exact = []
    for md5 in df['dup_md5']:
        if md5 and md5 in group_id:
            dup_group_exact.append(group_id[md5])
        elif md5:
            group_id[md5] = next_gid
            dup_group_exact.append(next_gid)
            next_gid += 1
        else:
            dup_group_exact.append(None)
    df['dup_group_exact'] = dup_group_exact

    # Perceptual near-duplicates (optional)
    df['dup_phash'] = None
    df['dup_group_phash'] = None
    if use_phash and _HAS_PHASH:
        phashes = []
        for p in df.get(img_col, []):
            phashes.append(file_phash(str(p)) if isinstance(p, str) else None)
        df['dup_phash'] = phashes
        # Greedy clustering by phash distance
        # Build index: value -> indices
        idxs = list(df.index)
        visited = set()
        group_id_p = 1
        group_map_p = {}
        for i in idxs:
            if i in visited:
                continue
            pi = df.at[i, 'dup_phash']
            if pi is None:
                continue
            # start a new group
            group_map_p[i] = group_id_p
            visited.add(i)
            for j in idxs:
                if j in visited:
                    continue
                pj = df.at[j, 'dup_phash']
                if pj is None:
                    continue
                if hamming(int(pi), int(pj)) <= phash_thresh:
                    group_map_p[j] = group_id_p
                    visited.add(j)
            group_id_p += 1
        df['dup_group_phash'] = df.index.map(group_map_p)

    # Canonical selection: keep the first occurrence in each exact group (or phash group if enabled)
    df['dup_canonical_key'] = None
    df['dup_reason'] = None
    if use_phash and _HAS_PHASH and df['dup_group_phash'].notna().any():
        # Use phash groups for canonicalization first
        for gid, g in df.groupby('dup_group_phash'):
            if pd.isna(gid):
                continue
            base_idx = g.index.min()
            df.loc[g.index, 'dup_canonical_key'] = int(base_idx)
            if len(g) > 1:
                df.loc[g.index, 'dup_reason'] = 'phash'
    # Exact groups
    for gid, g in df.groupby('dup_group_exact'):
        if pd.isna(gid):
            continue
        base_idx = g.index.min()
        # Respect existing canonical keys if already set by phash
        unset = g[df.loc[g.index, 'dup_canonical_key'].isna()].index
        df.loc[unset, 'dup_canonical_key'] = int(base_idx)
        if len(g) > 1:
            df.loc[g.index, 'dup_reason'] = df.loc[g.index, 'dup_reason'].fillna('md5')

    df.to_csv(output_csv, index=False)
    # Quick summary
    total = len(df)
    exact_dups = int(df['dup_group_exact'].nunique(dropna=True))
    phash_dups = int(df['dup_group_phash'].nunique(dropna=True)) if 'dup_group_phash' in df.columns else 0
    print(f"Saved {output_csv}")
    print(f"Rows: {total}; exact groups: {exact_dups}; phash groups: {phash_dups}")


def main():
    ap = argparse.ArgumentParser(description='Tag duplicate/near-duplicate pages in an analysis CSV')
    ap.add_argument('--input_csv', required=True, help='Analysis CSV (e.g., calibre_rcnn_vlm_analysis_v2.csv)')
    ap.add_argument('--output_csv', help='Output CSV; defaults to *_with_duplicates.csv')
    ap.add_argument('--use_phash', action='store_true', help='Enable perceptual hashing (requires Pillow + imagehash)')
    ap.add_argument('--phash_thresh', type=int, default=6, help='Max Hamming distance to cluster near-duplicates (default 6)')
    args = ap.parse_args()
    tag_duplicates(args.input_csv, args.output_csv, use_phash=args.use_phash, phash_thresh=args.phash_thresh)


if __name__ == '__main__':
    main()
