#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why pages ended up with panel_count=1 (fallback zero-tensor) during Stage 3 embedding.

For a sample of panel_count=1 pages, checks:
  1. Does the VLM JSON exist at the expected path?
  2. If yes, does it have valid panels with bboxes?

This tells us whether the issue is:
  (A) Path resolution failures during embedding (bug) -> fix and retrain
  (B) Genuine missing JSONs (data gap)               -> nothing to do

Results are written to: documentation/plots/panel_count_diagnosis.txt
"""

import os
import sys
import json
import random
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "documentation/plots/umap_stage3_pages.parquet"
JSON_ROOT    = "E:/Comic_Analysis_Results_v2/stage3_json"
OUTPUT_TXT   = "documentation/plots/panel_count_diagnosis.txt"
SAMPLE_SIZE  = 300
RANDOM_SEED  = 42

# All known subdirs under JSON_ROOT that hold comic folders
JSON_ROOT_SUBDIRS = [
    "CalibreComics_extracted",
    "CalibreComics_extracted_20251107",
    "NeonIchiban",
    "",  # bare json_root as last resort
]

# Canonical_id prefix variations to strip before path joining
ID_PREFIXES = [
    "CalibreComics_extracted/",
    "CalibreComics_extracted_20251107/",
    "CalibreComics_extracted\\",
    "amazon/",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def strip_prefix(cid):
    for pfx in ID_PREFIXES:
        if cid.startswith(pfx):
            return cid[len(pfx):]
    return cid


def try_find_json(canonical_id):
    """Try every plausible path to find the VLM JSON file."""
    bare = strip_prefix(canonical_id)
    candidates = set()

    for subdir in JSON_ROOT_SUBDIRS:
        base = os.path.join(JSON_ROOT, subdir) if subdir else JSON_ROOT
        # Try with bare id (prefix stripped)
        candidates.add(os.path.join(base, bare.replace('/', os.sep) + ".json"))
        # Try with full canonical_id (in case subdir already matches)
        candidates.add(os.path.join(base, canonical_id.replace('/', os.sep) + ".json"))
        # Try just the last two path components (book/page)
        parts = bare.split('/')
        if len(parts) >= 2:
            candidates.add(os.path.join(base, os.sep.join(parts[-2:]) + ".json"))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def count_valid_panels(json_path):
    """Return (total_panels, valid_panels_with_bbox)."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        panels = data.get('panels', [])
        valid = sum(
            1 for p in panels
            if p.get('bbox') and len(p['bbox']) == 4
            and p['bbox'][2] > 0 and p['bbox'][3] > 0
        )
        return len(panels), valid
    except Exception:
        return -1, -1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    out = open(OUTPUT_TXT, 'w', encoding='utf-8')
    def p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=out, **kwargs)

    p(f"Loading: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    df1 = df[df['panel_count'] == 1].copy()
    p(f"\npanel_count=1 pages: {len(df1):,} / {len(df):,} ({100*len(df1)/len(df):.1f}%)")

    sample = df1.sample(min(SAMPLE_SIZE, len(df1)), random_state=RANDOM_SEED)
    p(f"Sampling {len(sample)} pages...\n")

    counts = {
        'json_found_valid_panels': [],
        'json_found_no_valid_panels': [],
        'json_not_found': [],
        'json_parse_error': [],
    }

    for _, row in sample.iterrows():
        cid = row['canonical_id']
        json_path = try_find_json(cid)

        if json_path is None:
            counts['json_not_found'].append(cid)
        else:
            total_p, valid_p = count_valid_panels(json_path)
            if total_p < 0:
                counts['json_parse_error'].append(cid)
            elif valid_p > 0:
                counts['json_found_valid_panels'].append((cid, valid_p, json_path))
            else:
                counts['json_found_no_valid_panels'].append((cid, json_path))

    total = len(sample)
    p("=" * 70)
    p("RESULTS")
    p("=" * 70)
    for k, v in counts.items():
        pct = 100 * len(v) / total
        p(f"  {k:35s}: {len(v):4d} / {total}  ({pct:.1f}%)")

    found_pct = 100 * len(counts['json_found_valid_panels']) / total
    missing_pct = 100 * len(counts['json_not_found']) / total

    p("\nVERDICT")
    p("-" * 70)
    if found_pct > 40:
        p(f"  [WARNING] {found_pct:.0f}% of panel_count=1 pages have JSONs with real panels.")
        p("  Training failure was a PATH RESOLUTION BUG.")
        p("  Fix path resolution, regenerate Zarr embeddings, retrain.")
    elif missing_pct > 60:
        p(f"  [OK] {missing_pct:.0f}% of panel_count=1 pages genuinely have no JSON.")
        p("  These are true data gaps, panel_count=1 is correct.")
    else:
        p("  Mixed result - investigate the breakdown further.")

    p("\nEXAMPLES: JSON found WITH valid panels (should have been > 1 panel at training):")
    for cid, n, path in counts['json_found_valid_panels'][:8]:
        p(f"  [{n} panels] {cid}")
        p(f"           -> {path}")

    p("\nEXAMPLES: JSON NOT found:")
    for cid in counts['json_not_found'][:8]:
        p(f"  {cid}")

    p("\nEXAMPLES: JSON found but NO valid panels (genuine single-panel or bbox issue):")
    for cid, path in counts['json_found_no_valid_panels'][:8]:
        p(f"  {cid}")

    out.close()
    print(f"Results written to: {OUTPUT_TXT}")


if __name__ == '__main__':
    main()
