"""
Compare VLM box_2d panel grounding vs RCNN bbox panel detection
for the Stage 3 dataset.

For a sample of pages, reports:
  - VLM panel count   (box_2d entries in E:\vlm_cache)
  - RCNN panel count  (bbox entries in E:\Comic_Analysis_Results_v2\stage3_json)
  - Which source gives more panels
  - How many panel_count=1 (VLM) pages have >1 RCNN panel (i.e. VLM missed them)
  - Overall distribution comparison

Output: documentation/plots/vlm_vs_rcnn_panels.txt
"""

import os, json, random, collections
import pandas as pd

PARQUET    = "documentation/plots/umap_stage3_pages.parquet"
VLM_CACHE  = "E:/vlm_cache"
RCNN_ROOT  = "E:/Comic_Analysis_Results_v2/stage3_json"
OUT_FILE   = "documentation/plots/vlm_vs_rcnn_panels.txt"
SAMPLE     = 1000
SEED       = 42

RCNN_SUBDIRS = [
    "CalibreComics_extracted",
    "CalibreComics_extracted_20251107",
    "CalibreComics_extracted/amazon",
    "CalibreComics_extracted_20251107/amazon",
    "NeonIchiban",
    "",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def count_vlm_panels(canonical_id):
    """Count panels with valid box_2d in vlm_cache."""
    path = os.path.join(VLM_CACHE, canonical_id.replace('/', os.sep) + '.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        panels = data.get('panels') or []
        return sum(
            1 for p in panels
            if isinstance(p, dict)
            and isinstance(p.get('box_2d'), (list, tuple))
            and len(p['box_2d']) == 4
        ), True
    except FileNotFoundError:
        return 0, False
    except Exception:
        return 0, True   # file exists, parse error

def count_rcnn_panels(canonical_id):
    """Count panels with valid bbox in stage3_json, trying all subdirs."""
    bare = canonical_id
    for pfx in ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/",
                "CalibreComics_extracted\\", "amazon/"]:
        if bare.startswith(pfx):
            bare = bare[len(pfx):]
            break

    tried = set()
    for sub in RCNN_SUBDIRS:
        base = os.path.join(RCNN_ROOT, sub) if sub else RCNN_ROOT
        for variant in [bare, canonical_id]:
            path = os.path.join(base, variant.replace('/', os.sep) + '.json')
            if path in tried:
                continue
            tried.add(path)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    panels = data.get('panels') or []
                    return sum(
                        1 for p in panels
                        if isinstance(p, dict)
                        and isinstance(p.get('bbox'), (list, tuple))
                        and len(p['bbox']) == 4
                        and p['bbox'][2] > 0 and p['bbox'][3] > 0
                    ), True
                except Exception:
                    return 0, True
    return 0, False

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(PARQUET)
    sample = df.sample(SAMPLE, random_state=SEED)

    rows = []
    for _, row in sample.iterrows():
        cid = row['canonical_id']
        stored = int(row['panel_count'])
        vlm_n,  vlm_found  = count_vlm_panels(cid)
        rcnn_n, rcnn_found = count_rcnn_panels(cid)
        rows.append({
            'cid': cid,
            'stored': stored,
            'vlm_found': vlm_found,
            'vlm_n': vlm_n,
            'rcnn_found': rcnn_found,
            'rcnn_n': rcnn_n,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    total = len(rows)
    vlm_missing  = [r for r in rows if not r['vlm_found']]
    rcnn_missing = [r for r in rows if not r['rcnn_found']]
    both_found   = [r for r in rows if r['vlm_found'] and r['rcnn_found']]

    vlm_wins  = [r for r in both_found if r['vlm_n'] > r['rcnn_n']]
    rcnn_wins = [r for r in both_found if r['rcnn_n'] > r['vlm_n']]
    tied      = [r for r in both_found if r['vlm_n'] == r['rcnn_n']]

    # Pages where stored panel_count=1 (VLM fallback) but RCNN had >1 panel
    vlm1_rcnn_more = [r for r in both_found if r['stored'] == 1 and r['rcnn_n'] > 1]
    # Pages where VLM had 0 box_2d but RCNN had panels
    vlm0_rcnn_has  = [r for r in both_found if r['vlm_n'] == 0 and r['rcnn_n'] > 0]

    # Avg panel counts where both found
    vlm_avg  = sum(r['vlm_n']  for r in both_found) / max(len(both_found), 1)
    rcnn_avg = sum(r['rcnn_n'] for r in both_found) / max(len(both_found), 1)

    # Distribution of (vlm_n - rcnn_n)
    diffs = [r['vlm_n'] - r['rcnn_n'] for r in both_found]
    diff_counts = collections.Counter(diffs)

    lines = []
    lines.append(f"VLM vs RCNN Panel Count Comparison")
    lines.append(f"Sample: {total} pages, seed={SEED}")
    lines.append("")
    lines.append("COVERAGE")
    lines.append("=" * 60)
    lines.append(f"  VLM JSON missing from cache:      {len(vlm_missing):4d}  ({100*len(vlm_missing)/total:.1f}%)")
    lines.append(f"  RCNN JSON missing from stage3_json:{len(rcnn_missing):4d}  ({100*len(rcnn_missing)/total:.1f}%)")
    lines.append(f"  Both found:                       {len(both_found):4d}  ({100*len(both_found)/total:.1f}%)")
    lines.append("")
    lines.append("WHERE BOTH FOUND: PANEL COUNT COMPARISON")
    lines.append("=" * 60)
    lines.append(f"  Avg VLM panels:   {vlm_avg:.2f}")
    lines.append(f"  Avg RCNN panels:  {rcnn_avg:.2f}")
    lines.append(f"  VLM > RCNN (VLM grounded more):  {len(vlm_wins):4d}  ({100*len(vlm_wins)/max(len(both_found),1):.1f}%)")
    lines.append(f"  RCNN > VLM (RCNN detected more): {len(rcnn_wins):4d}  ({100*len(rcnn_wins)/max(len(both_found),1):.1f}%)")
    lines.append(f"  Tied (equal count):              {len(tied):4d}  ({100*len(tied)/max(len(both_found),1):.1f}%)")
    lines.append("")
    lines.append("THE KEY QUESTION: VLM panel_count=1 but RCNN had >1 panel")
    lines.append("=" * 60)
    lines.append(f"  Stored panel_count=1, RCNN had >1: {len(vlm1_rcnn_more):4d}  ({100*len(vlm1_rcnn_more)/total:.1f}% of sample)")
    lines.append(f"  VLM grounded 0, RCNN had panels:   {len(vlm0_rcnn_has):4d}  ({100*len(vlm0_rcnn_has)/total:.1f}% of sample)")
    lines.append("")
    lines.append("DIFFERENCE DISTRIBUTION (VLM_n - RCNN_n), most common:")
    for diff, cnt in sorted(diff_counts.items(), key=lambda x: -x[1])[:15]:
        bar = '#' * min(cnt, 40)
        lines.append(f"  {diff:+3d}  {cnt:4d}  {bar}")
    lines.append("")
    lines.append("EXAMPLES: stored=1, RCNN had >1 panel (VLM missed grounding):")
    for r in vlm1_rcnn_more[:10]:
        lines.append(f"  VLM={r['vlm_n']} RCNN={r['rcnn_n']}  {r['cid']}")
    lines.append("")
    lines.append("EXAMPLES: VLM wins (more panels than RCNN):")
    for r in sorted(vlm_wins, key=lambda x: x['vlm_n']-x['rcnn_n'], reverse=True)[:10]:
        lines.append(f"  VLM={r['vlm_n']} RCNN={r['rcnn_n']}  {r['cid']}")
    lines.append("")
    lines.append("EXAMPLES: RCNN wins (more panels than VLM):")
    for r in sorted(rcnn_wins, key=lambda x: x['rcnn_n']-x['vlm_n'], reverse=True)[:10]:
        lines.append(f"  VLM={r['vlm_n']} RCNN={r['rcnn_n']}  {r['cid']}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == '__main__':
    main()
