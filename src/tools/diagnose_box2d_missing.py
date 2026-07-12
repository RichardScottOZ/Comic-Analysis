import os, json, random, pandas as pd
from pathlib import Path

PARQUET   = "documentation/plots/umap_stage3_pages.parquet"
VLM_CACHE = "E:/vlm_cache"
OUT_FILE  = "documentation/plots/vlm_box2d_diagnosis.txt"
SAMPLE    = 500
SEED      = 42

def check_json(path):
    """
    Returns (found, total_panels, panels_with_box2d, panels_missing_box2d)
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        panels = data.get('panels') or []
        if not isinstance(panels, list):
            panels = []
        total = len(panels)
        has_box2d = sum(
            1 for p in panels
            if isinstance(p, dict)
            and isinstance(p.get('box_2d'), (list, tuple))
            and len(p['box_2d']) == 4
        )
        return True, total, has_box2d, total - has_box2d
    except FileNotFoundError:
        return False, 0, 0, 0
    except Exception:
        return True, -1, -1, -1   # parse error

def vlm_path(canonical_id):
    return os.path.join(VLM_CACHE, canonical_id.replace('/', os.sep) + '.json')

def main():
    df = pd.read_parquet(PARQUET)
    df1 = df[df['panel_count'] == 1].sample(SAMPLE, random_state=SEED)

    cats = {
        'json_missing':          [],   # not in vlm_cache at all
        'json_parse_error':      [],   # file exists but bad JSON
        'genuinely_1_panel':     [],   # 1 panel, has box_2d
        'box2d_missing_all':     [],   # panels exist but NO box_2d on any
        'box2d_partial':         [],   # some panels have box_2d, some don't
        'panels_empty':          [],   # panels array is empty
    }

    for _, row in df1.iterrows():
        cid  = row['canonical_id']
        path = vlm_path(cid)
        found, total, has_box2d, missing_box2d = check_json(path)

        if not found:
            cats['json_missing'].append(cid)
        elif total < 0:
            cats['json_parse_error'].append(cid)
        elif total == 0:
            cats['panels_empty'].append(cid)
        elif total == 1 and has_box2d == 1:
            cats['genuinely_1_panel'].append(cid)
        elif has_box2d == 0:
            cats['box2d_missing_all'].append((cid, total))
        else:
            cats['box2d_partial'].append((cid, total, has_box2d))

    total_sample = len(df1)

    lines = []
    lines.append(f"panel_count=1 pages in parquet: {len(df[df['panel_count']==1]):,}")
    lines.append(f"Sample size: {total_sample}")
    lines.append("")
    lines.append("BREAKDOWN")
    lines.append("=" * 60)

    def pct(lst): return 100 * len(lst) / total_sample

    lines.append(f"  json_missing (not in vlm_cache):    {len(cats['json_missing']):4d}  ({pct(cats['json_missing']):.1f}%)")
    lines.append(f"  json_parse_error:                   {len(cats['json_parse_error']):4d}  ({pct(cats['json_parse_error']):.1f}%)")
    lines.append(f"  panels array empty:                 {len(cats['panels_empty']):4d}  ({pct(cats['panels_empty']):.1f}%)")
    lines.append(f"  panels exist, ALL missing box_2d:  {len(cats['box2d_missing_all']):4d}  ({pct(cats['box2d_missing_all']):.1f}%)")
    lines.append(f"  panels exist, PARTIAL box_2d:      {len(cats['box2d_partial']):4d}  ({pct(cats['box2d_partial']):.1f}%)")
    lines.append(f"  genuinely 1 panel with box_2d:     {len(cats['genuinely_1_panel']):4d}  ({pct(cats['genuinely_1_panel']):.1f}%)")
    lines.append("")

    lines.append("EXAMPLES: panels exist but ALL missing box_2d (first 10):")
    for cid, total in cats['box2d_missing_all'][:10]:
        lines.append(f"  {cid}  (panels in array: {total})")
    lines.append("")

    lines.append("EXAMPLES: panels exist, PARTIAL box_2d (first 10):")
    for cid, total, has in cats['box2d_partial'][:10]:
        lines.append(f"  {cid}  (panels: {total}, has box_2d: {has})")
    lines.append("")

    lines.append("EXAMPLES: json_missing (first 10):")
    for cid in cats['json_missing'][:10]:
        lines.append(f"  {cid}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved to {OUT_FILE}")

if __name__ == '__main__':
    main()
