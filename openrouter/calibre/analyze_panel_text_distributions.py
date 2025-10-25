#!/usr/bin/env python3
"""
Analyze panel and text distributions from analysis CSV(s) (Calibre/Amazon) to inform
Zarr embedding schema decisions (max_panels, max_text_len, recommended percentiles).

This script reads one or more analysis CSV files (those produced by
`create_perfect_match_filter_calibre_v2.py`), computes distributions for:
 - rcnn_panels (per-page panel count from R-CNN)
 - vlm_panels (per-page panel count from VLM JSON)
 - text_coverage (fraction of panels with text)
 - vlm_panels_with_text and breakdowns (dialogue/narration/sfx)

Additionally, when `--sample_jsons N` is provided and VLM JSON paths are available
(as `vlm_json_path` column) or inline (`vlm_data` column), it will sample up to N
matched JSONs and compute per-panel text length distributions to recommend a
reasonable `max_text_len` for the Zarr store.

Output:
 - Prints summary statistics
 - Writes JSON recommendations to `panel_text_distribution_recommendations.json`
 - Optional CSV histograms

Usage examples (PowerShell):
  python .\benchmarks\detections\openrouter\analyze_panel_text_distribution.py --csv calibre_rcnn_vlm_analysis_v2.csv amazon_rcnn_vlm_analysis.csv --sample_jsons 200

"""
from __future__ import annotations
import argparse
import json
import os
import sys
import math
import random
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


def safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def extract_panels_from_vlm(obj: Any) -> List[Dict[str, Any]]:
    """Robustly extract a list of panel dicts from a VLM JSON object (mirrors heuristics in analyzer)."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        if isinstance(obj.get('panels'), list):
            return obj.get('panels')
        # common nesting
        for k in ('result', 'page', 'data'):
            v = obj.get(k)
            if isinstance(v, dict) and isinstance(v.get('panels'), list):
                return v.get('panels')
            if isinstance(v, list) and v and isinstance(v[0], dict) and isinstance(v[0].get('panels'), list):
                return v[0].get('panels')
        # single panel as dict
        if any(k in obj for k in ('bbox', 'text', 'mask', 'polygon', 'dialogue', 'narration')):
            return [obj]
        return []
    if isinstance(obj, list):
        # list of panels
        if obj and isinstance(obj[0], dict) and any(k in obj[0] for k in ('bbox', 'text', 'dialogue', 'narration')):
            return obj
        # single wrapper list
        if len(obj) == 1 and isinstance(obj[0], dict):
            return extract_panels_from_vlm(obj[0])
        return []
    return []


def panel_text_length(panel: Any) -> int:
    """Estimate amount of text in a panel (character count)."""
    if panel is None:
        return 0
    def _s(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return ' '.join([_s(y) for y in x if isinstance(y, (str, list, dict))])
        if isinstance(x, dict):
            # join text-like fields
            out = []
            for k in ('dialogue', 'narration', 'caption', 'text', 'ocr', 'ocr_text'):
                v = x.get(k)
                if isinstance(v, str):
                    out.append(v)
                elif isinstance(v, list):
                    out.extend([str(t) for t in v if isinstance(t, str)])
            return ' '.join(out)
        return ''

    # Speakers list
    if isinstance(panel, dict):
        total = []
        if isinstance(panel.get('dialogue'), str):
            total.append(panel.get('dialogue'))
        if isinstance(panel.get('narration'), str):
            total.append(panel.get('narration'))
        if isinstance(panel.get('caption'), str):
            total.append(panel.get('caption'))
        if isinstance(panel.get('text'), str):
            total.append(panel.get('text'))
        for okey in ('ocr', 'ocr_text', 'ocr_lines', 'texts'):
            v = panel.get(okey)
            if isinstance(v, str):
                total.append(v)
            elif isinstance(v, list):
                total.extend([str(t) for t in v if isinstance(t, str)])
        speakers = panel.get('speakers')
        if isinstance(speakers, list):
            for sp in speakers:
                if isinstance(sp, dict):
                    dlg = sp.get('dialogue') or sp.get('text')
                    if isinstance(dlg, str):
                        total.append(dlg)
        # Fallback: convert any string fields
        for k, v in panel.items():
            if k not in ('bbox', 'mask', 'polygon') and isinstance(v, str) and len(v) > 0:
                # already captured many, but include any other strings
                if k not in ('dialogue','narration','caption','text','ocr','ocr_text'):
                    total.append(v)
        joined = ' '.join(total)
        return len(joined)
    elif isinstance(panel, list):
        return len(' '.join([str(x) for x in panel if isinstance(x, str)]))
    elif isinstance(panel, str):
        return len(panel)
    return 0


def extract_speech_type_lengths(panel: Any) -> Dict[str, int]:
    """Return a dict mapping speech_type -> character length for text found for that type in a panel.

    speech types we try to capture: dialogue, narration, caption, sfx, ocr, other
    """
    out = defaultdict(int)
    if panel is None:
        return out

    # Panel-level fields
    if isinstance(panel, dict):
        # direct fields
        for key, stype in (('dialogue', 'dialogue'), ('narration', 'narration'), ('caption', 'caption'), ('text', 'other')):
            v = panel.get(key)
            if isinstance(v, str) and v.strip():
                out[stype] += len(v)

        # ocr-like fields
        for okey in ('ocr', 'ocr_text', 'ocr_lines', 'texts'):
            v = panel.get(okey)
            if isinstance(v, str) and v.strip():
                out['ocr'] += len(v)
            elif isinstance(v, list):
                out['ocr'] += sum(len(str(x)) for x in v if isinstance(x, str))

        # speakers list: each may carry a speech_type and dialogue/text
        speakers = panel.get('speakers')
        if isinstance(speakers, list):
            for sp in speakers:
                if not isinstance(sp, dict):
                    continue
                st = sp.get('speech_type') or sp.get('type') or sp.get('speech') or 'dialogue'
                st = str(st).lower() if st is not None else 'dialogue'
                # normalize some labels
                if 'narr' in st:
                    st = 'narration'
                elif 'cap' in st or st == 'caption':
                    st = 'caption'
                elif 'sfx' in st or 'sound' in st:
                    st = 'sfx'
                elif 'ocr' in st:
                    st = 'ocr'
                else:
                    st = 'dialogue'

                text = ''
                # try common fields
                for k in ('dialogue', 'text', 'speech', 'utterance', 'caption'):
                    v = sp.get(k)
                    if isinstance(v, str) and v.strip():
                        text = v
                        break
                if text:
                    out[st] += len(text)

    # convert defaultdict to normal dict
    return dict(out)


def analyze_csv_paths(csv_paths: List[str], sample_jsons: int = 200, use_inline_vlm: bool = False, vlm_json_field: str = 'vlm_json_path', out_dir: str = 'panel_text_distribution_results') -> Dict[str, Any]:
    # Read each CSV, compute per-source stats (including sampling) and then combined stats
    dfs = []
    src_paths = []
    failed_paths = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"Warning: CSV not found: {p}")
            failed_paths.append(p)
            continue
        print(f"Reading CSV: {p}")
        try:
            df = pd.read_csv(p)
            print(f"  rows={len(df)}")
            dfs.append(df)
            src_paths.append(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            failed_paths.append(p)
            continue
    if failed_paths:
        raise RuntimeError(f"One or more CSVs could not be read: {failed_paths}")
    if not dfs:
        raise RuntimeError('No CSVs loaded')

    # ensure out_dir exists
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_stats: Dict[str, Any] = {'per_source': {}, 'combined': {}}

    def process_one(df: pd.DataFrame, src_name: str) -> Dict[str, Any]:
        stats_local: Dict[str, Any] = {}
        stats_local['summary'] = summarize_df(df)

        # CSV-driven aggregates
        type_cols = {
            'dialogue': 'vlm_text_dialogue_panels',
            'narration': 'vlm_text_narration_panels',
            'sfx': 'vlm_text_sfx_panels',
            'panels_with_text': 'vlm_panels_with_text'
        }
        stats_local['vlm_panel_type_counts'] = {}
        for label, col in type_cols.items():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors='coerce').fillna(0)
                stats_local['vlm_panel_type_counts'][label] = {
                    'mean': float(s.mean()),
                    'median': float(s.median()),
                    'p95': float(np.percentile(s,95)),
                    'count_nonzero': int((s>0).sum()),
                }

        # Sampling within this source
        text_lengths = []
        panels_sampled = 0
        lengths_by_type: Dict[str, List[int]] = defaultdict(list)
        if sample_jsons and sample_jsons > 0:
            if use_inline_vlm:
                candidates = [row for _, row in df.iterrows() if isinstance(row.get('vlm_data'), str) and row.get('vlm_data')]
            else:
                candidates = [row for _, row in df.iterrows() if isinstance(row.get(vlm_json_field), str) and row.get(vlm_json_field)]
            print(f"  [{src_name}] Found {len(candidates)} rows with VLM JSON info (inline={use_inline_vlm})")
            if candidates:
                sample_size = min(sample_jsons, len(candidates))
                sampled = random.sample(candidates, sample_size)
                for row in sampled:
                    try:
                        vlm_obj = None
                        if use_inline_vlm:
                            try:
                                vlm_obj = json.loads(row.get('vlm_data'))
                            except Exception:
                                import ast
                                try:
                                    vlm_obj = ast.literal_eval(row.get('vlm_data'))
                                except Exception:
                                    vlm_obj = None
                        else:
                            path = row.get(vlm_json_field)
                            if isinstance(path, str) and os.path.exists(path):
                                vlm_obj = safe_load_json(path)
                        if vlm_obj is None:
                            continue
                        panels = extract_panels_from_vlm(vlm_obj)
                        for p in panels:
                            l = panel_text_length(p)
                            text_lengths.append(l)
                            per_type = extract_speech_type_lengths(p)
                            for t, ll in per_type.items():
                                try:
                                    lengths_by_type[t].append(int(ll))
                                except Exception:
                                    pass
                        panels_sampled += len(panels)
                    except Exception:
                        continue
                print(f"  [{src_name}] Sampled {len(text_lengths)} panel-level text lengths from {sample_size} JSONs (panels sampled: {panels_sampled})")

        # Summarize sampled lengths
        if text_lengths:
            arr = np.array(text_lengths)
            q = [50,75,90,95,97,99,100]
            pct = {f'p{int(x)}': int(np.percentile(arr, x)) for x in q}
            stats_local['panel_text_length_percentiles'] = pct
            stats_local['panel_text_length_mean'] = float(arr.mean())
            stats_local['panel_text_length_median'] = int(np.median(arr))
            stats_local['panel_text_length_count'] = int(len(arr))
            rec = int(pct['p95'])
            rec_rounded = int(math.ceil(rec / 16.0) * 16)
            rec_rounded = min(max(rec_rounded, 64), 4096)
            stats_local['recommend_max_text_len'] = rec_rounded
        else:
            stats_local['panel_text_length_percentiles'] = {}
            stats_local['recommend_max_text_len'] = None

        stats_local['panel_text_length_percentiles_by_type'] = {}
        if lengths_by_type:
            for st, vals in lengths_by_type.items():
                try:
                    a = np.array(vals)
                    q = [50,75,90,95,97,99,100]
                    pct = {f'p{int(x)}': int(np.percentile(a, x)) for x in q}
                    stats_local['panel_text_length_percentiles_by_type'][st] = {
                        'count': int(a.size),
                        'mean': float(a.mean()),
                        'median': int(np.median(a)),
                        'percentiles': pct
                    }
                except Exception:
                    continue

        # panels per page
        def safe_int_series(df_local, col):
            if col in df_local.columns:
                return pd.to_numeric(df_local[col], errors='coerce').fillna(0).astype(int)
            return pd.Series([], dtype=int)
        rcnn_s = safe_int_series(df, 'rcnn_panels')
        vlm_s = safe_int_series(df, 'vlm_panels')
        combined_max = np.maximum(rcnn_s.values if len(rcnn_s)>0 else np.array([0]), vlm_s.values if len(vlm_s)>0 else np.array([0]))
        if combined_max.size > 0:
            stats_local['panels_per_page'] = {
                'mean': float(combined_max.mean()),
                'median': int(np.median(combined_max)),
                'p90': int(np.percentile(combined_max, 90)),
                'p95': int(np.percentile(combined_max, 95)),
                'p99': int(np.percentile(combined_max, 99)),
                'max': int(combined_max.max()),
            }
            rec_panels = int(np.percentile(combined_max, 95))
            rec_panels = int(math.ceil(rec_panels / 4.0) * 4)
            stats_local['recommend_max_panels'] = max(rec_panels, 4)
            try:
                hist = Counter(int(x) for x in combined_max)
                hist_items = sorted(hist.items())
                hist_df = pd.DataFrame(hist_items, columns=['panels', 'count'])
                hist_out = os.path.join(out_dir, f"panels_histogram_{src_name}.csv")
                hist_df.to_csv(hist_out, index=False)
                stats_local['panels_histogram_csv'] = hist_out
            except Exception:
                pass
        else:
            stats_local['panels_per_page'] = {}
            stats_local['recommend_max_panels'] = None

        # text_coverage
        if 'text_coverage' in df.columns:
            tc = pd.to_numeric(df['text_coverage'], errors='coerce').fillna(0.0)
            stats_local['text_coverage'] = {
                'mean': float(tc.mean()),
                'median': float(tc.median()),
                'p90': float(np.percentile(tc,90)),
                'p95': float(np.percentile(tc,95)),
                'count': int(len(tc)),
            }
        else:
            stats_local['text_coverage'] = {}

        out_json = os.path.join(out_dir, f"{src_name}_panel_text_distribution_recommendations.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(stats_local, f, indent=2)
        print(f"  Wrote per-source recommendations: {out_json}")
        return stats_local

    # process sources
    for path, df in zip(src_paths, dfs):
        src_name = os.path.splitext(os.path.basename(path))[0]
        all_stats['per_source'][src_name] = process_one(df, src_name)

    # combined
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Loaded {len(all_df)} rows across {len(dfs)} CSV(s)")
    combined = {}
    combined['summary'] = summarize_df(all_df)
    # combined CSV-driven aggregates
    type_cols = {
        'dialogue': 'vlm_text_dialogue_panels',
        'narration': 'vlm_text_narration_panels',
        'sfx': 'vlm_text_sfx_panels',
        'panels_with_text': 'vlm_panels_with_text'
    }
    combined['vlm_panel_type_counts'] = {}
    for label, col in type_cols.items():
        if col in all_df.columns:
            s = pd.to_numeric(all_df[col], errors='coerce').fillna(0)
            combined['vlm_panel_type_counts'][label] = {
                'mean': float(s.mean()),
                'median': float(s.median()),
                'p95': float(np.percentile(s,95)),
                'count_nonzero': int((s>0).sum()),
            }

    # panels per page combined
    def safe_int_series(df_local, col):
        if col in df_local.columns:
            return pd.to_numeric(df_local[col], errors='coerce').fillna(0).astype(int)
        return pd.Series([], dtype=int)
    rcnn_s = safe_int_series(all_df, 'rcnn_panels')
    vlm_s = safe_int_series(all_df, 'vlm_panels')
    combined_max = np.maximum(rcnn_s.values if len(rcnn_s)>0 else np.array([0]), vlm_s.values if len(vlm_s)>0 else np.array([0]))
    if combined_max.size > 0:
        combined['panels_per_page'] = {
            'mean': float(combined_max.mean()),
            'median': int(np.median(combined_max)),
            'p90': int(np.percentile(combined_max, 90)),
            'p95': int(np.percentile(combined_max, 95)),
            'p99': int(np.percentile(combined_max, 99)),
            'max': int(combined_max.max()),
        }
        rec_panels = int(np.percentile(combined_max, 95))
        rec_panels = int(math.ceil(rec_panels / 4.0) * 4)
        combined['recommend_max_panels'] = max(rec_panels, 4)
        try:
            hist = Counter(int(x) for x in combined_max)
            hist_items = sorted(hist.items())
            hist_df = pd.DataFrame(hist_items, columns=['panels', 'count'])
            hist_out = os.path.join(out_dir, f"panels_histogram_combined.csv")
            hist_df.to_csv(hist_out, index=False)
            combined['panels_histogram_csv'] = hist_out
        except Exception:
            pass
    else:
        combined['panels_per_page'] = {}
        combined['recommend_max_panels'] = None

    # combine per-source sampled percentiles (take conservative max p95)
    p95_vals = []
    for src, s in all_stats['per_source'].items():
        pct = s.get('panel_text_length_percentiles', {})
        if pct and 'p95' in pct:
            p95_vals.append(int(pct['p95']))
    if p95_vals:
        combined_p95 = int(max(p95_vals))
        combined['panel_text_length_percentiles'] = {'p95_across_sources_max': combined_p95}
        rec = combined_p95
        rec_rounded = int(math.ceil(rec / 16.0) * 16)
        rec_rounded = min(max(rec_rounded, 64), 4096)
        combined['recommend_max_text_len'] = rec_rounded
    else:
        combined['recommend_max_text_len'] = None

    combined_out = os.path.join(out_dir, 'panel_text_distribution_recommendations_combined.json')
    with open(combined_out, 'w', encoding='utf-8') as f:
        json.dump({'per_source': all_stats['per_source'], 'combined': combined}, f, indent=2)
    print(f"Wrote combined recommendations: {combined_out}")

    return {'per_source': all_stats['per_source'], 'combined': combined}


def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    cols = df.columns.tolist()
    out['rows'] = int(len(df))
    for col in ('rcnn_panels', 'vlm_panels', 'vlm_panels_with_text', 'vlm_text_dialogue_panels', 'vlm_text_narration_panels', 'vlm_text_sfx_panels'):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').fillna(0)
            out[f'{col}_mean'] = float(s.mean())
            out[f'{col}_median'] = float(s.median())
            out[f'{col}_p95'] = float(np.percentile(s,95))
    if 'text_coverage' in df.columns:
        tc = pd.to_numeric(df['text_coverage'], errors='coerce').fillna(0.0)
        out['text_coverage_mean'] = float(tc.mean())
        out['text_coverage_median'] = float(tc.median())
        out['text_coverage_p95'] = float(np.percentile(tc,95))
    return out


def main():
    ap = argparse.ArgumentParser(description='Analyze panel/text distributions and recommend zarr schema values')
    ap.add_argument('--csv', nargs='+', required=True, help='One or more analysis CSV paths')
    ap.add_argument('--sample_jsons', type=int, default=200, help='Number of VLM JSONs to sample for per-panel text length (default 200)')
    ap.add_argument('--use_inline_vlm', action='store_true', help='Parse inline vlm_data column instead of reading vlm_json_path files')
    ap.add_argument('--vlm_json_field', default='vlm_json_path', help='Column name for VLM JSON path (default: vlm_json_path)')
    ap.add_argument('--out_dir', default='panel_text_distribution_results', help='Directory to write per-source and combined JSON/CSV results')
    args = ap.parse_args()
    stats = analyze_csv_paths(args.csv, sample_jsons=args.sample_jsons, use_inline_vlm=args.use_inline_vlm, vlm_json_field=args.vlm_json_field, out_dir=args.out_dir)

    combined = stats.get('combined', {})
    print('\n=== Recommendations Summary (combined) ===')
    print(json.dumps({
        'recommend_max_panels': combined.get('recommend_max_panels'),
        'recommend_max_text_len': combined.get('recommend_max_text_len'),
        'panels_per_page': combined.get('panels_per_page'),
        'panel_text_length_percentiles': combined.get('panel_text_length_percentiles')
    }, indent=2))


if __name__ == '__main__':
    main()
