"""
Analyze R-CNN vs VLM alignment for Calibre-style comics.

This merges the robust VLM key loading and generic page-key normalization
with the detailed metrics/scoring from analyze_rcnn_vlm_alignment.py.

Key points:
- Recursively loads VLM JSON files and creates multiple lookup keys per file
  based on common Calibre/comic naming patterns (jpg4cbz, page_NNN, numeric, etc.).
- Creates a normalized image key from COCO image paths for O(1) exact lookup,
  then applies a small, bounded fuzzy fallback using folder tokens + number.
- Computes alignment metrics (panel count ratio, text coverage, density) and
  assigns a quality score comparable to the original analyzer.
- No series-specific logic; only general patterns.
"""

from __future__ import annotations

import os
import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Any
from tqdm import tqdm
import multiprocessing as mp
import difflib


# ---------------------------
# COCO loading (from original)
# ---------------------------
def load_coco_data(coco_path: str) -> Tuple[Dict[int, str], Dict[int, list], Dict[int, dict]]:
    print("Loading COCO data...")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Categories
    cat_map: Dict[int, str] = {}
    for c in coco.get('categories', []):
        cat_map[c['id']] = c['name']

    # Annotations grouped by image
    anns_by_img: Dict[int, list] = defaultdict(list)
    for a in coco.get('annotations', []):
        anns_by_img[a['image_id']].append(a)

    # Images map
    imginfo: Dict[int, dict] = {}
    if isinstance(coco.get('images'), list) and coco['images'] and isinstance(coco['images'][0], dict):
        for im in coco['images']:
            iid = im['id']
            p = im.get('file_name', str(iid))
            imginfo[iid] = {
                'path': p.replace('\\', '/'),
                'width': im.get('width'),
                'height': im.get('height'),
            }
    return cat_map, anns_by_img, imginfo


# -------------------------------------------
# VLM loading with robust, generic key mapping
# -------------------------------------------
def _normalize_token(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r'[\s_-]+', '_', s)


def _extract_trailing_number(s: str) -> str | None:
    """Return the trailing number (after '_' or '-' or end), else the last number found.
    Example: 'foo_001' -> '001', '2000ad_regened_001' -> '001', 'bar200' -> '200'.
    """
    m = re.search(r'(?:^|[_\-])(\d+)$', s)
    if m:
        return m.group(1)
    nums = re.findall(r'(\d+)', s)
    return nums[-1] if nums else None


def _collapse_double_prefix(normalized_base: str) -> str:
    """If the normalized base looks like X_Y_X_Y_Z (duplicated prefix), collapse to X_Y_Z.
    Only operates on already-normalized token strings separated by '_'.
    """
    tokens = normalized_base.split('_')
    if not tokens:
        return normalized_base
    last = tokens[-1]
    rem = tokens[:-1]
    if rem and rem == rem:  # dummy to keep structure clear
        pass
    if rem and len(rem) % 2 == 0:
        half = len(rem) // 2
        first, second = rem[:half], rem[half:]
        if first == second:
            return '_'.join(first + [last])
    return normalized_base


def _series_core_candidates(normalized_base: str) -> List[str]:
    """Heuristically derive 'series core' tokens from a normalized base name.
    We look at tokens before the trailing number and return candidates such as:
    - last two non-numeric tokens joined with '_'
    - last one non-numeric token
    This helps build inverted-index keys like 'series_core_001'.
    """
    toks = [t for t in normalized_base.split('_') if t]
    if not toks:
        return []
    # find trailing number index
    tnum = _extract_trailing_number(normalized_base)
    if not tnum:
        return []
    # locate the last occurrence of tnum as a full token if present
    end_idx = len(toks)
    for i in range(len(toks)-1, -1, -1):
        if toks[i].isdigit() and toks[i] == tnum:
            end_idx = i
            break
    pre = toks[:end_idx]
    # take last two/one non-numeric tokens
    nn = [t for t in pre if not t.isdigit()]
    cands: List[str] = []
    if len(nn) >= 2:
        cands.append(f"{nn[-2]}_{nn[-1]}")
    if len(nn) >= 1:
        cands.append(nn[-1])
    # dedup preserve order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def _normalize_image_key(img_path: str) -> str:
    """Create a generic key for an image path: <last_folder>_<page-token>.
    Patterns handled (order):
    - jpg4cbz_\d+
    - page[ _-]*\d+
    - pure digits filename (e.g., 001.jpg)
    - first number found in the filename
    - fallback: full cleaned filename
    """
    p = img_path.replace('\\', '/').lower()
    folder_path = os.path.dirname(p)
    if not folder_path:
        parts = p.split('/')
        folder_path = parts[-2] if len(parts) > 1 else ''
    folder = _normalize_token(os.path.basename(folder_path))

    filename = os.path.basename(p).lower()
    # Strip chained image extensions like .jpg.png.png
    base = re.sub(r'(?:\.(?:jpg|jpeg|png|webp|tif|tiff|bmp))+$', '', filename)

    m = re.search(r'(jpg4cbz_\d+)', base)
    if m:
        return f"{folder}_{m.group(1)}"

    m = re.search(r'page[_\-\s]*(\d+)', base)
    if m:
        return f"{folder}_{m.group(1).zfill(3)}"

    if base.isdigit():
        return f"{folder}_{base.zfill(3)}"

    tnum = _extract_trailing_number(base)
    if tnum:
        return f"{folder}_{tnum.zfill(3)}"

    clean = re.sub(r'[^a-z0-9]', '_', base)
    # also try to collapse duplicated series prefixes in the clean token
    clean = _collapse_double_prefix(_normalize_token(clean))
    return f"{folder}_{clean}"


def _vlm_key_variants(filename: str, parent_dir: str | None) -> List[str]:
    """Generate multiple key variants for a VLM JSON file based on general patterns."""
    base_raw = filename[:-5] if filename.lower().endswith('.json') else filename
    # Strip chained image extensions that may be retained in JSON names, e.g., '..._001.jpg.json' or '..._001.jpg.png.json'
    base_clean = re.sub(r'(?:\.(?:jpg|jpeg|png|webp|tif|tiff|bmp))+$', '', base_raw.lower())
    normalized = _normalize_token(base_clean)
    deduped = _collapse_double_prefix(normalized)
    keys = [normalized]
    if deduped != normalized:
        keys.append(deduped)

    if parent_dir:
        parent = _normalize_token(os.path.basename(parent_dir))
        keys.append(f"{parent}_{normalized}")
        # also try combining last two folders for better series scoping
        grand = _normalize_token(os.path.basename(os.path.dirname(parent_dir))) if os.path.dirname(parent_dir) else ''
        if grand:
            keys.append(f"{grand}_{parent}_{normalized}")

    # numeric patterns
    m_num = re.search(r'(\d+)', normalized)
    if m_num:
        num = m_num.group(1)
        if parent_dir:
            parent = _normalize_token(os.path.basename(parent_dir))
            keys.append(f"{parent}_{num}")
            for pad in (2,3,4,5):
                keys.append(f"{parent}_{num.zfill(pad)}")
        keys.append(num)
        for pad in (2,3,4,5):
            keys.append(num.zfill(pad))

        # page prefix patterns
        if 'page' in normalized:
            m_page = re.search(r'page[_\-\s]*(\d+)', normalized)
            if m_page and parent_dir:
                parent = _normalize_token(os.path.basename(parent_dir))
                page_num = m_page.group(1)
                keys.append(f"{parent}_{page_num}")

    # Also add variants based on trailing number (often page number)
    tnum = _extract_trailing_number(normalized)
    if tnum:
        if parent_dir:
            parent = _normalize_token(os.path.basename(parent_dir))
            keys.append(f"{parent}_{tnum}")
            for pad in (2,3,4,5):
                keys.append(f"{parent}_{tnum.zfill(pad)}")
        keys.append(tnum)
        for pad in (2,3,4,5):
            keys.append(tnum.zfill(pad))

    if parent_dir:
        rel_path = f"{os.path.basename(parent_dir)}/{base_clean}".lower()
        clean_path = _normalize_token(rel_path)
        keys.append(clean_path)
        # two-level relative path
        grand = os.path.basename(os.path.dirname(parent_dir)) if os.path.dirname(parent_dir) else ''
        if grand:
            rel2 = f"{grand}/{os.path.basename(parent_dir)}/{base_clean}".lower()
            keys.append(_normalize_token(rel2))

    return list(dict.fromkeys(keys))  # dedup, preserve order


def load_vlm_index(vlm_dir: str) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    print("Loading VLM data (recursive)...")
    index: Dict[str, dict] = {}
    inv_map: Dict[str, List[str]] = defaultdict(list)  # normalized base -> list of keys
    files = list(Path(vlm_dir).glob('**/*.json'))
    print(f"Found {len(files)} VLM files")
    for jf in tqdm(files, desc="Indexing VLM"):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            filename = os.path.basename(jf)
            parent_dir = os.path.dirname(jf)
            variants = _vlm_key_variants(filename, parent_dir)
            # primary before alts
            for key in variants:
                if key not in index:
                    index[key] = data
            # build inverted index using normalized base of the JSON filename
            base_raw = filename[:-5] if filename.lower().endswith('.json') else filename
            base_clean = re.sub(r'(?:\.(?:jpg|jpeg|png|webp|tif|tiff|bmp))+$', '', base_raw.lower())
            norm_base = _normalize_token(base_clean)
            norm_base_dedup = _collapse_double_prefix(norm_base)
            # Point to a couple of representative keys (primary + parent-prefixed if present)
            # to avoid overgrowth while still disambiguating series
            for k in variants[:2]:
                if k not in inv_map[norm_base]:
                    inv_map[norm_base].append(k)
                if k not in inv_map[norm_base_dedup]:
                    inv_map[norm_base_dedup].append(k)
            # Also index by guessed 'series_core_<tnum>' to match image bases like 'series_core_001'
            tnum = _extract_trailing_number(norm_base)
            if tnum:
                for core in _series_core_candidates(norm_base):
                    key = f"{core}_{tnum}"
                    for k in variants[:2]:
                        if k not in inv_map[key]:
                            inv_map[key].append(k)
        except Exception as e:
            print(f"Error loading {jf}: {e}")
    print(f"VLM index keys: {len(index)}")
    return index, inv_map


# ------------------------------
# Matching helpers (fast fallback)
# ------------------------------
def _fuzzy_candidates(image_key: str, img_path: str) -> List[str]:
    """Generate a small set of fallback keys to probe the index with.
    Keep this bounded to avoid O(N*M) scans.
    """
    keys = [image_key]
    # Try number variants
    fname = os.path.basename(img_path).lower()
    clean_base = re.sub(r'(?:\.(?:jpg|jpeg|png|webp|tif|tiff|bmp))+$', '', fname)
    tnum = _extract_trailing_number(clean_base)
    if tnum is not None:
        num = tnum
        folder = _normalize_token(os.path.basename(os.path.dirname(img_path)))
        for v in {num, num.lstrip('0') or '0', num.zfill(2), num.zfill(3), num.zfill(4), num.zfill(5)}:
            keys.append(f"{folder}_{v}")
        # also consider two-level folder scope
        parent = os.path.basename(os.path.dirname(img_path))
        grand = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        if grand:
            f2 = _normalize_token(f"{grand}_{parent}")
            for v in {num, num.lstrip('0') or '0', num.zfill(3), num.zfill(4), num.zfill(5)}:
                keys.append(f"{f2}_{v}")
    # Always include fully normalized base (to match VLM's filename-normalized key)
    norm_base = _normalize_token(clean_base)
    keys.append(norm_base)
    dedup_base = _collapse_double_prefix(norm_base)
    if dedup_base != norm_base:
        keys.append(dedup_base)
    # If jpg4cbz pattern present, try series_prefix + jpg4cbz_num
    mcbz = re.search(r'(jpg4cbz)_(\d+)', clean_base)
    if mcbz:
        num = mcbz.group(2)
        series_prefix = re.sub(r'(jpg4cbz_\d+).*$', '', clean_base).strip(' _-')
        series_prefix = _normalize_token(series_prefix)
        keys.append(f"{series_prefix}_jpg4cbz_{num}")
    return list(dict.fromkeys(keys))


# -------------------------------------
# Single-image analysis (metrics/scoring)
# -------------------------------------
def analyze_single_image(
    img_id: int,
    anns: List[dict],
    cat_map: Dict[int, str],
    imginfo: Dict[int, dict],
    vlm_index: Dict[str, dict],
    panel_thr: float = 0.75,
    panel_nms: float = 0.25,
) -> Dict[str, Any]:

    # image info
    if img_id in imginfo:
        path = imginfo[img_id]['path']
        W, H = imginfo[img_id]['width'], imginfo[img_id]['height']
    else:
        path = str(img_id)
        W, H = 1000, 1000

    # normalized key + bounded fallback
    img_key = _normalize_image_key(path)
    vlm_match = vlm_index.get(img_key)
    matched_key = ''
    tried_keys = [img_key]
    if not vlm_match:
        for k in _fuzzy_candidates(img_key, path):
            if k in tried_keys:
                continue
            tried_keys.append(k)
            vlm_match = vlm_index.get(k)
            if vlm_match:
                matched_key = k
                break
    if vlm_match and not matched_key:
        matched_key = img_key
    # If still no match, try inverted index by normalized base
    if vlm_match is None:
        try:
            # derive normalized base from image filename
            fname = os.path.basename(path).lower()
            clean_base = re.sub(r'(?:\.(?:jpg|jpeg|png|webp|tif|tiff|bmp))+$', '', fname)
            norm_base = _normalize_token(clean_base)
            inv = globals().get('G_INV_MAP')
            if inv and norm_base in inv:
                candidates = inv[norm_base]
                # If one candidate, pick it; if multiple, prefer matching page/jpg4cbz number
                if len(candidates) == 1:
                    matched_key = candidates[0]
                    vlm_match = vlm_index.get(matched_key)
                else:
                    # extract preferred page tokens
                    # 1) jpg4cbz_XXXX 2) trailing number
                    pref = None
                    mcbz = re.search(r'(jpg4cbz)_(\d+)', clean_base)
                    if mcbz:
                        pref = f"jpg4cbz_{mcbz.group(2)}"
                    else:
                        mtail = re.search(r'(?:^|[_\-])(\d+)$', clean_base)
                        if mtail:
                            pref = mtail.group(1)
                    if pref:
                        # prefer candidate containing the preferred token
                        filt = [k for k in candidates if pref in k]
                        pick = filt[0] if filt else candidates[0]
                    else:
                        pick = candidates[0]
                    matched_key = pick
                    vlm_match = vlm_index.get(matched_key)
            # Try series-core with trailing number: e.g., '2000ad_regened_001'
            if vlm_match is None and inv:
                tnum = _extract_trailing_number(norm_base)
                if tnum:
                    # derive core candidates from base
                    for core in _series_core_candidates(norm_base):
                        key = f"{core}_{tnum}"
                        if key in inv and inv[key]:
                            # prefer a candidate that also contains the folder token if possible
                            folder_token = _normalize_token(os.path.basename(os.path.dirname(path)))
                            picks = inv[key]
                            pick = None
                            if folder_token:
                                for k in picks:
                                    if folder_token in k:
                                        pick = k; break
                            if pick is None:
                                pick = picks[0]
                            matched_key = pick
                            vlm_match = vlm_index.get(matched_key)
                            if vlm_match:
                                break
        except Exception:
            pass

    # Log failing matches (no VLM found)
    try:
        if vlm_match is None:
            log_path = globals().get('G_LOG_PATH', None)
            if log_path:
                with open(log_path, 'a', encoding='utf-8') as lf:
                    # Extract quick features for later diagnostics
                    base = os.path.splitext(os.path.basename(path))[0].lower()
                    folder = os.path.basename(os.path.dirname(path)).lower()
                    # prefer trailing number for diagnostics clarity
                    tnum = _extract_trailing_number(base)
                    num = tnum if tnum is not None else ''
                    lf.write(f"NO_MATCH | image={path} | folder={folder} | base={base} | num={num} | key={img_key} | tried={';'.join(tried_keys)}\n")
    except Exception:
        pass

    # collect R-CNN detections
    panels, p_scores = [], []
    texts, t_scores = [], []
    balloons, b_scores = [], []
    chars, c_scores = [], []
    faces, f_scores = [], []

    for a in anns:
        cid = a['category_id']
        b = [float(x) for x in a['bbox']]
        s = float(a.get('score', 1.0))
        name = cat_map.get(cid, str(cid))
        if name == 'panel' and s >= panel_thr:
            panels.append(b); p_scores.append(s)
        elif name == 'balloon' and s >= 0.5:
            balloons.append(b); b_scores.append(s)
        elif name in ('text','onomatopoeia') and s >= 0.5:
            texts.append(b); t_scores.append(s)
        elif name in ('character','face') and s >= 0.6:
            if name == 'character': chars.append(b); c_scores.append(s)
            else: faces.append(b); f_scores.append(s)

    # NMS on panels (from original)
    if len(panels) > 1:
        import numpy as np

        def iou_xywh(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ax2, ay2 = ax+aw, ay+ah
            bx2, by2 = bx+bw, by+bh
            inter_w = max(0, min(ax2, bx2) - max(ax, bx))
            inter_h = max(0, min(ay2, by2) - max(ay, by))
            inter = inter_w * inter_h
            ua = aw*ah + bw*bh - inter + 1e-6
            return inter / ua

        def nms(boxes, scores, iou_thr):
            idxs = np.argsort(scores)[::-1]
            keep = []
            while len(idxs) > 0:
                i = idxs[0]
                keep.append(int(i))
                if len(idxs) == 1: break
                rest = idxs[1:]
                ious = np.array([iou_xywh(boxes[i], boxes[j]) for j in rest])
                idxs = rest[ious < iou_thr]
            return keep

        keep_idx = nms(panels, p_scores, iou_thr=panel_nms)
        panels = [panels[i] for i in keep_idx]
        p_scores = [p_scores[i] for i in keep_idx]

    # filter tiny panels
    clean = []
    for b in panels:
        x, y, w, h = b
        if w < 32 or h < 32: continue
        if W and H and (w * h) < 0.01 * (W * H): continue
        clean.append(b)
    panels = clean

    # VLM panels/text analysis
    vlm_panels = []
    vlm_panels_with_text = 0
    vlm_total_dialogue = 0
    vlm_total_narration = 0

    if vlm_match and isinstance(vlm_match, dict):
        vlm_panels = vlm_match.get('panels', []) or []
        for p in vlm_panels:
            has_text = False
            # dialogue
            for s in p.get('speakers', []) or []:
                if 'dialogue' in s and s['dialogue']:
                    vlm_total_dialogue += 1
                    has_text = True
            # narration/caption
            if p.get('caption'):
                vlm_total_narration += 1
                has_text = True
            if has_text:
                vlm_panels_with_text += 1

    # metrics
    rcnn_panel_count = len(panels)
    vlm_panel_count = len(vlm_panels)
    vlm_text_panel_count = vlm_panels_with_text

    panel_count_ratio = (vlm_panel_count / rcnn_panel_count) if rcnn_panel_count > 0 else 0.0
    text_coverage_ratio = (vlm_text_panel_count / rcnn_panel_count) if rcnn_panel_count > 0 else 0.0
    vlm_text_density = ((vlm_total_dialogue + vlm_total_narration) / vlm_panel_count) if vlm_panel_count > 0 else 0.0

    has_vlm_data = vlm_match is not None
    has_rcnn_panels = rcnn_panel_count > 0
    panel_alignment_good = 0.7 <= panel_count_ratio <= 1.3 if rcnn_panel_count > 0 else False
    text_coverage_good = text_coverage_ratio >= 0.3 if rcnn_panel_count > 0 else False
    vlm_quality_good = vlm_text_density >= 1.0 if vlm_panel_count > 0 else False

    quality_score = 0
    if has_rcnn_panels: quality_score += 1
    if has_vlm_data: quality_score += 1
    if panel_alignment_good: quality_score += 1
    if text_coverage_good: quality_score += 1
    if vlm_quality_good: quality_score += 1

    return {
        'image_id': img_id,
        'image_path': path,
        'image_size': f"{W}x{H}",
        'has_vlm_data': has_vlm_data,
        'matched_vlm_key': matched_key if vlm_match is not None else '',
        'rcnn_panel_count': rcnn_panel_count,
        'vlm_panel_count': vlm_panel_count,
        'vlm_text_panel_count': vlm_text_panel_count,
        'panel_count_ratio': panel_count_ratio,
        'text_coverage_ratio': text_coverage_ratio,
        'vlm_text_density': vlm_text_density,
        'rcnn_text_regions': len(texts),
        'rcnn_balloon_regions': len(balloons),
        'rcnn_character_regions': len(chars),
        'rcnn_face_regions': len(faces),
        'panel_alignment_good': panel_alignment_good,
        'text_coverage_good': text_coverage_good,
        'vlm_quality_good': vlm_quality_good,
        'quality_score': quality_score,
        'recommend_training': quality_score >= 4,
    }


# -----------------
# Main entry-point
# -----------------
def analyze_alignment_calibre(
    coco_path: str,
    vlm_dir: str,
    output_csv: str,
    limit: int | None = None,
    panel_thr: float = 0.75,
    panel_nms: float = 0.25,
    num_workers: int = 1,
):
    print("🔍 Analyzing R-CNN vs VLM Alignment (Calibre)")
    print("=" * 60)

    cat_map, anns_by_img, imginfo = load_coco_data(coco_path)
    vlm_index, inv_map = load_vlm_index(vlm_dir)

    print(f"Found {len(anns_by_img)} images with R-CNN data")
    print(f"VLM index size (keys): {len(vlm_index)}")

    results: List[dict] = []
    work_items = list(anns_by_img.items())
    if limit:
        work_items = work_items[:limit]
        print(f"Analyzing first {limit} images (limit mode)")

    # Prepare failure log file
    fail_log_path = output_csv.replace('.csv', '_match_failures.log')
    with open(fail_log_path, 'w', encoding='utf-8') as lf:
        lf.write('# No-match log\n')
    
    if num_workers and num_workers > 1:
        # Use globals to avoid repeatedly pickling large dicts per task
        def _init_worker(_cat_map, _imginfo, _vlm_index, _inv_map, _panel_thr, _panel_nms, _log_path):
            globals()['G_CAT_MAP'] = _cat_map
            globals()['G_IMGINFO'] = _imginfo
            globals()['G_VLM_INDEX'] = _vlm_index
            globals()['G_INV_MAP'] = _inv_map
            globals()['G_PANEL_THR'] = _panel_thr
            globals()['G_PANEL_NMS'] = _panel_nms
            globals()['G_LOG_PATH'] = _log_path

        def _worker(task):
            img_id, anns = task
            try:
                return analyze_single_image(
                    img_id, anns,
                    globals()['G_CAT_MAP'],
                    globals()['G_IMGINFO'],
                    globals()['G_VLM_INDEX'],
                    panel_thr=globals()['G_PANEL_THR'],
                    panel_nms=globals()['G_PANEL_NMS'],
                )
            except Exception as e:
                # Best-effort logging; return minimal record to keep pipeline moving
                return {
                    'image_id': img_id,
                    'image_path': globals()['G_IMGINFO'].get(img_id, {}).get('path', str(img_id)),
                    'image_size': '0x0',
                    'has_vlm_data': False,
                    'rcnn_panel_count': 0,
                    'vlm_panel_count': 0,
                    'vlm_text_panel_count': 0,
                    'panel_count_ratio': 0.0,
                    'text_coverage_ratio': 0.0,
                    'vlm_text_density': 0.0,
                    'rcnn_text_regions': 0,
                    'rcnn_balloon_regions': 0,
                    'rcnn_character_regions': 0,
                    'rcnn_face_regions': 0,
                    'panel_alignment_good': False,
                    'text_coverage_good': False,
                    'vlm_quality_good': False,
                    'quality_score': 0,
                    'recommend_training': False,
                }

        with mp.Pool(processes=num_workers, initializer=_init_worker,
                      initargs=(cat_map, imginfo, vlm_index, inv_map, panel_thr, panel_nms, fail_log_path)) as pool:
            for res in tqdm(pool.imap_unordered(_worker, work_items), total=len(work_items), desc="Analyzing images"):
                if res is not None:
                    results.append(res)
    else:
        # Single-process path
        globals()['G_LOG_PATH'] = fail_log_path
        globals()['G_INV_MAP'] = inv_map
        for img_id, anns in tqdm(work_items, desc="Analyzing images"):
            try:
                res = analyze_single_image(
                    img_id, anns, cat_map, imginfo, vlm_index,
                    panel_thr=panel_thr, panel_nms=panel_nms,
                )
                results.append(res)
            except Exception as e:
                print(f"Error analyzing image {img_id}: {e}")
                continue

    if not results:
        print("❌ No results to analyze!")
        return

    # Write CSV
    fieldnames = list(results[0].keys())
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊 Analysis Complete! Results saved to: {output_csv}")

    # Summary
    total_images = len(results)
    has_vlm = sum(1 for r in results if r['has_vlm_data'])
    good_alignment = sum(1 for r in results if r['panel_alignment_good'])
    good_text_coverage = sum(1 for r in results if r['text_coverage_good'])
    good_vlm_quality = sum(1 for r in results if r['vlm_quality_good'])
    recommend_training = sum(1 for r in results if r['recommend_training'])

    print(f"\n📈 Summary Statistics:")
    print(f"  Total images analyzed: {total_images}")
    print(f"  Images with VLM data: {has_vlm} ({has_vlm/total_images*100:.1f}%)")
    print(f"  Good panel alignment: {good_alignment} ({good_alignment/total_images*100:.1f}%)")
    print(f"  Good text coverage: {good_text_coverage} ({good_text_coverage/total_images*100:.1f}%)")
    print(f"  Good VLM quality: {good_vlm_quality} ({good_vlm_quality/total_images*100:.1f}%)")
    print(f"  Recommended for training: {recommend_training} ({recommend_training/total_images*100:.1f}%)")

    # Distribution
    quality_scores = [r['quality_score'] for r in results]
    dist = {i: quality_scores.count(i) for i in range(6)}
    print(f"\n🎯 Quality Score Distribution:")
    for score in range(6):
        count = dist.get(score, 0)
        print(f"  Score {score}: {count} images ({count/total_images*100:.1f}%)")

    # Panel counts
    rcnn_counts = [r['rcnn_panel_count'] for r in results if r['rcnn_panel_count'] > 0]
    vlm_counts = [r['vlm_panel_count'] for r in results if r['vlm_panel_count'] > 0]
    if rcnn_counts:
        print(f"\n📊 Panel Count Statistics:")
        print(f"  R-CNN panels - Avg: {sum(rcnn_counts)/len(rcnn_counts):.1f}, Min: {min(rcnn_counts)}, Max: {max(rcnn_counts)}")
    if vlm_counts:
        print(f"  VLM panels - Avg: {sum(vlm_counts)/len(vlm_counts):.1f}, Min: {min(vlm_counts)}, Max: {max(vlm_counts)}")

    # Filtered lists
    perfect_matches = [r for r in results if r['panel_count_ratio'] == 1.0]
    near_perfect = [r for r in results if 0.9 <= r['panel_count_ratio'] <= 1.1]
    high_quality = [r for r in results if r['quality_score'] >= 4]
    medium_quality = [r for r in results if r['quality_score'] == 3]

    # Print explicit counts for perfect and near-perfect matches
    print(f"\n✅ Match Counts:")
    print(f"  Perfect panel matches (ratio == 1.0): {len(perfect_matches)} ({len(perfect_matches)/total_images*100:.1f}%)")
    print(f"  Near-perfect panel matches (0.9–1.1): {len(near_perfect)} ({len(near_perfect)/total_images*100:.1f}%)")

    def _save_list(path: str, rows: List[dict]):
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(f"{r['image_path']}\n")

    _save_list(output_csv.replace('.csv', '_perfect_matches.txt'), perfect_matches)
    _save_list(output_csv.replace('.csv', '_near_perfect.txt'), near_perfect)
    _save_list(output_csv.replace('.csv', '_high_quality.txt'), high_quality)
    _save_list(output_csv.replace('.csv', '_medium_quality.txt'), medium_quality)

    print(f"\n💾 Filtered lists saved:")
    print(f"  Perfect matches: {output_csv.replace('.csv', '_perfect_matches.txt')} ({len(perfect_matches)})")
    print(f"  Near-perfect: {output_csv.replace('.csv', '_near_perfect.txt')} ({len(near_perfect)})")
    print(f"  High quality: {output_csv.replace('.csv', '_high_quality.txt')}")
    print(f"  Medium quality: {output_csv.replace('.csv', '_medium_quality.txt')}")

    # Optional post-mortem diagnostics on failures (bounded sample)
    fail_log = output_csv.replace('.csv', '_match_failures.log')
    if os.path.exists(fail_log):
        try:
            with open(fail_log, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith('#')]
            sample = lines[:100]
            if sample:
                print(f"\n🔎 Diagnostics on first {len(sample)} no-match cases:")
                vlm_keys = list(vlm_index.keys())
                for ln in sample[:10]:  # print first 10
                    # Parse quick fields
                    # Format: NO_MATCH | image=... | folder=... | base=... | num=... | key=... | tried=...
                    parts = {kv.split('=')[0].strip(): kv.split('=')[1].strip() for kv in ln.split('|')[1:] if '=' in kv}
                    folder_raw = parts.get('folder', '')
                    folder = _normalize_token(folder_raw)
                    base_raw = parts.get('base', '')
                    norm_base = _normalize_token(base_raw)
                    dedup_base = _collapse_double_prefix(norm_base)
                    num = parts.get('num', '')
                    # Report quick existence checks
                    has_norm_key = norm_base in vlm_index
                    has_dedup_key = dedup_base in vlm_index
                    inv = inv_map if 'inv_map' in locals() else {}
                    has_inv_norm = norm_base in inv
                    has_inv_dedup = dedup_base in inv
                    print(f"  image={parts.get('image','')} | folder={folder} | base={norm_base} | num={num}")
                    print(f"    ↳ direct base key exists: {has_norm_key}")
                    if dedup_base != norm_base:
                        print(f"    ↳ dedup base key exists: {has_dedup_key}")
                    if has_inv_norm:
                        for k in inv[norm_base][:2]:
                            print(f"    ↳ invmap(norm) cand: {k}")
                    if dedup_base != norm_base and has_inv_dedup:
                        for k in inv[dedup_base][:2]:
                            print(f"    ↳ invmap(dedup) cand: {k}")
                    # Candidate suggestions prioritized by folder and trailing number
                    candidates = []
                    if num:
                        num_vars = {num, num.lstrip('0') or '0', num.zfill(2), num.zfill(3), num.zfill(4), num.zfill(5)}
                        for nk in vlm_keys:
                            if folder and folder in nk and any(v in nk for v in num_vars):
                                candidates.append(nk)
                    if not candidates and folder:
                        close = difflib.get_close_matches(folder, [k.split('_')[0] for k in vlm_keys], n=5, cutoff=0.6)
                        for c in close:
                            for nk in vlm_keys:
                                if nk.startswith(c + '_'):
                                    candidates.append(nk)
                    for c in candidates[:3]:
                        print(f"    ↳ candidate: {c}")
        except Exception as e:
            print(f"Diagnostics failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze R-CNN vs VLM alignment (Calibre-style)')
    parser.add_argument('--coco', required=True, help='Path to COCO detection JSON')
    parser.add_argument('--vlm_dir', required=True, help='Directory containing VLM JSON files (recursive)')
    parser.add_argument('--output_csv', required=True, help='Output CSV file for analysis results')
    parser.add_argument('--limit', type=int, default=None, help='Limit analysis to first N images (default: all)')
    parser.add_argument('--panel_thr', type=float, default=0.75, help='Panel detection threshold')
    parser.add_argument('--panel_nms', type=float, default=0.25, help='Panel NMS IoU threshold')
    parser.add_argument('--num_workers', type=int, default=1, help='Parallel workers (Windows-safe; default: 1)')
    args = parser.parse_args()

    analyze_alignment_calibre(
        coco_path=args.coco,
        vlm_dir=args.vlm_dir,
        output_csv=args.output_csv,
        limit=args.limit,
        panel_thr=args.panel_thr,
        panel_nms=args.panel_nms,
        num_workers=args.num_workers,
    )
