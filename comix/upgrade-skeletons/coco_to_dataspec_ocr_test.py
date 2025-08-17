Love it. Let’s wire PaddleOCR into your multiprocessing converter so you can auto-fill dialogue/narration/SFX per panel using your CoMix detections (balloons, text, onomatopoeia, faces/characters). I’ve kept your filename handling, MP, and robustness, and added OCR as an optional module with clean flags.

What’s new
- Optional PaddleOCR integration (--ocr) with GPU/CPU toggle, language, confidence threshold, and scope (page vs panel OCR).
- Token bucketing:
  - dialogue: tokens inside balloon boxes
  - sfx: tokens inside onomatopoeia boxes OR outside balloons with big angle/size
  - narration: tokens inside panel but not in balloons/SFX
- Per-panel ocr_tokens including polygon, bbox (page coords), rel_bbox (panel-normalized), conf, angle, and assigned region.
- Fallback logic: if OCR has no content for a panel, we fall back to your VLM page JSON for dialogue/narration.

Install notes
- CPU: pip install paddleocr paddlepaddle
- GPU (CUDA): pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html then pip install paddleocr
- Windows note: if pip can’t find paddlepaddle-gpu, check Paddle’s official install page for the right wheel for your CUDA/driver.

Updated script (drop-in replacement)
- Adds OCR helpers and integrates into process_single_image.
- Adds CLI flags: --ocr, --ocr-gpu, --ocr-lang, --ocr-min-conf, --ocr-scope, --ocr-angle-thr, --ocr-sfx-size-frac.

```python
import json, os, re
from collections import defaultdict
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time
import math

# --------- Utils (same as original)
def norm_path(p: str) -> str:
    return p.replace('\\', '/')

def page_id_from_path(p: str) -> str:
    p = norm_path(p)
    base = os.path.splitext(p)[0]
    base = re.sub(r'[^A-Za-z0-9/_\-]+', '_', base)
    return base

def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay1 := ay, by1 := by))
    inter = inter_w * inter_h
    ua = aw*ah + bw*bh - inter + 1e-6
    return inter / ua

def nms(boxes: List[List[float]], scores: List[float], iou_thr: float) -> List[int]:
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

def box_center(b):
    return (b[0] + b[2]/2.0, b[1] + b[3]/2.0)

def vertical_overlap(a,b):
    ay1, ay2 = a[1], a[1]+a[3]
    by1, by2 = b[1], b[1]+b[3]
    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / max(1e-6, min(a[3], b[3]))

def horizontal_overlap(a,b):
    ax1, ax2 = a[0], a[0]+a[2]
    bx1, bx2 = b[0], b[0]+b[2]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return inter / max(1e-6, min(a[2], b[2]))

def compute_reading_order(panel_boxes: List[List[float]], rtl=False) -> List[int]:
    idxs = list(range(len(panel_boxes)))
    idxs.sort(key=lambda i: panel_boxes[i][1])
    rows = []
    cur = [idxs[0]] if idxs else []
    for i in idxs[1:]:
        if vertical_overlap(panel_boxes[cur[-1]], panel_boxes[i]) > 0.25:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    if cur: rows.append(cur)
    order = []
    for row in rows:
        row.sort(key=lambda i: panel_boxes[i][0], reverse=rtl)
        order.extend(row)
    return order

def build_adjacency(panel_boxes: List[List[float]], order: List[int], k_fallback=2) -> Tuple[np.ndarray, Dict[int, List[int]], List[int]]:
    N = len(panel_boxes)
    adj = np.zeros((N,N), dtype=np.int64)
    for pos,i in enumerate(order):
        if pos < len(order)-1:
            j = order[pos+1]; adj[i,j] = 1
    centers = [box_center(b) for b in panel_boxes]
    for i in range(N):
        bi, ci = panel_boxes[i], centers[i]
        left = right = above = below = None
        left_dx = right_dx = up_dy = down_dy = 1e9
        for j in range(N):
            if i==j: continue
            bj, cj = panel_boxes[j], centers[j]
            dx, dy = cj[0]-ci[0], cj[1]-ci[1]
            if dx < 0 and vertical_overlap(bi,bj) > 0.2 and abs(dx) < left_dx:
                left, left_dx = j, abs(dx)
            if dx > 0 and vertical_overlap(bi,bj) > 0.2 and dx < right_dx:
                right, right_dx = j, dx
            if dy < 0 and horizontal_overlap(bi,bj) > 0.2 and abs(dy) < up_dy:
                above, up_dy = j, abs(dy)
            if dy > 0 and horizontal_overlap(bi,bj) > 0.2 and dy < down_dy:
                below, down_dy = j, dy
        for j in [left, right, above, below]:
            if j is not None: adj[i,j] = 1
    C = np.array(centers)
    for i in range(N):
        if adj[i].sum() == 0 and N > 1:
            d = np.sqrt(((C - C[i])**2).sum(axis=1))
            nn = np.argsort(d)[1:min(k_fallback+1, N)]
            for j in nn: adj[i,int(j)] = 1
    neighbor_dict = {}
    for i in range(N):
        neighbor_dict[i] = {
            'left': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][0] < panel_boxes[i][0]],
            'right': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][0] > panel_boxes[i][0]],
            'above': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][1] < panel_boxes[i][1]],
            'below': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][1] > panel_boxes[i][1]],
        }
    next_idx = [-100]*N
    for pos,i in enumerate(order):
        if pos < len(order)-1:
            next_idx[i] = order[pos+1]
    return adj, neighbor_dict, next_idx

# --------- PaddleOCR integration
_OCR = None
_OCR_CFG = {}

def _ensure_ocr(use_gpu=False, lang='en'):
    global _OCR, _OCR_CFG
    if _OCR is not None:
        return _OCR
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise RuntimeError("PaddleOCR not installed. Install with `pip install paddleocr` "
                           "and `pip install paddlepaddle` (or paddlepaddle-gpu).") from e
    _OCR = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
    _OCR_CFG = {'use_gpu': use_gpu, 'lang': lang}
    return _OCR

def _poly_to_bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [float(x1), float(y1), float(x2-x1), float(y2-y1)]

def _point_in_box(px, py, box):
    x,y,w,h = box
    return (x <= px <= x+w) and (y <= py <= y+h)

def _center_of_poly(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return (float(sum(xs)/len(xs)), float(sum(ys)/len(ys)))

def _poly_angle_deg(poly):
    # approximate using top edge (p0->p1)
    p0, p1 = poly[0], poly[1]
    ang = math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))
    return abs(ang)

def _assign_tokens_ocr_to_buckets(tokens, panel_box, balloon_boxes, ono_boxes,
                                  angle_thr=15.0, sfx_size_frac=0.08, min_conf=0.5):
    """
    tokens: list of dicts {text, conf, poly, bbox}
    Returns: ocr_tokens (annotated) and text_buckets dict
    """
    px, py, pw, ph = panel_box
    # Precompute centers
    ann_tokens = []
    # Group dialogue by balloon
    dialogue_by_balloon = [[] for _ in balloon_boxes]
    narration_tokens, sfx_tokens = [], []
    for t in tokens:
        if t['conf'] < min_conf or not t['text'].strip():
            continue
        poly = t['poly']; bbox = t['bbox']
        cx, cy = _center_of_poly(poly)
        # discard tokens clearly outside panel
        if not _point_in_box(cx, cy, panel_box):
            continue
        angle = _poly_angle_deg(poly)
        # token height approx
        th = bbox[3]
        # Check onomatopoeia region first
        in_ono = any(_point_in_box(cx, cy, b) for b in ono_boxes)
        # Check balloons
        b_idx = -1
        for bi, bb in enumerate(balloon_boxes):
            if _point_in_box(cx, cy, bb):
                b_idx = bi; break
        region = None
        if in_ono:
            region = 'sfx'
            sfx_tokens.append(t['text'])
        elif b_idx >= 0:
            region = 'dialogue'
            dialogue_by_balloon[b_idx].append((cy, cx, t['text']))
        else:
            # Heuristic SFX: large/angled text outside balloons
            if angle > angle_thr or (ph > 1 and (th/ph) > sfx_size_frac):
                region = 'sfx'
                sfx_tokens.append(t['text'])
            else:
                region = 'narration'
                narration_tokens.append((cy, cx, t['text']))
        rel_bbox = [(bbox[0]-px)/max(1.0,pw), (bbox[1]-py)/max(1.0,ph), bbox[2]/max(1.0,pw), bbox[3]/max(1.0,ph)]
        ann_tokens.append({
            'text': t['text'], 'conf': float(t['conf']),
            'poly': poly, 'bbox': bbox, 'rel_bbox': rel_bbox,
            'angle': float(angle), 'region': region
        })

    # Sort and join text
    def _join_sorted(triples):
        triples.sort(key=lambda z: (z[0], z[1]))  # by y then x
        return ' '.join([z[2] for z in triples]).strip()

    dialogue = []
    for lst in dialogue_by_balloon:
        if lst:
            dialogue.append(_join_sorted(lst))
    narration = []
    if narration_tokens:
        narration_str = _join_sorted(narration_tokens)
        if narration_str:
            narration.append(narration_str)
    # sfx: keep as is (list of strings)
    sfx = []
    if sfx_tokens:
        # de-dup preserving order
        seen = set()
        for s in sfx_tokens:
            if s not in seen:
                seen.add(s); sfx.append(s)

    return ann_tokens, {'dialogue': dialogue, 'narration': narration, 'sfx': sfx}

def _run_ocr_on_image(img_array_or_path):
    ocr = _ensure_ocr(use_gpu=_OCR_CFG.get('use_gpu', False), lang=_OCR_CFG.get('lang', 'en'))
    res = ocr.ocr(img_array_or_path, cls=True)
    # PaddleOCR returns list per image: [ [ [points, (txt, conf)], ... ] ]
    lines = res[0] if isinstance(res, list) and len(res)>0 else []
    tokens = []
    for ln in lines:
        poly = [[float(p[0]), float(p[1])] for p in ln[0]]
        text = ln[1][0]
        conf = float(ln[1][1])
        bbox = _poly_to_bbox(poly)
        tokens.append({'text': text, 'conf': conf, 'poly': poly, 'bbox': bbox})
    return tokens

# --------- Single image processor (for multiprocessing)
def process_single_image(args):
    """Process a single image - designed for multiprocessing"""
    (
        img_id, anns, cat_map, imginfo, vlm_pages_dir,
        panel_thr, panel_nms, text_thr, char_thr, balloon_thr, rtl,
        output_dir, skip_existing,
        # OCR params
        enable_ocr, ocr_gpu, ocr_lang, ocr_min_conf, ocr_scope,
        ocr_angle_thr, ocr_sfx_size_frac
    ) = args
    
    # Resolve image path and size
    if img_id in imginfo:
        path = imginfo[img_id]['path']
        W, H = imginfo[img_id]['width'], imginfo[img_id]['height']
    else:
        path = norm_path(str(img_id))
        try:
            with Image.open(path) as im:
                W, H = im.size
        except Exception:
            alt = re.sub(r'\.jpg\.png$', '.jpg', path, flags=re.IGNORECASE)
            alt = re.sub(r'\.png\.jpg$', '.png', alt, flags=re.IGNORECASE)
            with Image.open(alt) as im:
                W, H = im.size
            path = alt

    # Check if output file already exists (for skip-existing)
    def create_unique_filename(image_path):
        clean_path = norm_path(image_path)
        path_parts = [p for p in clean_path.split('/') if p]
        if len(path_parts) >= 2:
            comic_name = re.sub(r'[<>:"/\\|?*]', '_', path_parts[-2])
            page_name = re.sub(r'[<>:"/\\|?*]', '_', os.path.splitext(path_parts[-1])[0])
            if '_' in page_name:
                parts = page_name.split('_')
                if len(parts) >= 2:
                    potential_page_num = parts[-1]
                    if potential_page_num.isdigit():
                        comic_from_filename = '_'.join(parts[:-1])
                        return f"{comic_from_filename}_page_{potential_page_num}.json"
            page_match = re.search(r'(\d+)', page_name)
            if page_match:
                page_num = page_match.group(1)
                if (page_name.isdigit() or re.search(r'\b(page|pg|p)\b', page_name.lower())):
                    return f"{comic_name}_page_{page_num}.json"
                else:
                    return f"{comic_name}_{page_name}.json"
            else:
                page_lower = page_name.lower()
                if any(keyword in page_lower for keyword in ['cover', 'title', 'credits', 'back']):
                    return f"{comic_name}_{page_name}.json"
                else:
                    comic_match = re.search(r'(\d+)', comic_name)
                    if comic_match:
                        issue_num = comic_match.group(1)
                        return f"{comic_name}_page_{page_name}.json"
                    else:
                        return f"{comic_name}_{page_name}.json"
        elif len(path_parts) == 1:
            page_name = re.sub(r'[<>:"/\\|?*]', '_', os.path.splitext(path_parts[0])[0])
            return f"{page_name}.json"
        else:
            return f"image_{img_id}.json"
    
    out_name = create_unique_filename(path)
    output_path = os.path.join(output_dir, out_name)
    if skip_existing and os.path.exists(output_path):
        return None  # Skip this image

    # Collect boxes per category
    panels, panel_scores = [], []
    texts, text_scores = [], []
    balloons, balloon_scores = [], []
    chars, char_scores = [], []
    faces, face_scores = [], []
    onos, ono_scores = [], []

    for a in anns:
        cid = a['category_id']
        b = [float(x) for x in a['bbox']]
        s = float(a.get('score', 1.0))
        name = cat_map.get(cid, str(cid))
        if name == 'panel' and s >= panel_thr:
            panels.append(b); panel_scores.append(s)
        elif name == 'balloon' and s >= balloon_thr:
            balloons.append(b); balloon_scores.append(s)
        elif name == 'text' and s >= text_thr:
            texts.append(b); text_scores.append(s)
        elif name == 'onomatopoeia' and s >= text_thr:
            onos.append(b); ono_scores.append(s)
        elif name == 'character' and s >= char_thr:
            chars.append(b); char_scores.append(s)
        elif name == 'face' and s >= char_thr:
            faces.append(b); face_scores.append(s)

    if not panels:
        panels = [a['bbox'] for a in sorted(anns, key=lambda x: -x.get('score', 0)) if cat_map.get(a['category_id'])=='panel'][:6]
        panel_scores = [1.0]*len(panels)

    if len(panels) > 1:
        keep = nms(panels, panel_scores, iou_thr=panel_nms)
        panels = [panels[i] for i in keep]
        panel_scores = [panel_scores[i] for i in keep]

    clean_panels = []
    for b in panels:
        x,y,w,h = b
        if w < 32 or h < 32: continue
        if w*h < 0.01 * (W*H):  # drop super tiny fragments
            continue
        clean_panels.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
    panels = clean_panels
    if not panels:
        return None

    def assign_to_panels(boxes: List[List[float]]):
        assign = [[] for _ in panels]
        for b in boxes:
            cx, cy = box_center(b)
            for i, pb in enumerate(panels):
                if pb[0] <= cx <= pb[0]+pb[2] and pb[1] <= cy <= pb[1]+pb[3]:
                    assign[i].append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
                    break
        return assign

    texts_by_panel = assign_to_panels(texts)
    balloons_by_panel = assign_to_panels(balloons)
    chars_by_panel = assign_to_panels(chars)
    faces_by_panel = assign_to_panels(faces)
    onos_by_panel = assign_to_panels(onos)

    order = compute_reading_order(panels, rtl=rtl)
    adj_mask, neighbor_dict, next_idx = build_adjacency(panels, order)

    # Optional: fuse VLM text per panel if provided
    vlm_text_buckets = None
    if vlm_pages_dir:
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = re.sub(r'\.(png|jpg|jpeg)$', '', stem, flags=re.IGNORECASE)
        candidates = []
        if os.path.exists(vlm_pages_dir):
            for f in os.listdir(vlm_pages_dir):
                if f.endswith('.json') and stem in f:
                    candidates.append(f)
        if candidates:
            exact_matches = [c for c in candidates if c.startswith(stem + '.') or c == stem + '.json']
            vlm_file = exact_matches[0] if exact_matches else candidates[0]
            try:
                with open(os.path.join(vlm_pages_dir, vlm_file), 'r', encoding='utf-8') as vf:
                    vlm = json.load(vf)
                    vlm_panels = vlm.get('panels', [])
                    vlm_text_buckets = []
                    for p in vlm_panels:
                        dialog = [s['dialogue'] for s in p.get('speakers', []) if 'dialogue' in s and s['dialogue']]
                        narration = []
                        cap = p.get('caption')
                        if cap: narration.append(cap)
                        vlm_text_buckets.append({'dialogue': dialog, 'narration': narration, 'sfx': []})
                    if len(vlm_text_buckets) < len(panels):
                        pad = len(panels) - len(vlm_text_buckets)
                        vlm_text_buckets += [{'dialogue': [], 'narration': [], 'sfx': []}]*pad
                    elif len(vlm_text_buckets) > len(panels):
                        vlm_text_buckets = vlm_text_buckets[:len(panels)]
            except Exception:
                vlm_text_buckets = None

    # OCR: run and bucket tokens
    ocr_tokens_by_panel = [[] for _ in panels]
    ocr_text_buckets_by_panel = [None for _ in panels]
    if enable_ocr:
        # configure OCR (one per process)
        global _OCR_CFG
        _OCR_CFG = {'use_gpu': ocr_gpu, 'lang': ocr_lang}
        try:
            if ocr_scope == 'panel':
                # Run OCR per panel crop for better precision
                with Image.open(path).convert('RGB') as im:
                    for i, pb in enumerate(panels):
                        x,y,w,h = pb
                        crop = im.crop((x, y, x+w, y+h))
                        arr = np.array(crop)
                        toks = _run_ocr_on_image(arr)
                        # Convert token polys/bboxes back to page coords by offsetting
                        for t in toks:
                            t['poly'] = [[p[0]+x, p[1]+y] for p in t['poly']]
                            t['bbox'] = [t['bbox'][0]+x, t['bbox'][1]+y, t['bbox'][2], t['bbox'][3]]
                        ann_tokens, buckets = _assign_tokens_ocr_to_buckets(
                            toks, pb, balloons_by_panel[i], onos_by_panel[i],
                            angle_thr=ocr_angle_thr, sfx_size_frac=ocr_sfx_size_frac, min_conf=ocr_min_conf
                        )
                        ocr_tokens_by_panel[i] = ann_tokens
                        ocr_text_buckets_by_panel[i] = buckets
            else:
                # Run OCR once on the full page, then assign to panels
                toks_page = _run_ocr_on_image(path)
                # Partition tokens to panel by center
                for t in toks_page:
                    cx, cy = _center_of_poly(t['poly'])
                    for i, pb in enumerate(panels):
                        if _point_in_box(cx, cy, pb):
                            ann_tokens, buckets = _assign_tokens_ocr_to_buckets(
                                [t], pb, balloons_by_panel[i], onos_by_panel[i],
                                angle_thr=ocr_angle_thr, sfx_size_frac=ocr_sfx_size_frac, min_conf=ocr_min_conf
                            )
                            ocr_tokens_by_panel[i].extend(ann_tokens)
                            # Merge buckets
                            if ocr_text_buckets_by_panel[i] is None:
                                ocr_text_buckets_by_panel[i] = {'dialogue': [], 'narration': [], 'sfx': []}
                            for k in ('dialogue','narration','sfx'):
                                ocr_text_buckets_by_panel[i][k].extend(buckets[k])
        except Exception as e:
            # Fail-soft: keep OCR off for this page
            ocr_tokens_by_panel = [[] for _ in panels]
            ocr_text_buckets_by_panel = [None for _ in panels]

    # Build page JSON
    page = {
        'page_id': page_id_from_path(path),
        'page_image_path': norm_path(path),
        'page_size': {'width': int(W), 'height': int(H)},
        'panels': []
    }

    # Reindex panels to reading order for clean alignment
    for ord_idx, pi in enumerate(order):
        b = panels[pi]
        # Pick text buckets: OCR first if present, else VLM, else empty
        if ocr_text_buckets_by_panel[pi]:
            text_bucket = ocr_text_buckets_by_panel[pi]
        elif vlm_text_buckets is not None:
            text_bucket = vlm_text_buckets[ord_idx]
        else:
            text_bucket = {'dialogue': [], 'narration': [], 'sfx': []}

        # Build record
        panel_rec = {
            'panel_id': f'p{ord_idx}',
            'panel_coords': [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
            'order_index': ord_idx,
            'neighbors': {
                'left': [f'p{j}' for j in neighbor_dict[pi]['left']],
                'right': [f'p{j}' for j in neighbor_dict[pi]['right']],
                'above': [f'p{j}' for j in neighbor_dict[pi]['above']],
                'below': [f'p{j}' for j in neighbor_dict[pi]['below']],
            },
            'text': text_bucket,
            'ocr_tokens': ocr_tokens_by_panel[pi] if enable_ocr else [],
            'character_coords': chars_by_panel[pi],
            'face_coords': faces_by_panel[pi],
            'balloon_coords': balloons_by_panel[pi],
            'raw_text_regions': texts_by_panel[pi],
            'onomatopoeia_coords': onos_by_panel[pi],
        }
        page['panels'].append(panel_rec)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
        return {'page': page, 'filename': out_name, 'status': 'written'}
    except Exception as e:
        return {'page': page, 'filename': out_name, 'status': 'error', 'error': str(e)}

# --------- Core converter with multiprocessing
def build_pages_from_coco_multi(coco_path: str, vlm_pages_dir: str=None,
                               panel_thr=0.75, panel_nms=0.25,
                               text_thr=0.5, char_thr=0.6, balloon_thr=0.5,
                               rtl=False, limit=None, num_workers=None,
                               output_dir: str=None, skip_existing: bool=False,
                               # OCR flags
                               ocr=False, ocr_gpu=False, ocr_lang='en', ocr_min_conf=0.5,
                               ocr_scope='panel', ocr_angle_thr=15.0, ocr_sfx_size_frac=0.08) -> List[Dict]:
    
    print("Loading COCO data...")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    cat_map = {}
    for c in coco.get('categories', []):
        cat_map[c['id']] = c['name']
    
    anns_by_img = defaultdict(list)
    for a in coco.get('annotations', []):
        anns_by_img[a['image_id']].append(a)

    imginfo = {}
    if isinstance(coco.get('images'), list) and coco['images'] and isinstance(coco['images'][0], dict):
        for im in coco['images']:
            iid = im['id']
            p = im.get('file_name', iid)
            imginfo[iid] = {'path': norm_path(p), 'width': im.get('width'), 'height': im.get('height')}

    work_items = []
    for img_id, anns in anns_by_img.items():
        work_items.append((
            img_id, anns, cat_map, imginfo, vlm_pages_dir, 
            panel_thr, panel_nms, text_thr, char_thr, balloon_thr, rtl, output_dir, skip_existing,
            # OCR
            ocr, ocr_gpu, ocr_lang, ocr_min_conf, ocr_scope, ocr_angle_thr, ocr_sfx_size_frac
        ))
    
    if limit is not None:
        work_items = work_items[:limit]
        print(f"DEBUG: Applied limit of {limit} images")
    
    print(f"Processing {len(work_items)} images (out of {len(anns_by_img)} total in dataset)")
    print(f"Using {num_workers or mp.cpu_count()} workers")
    if skip_existing:
        print("Skip-existing mode: Will skip files that already exist")
    if ocr:
        print(f"OCR enabled (scope={ocr_scope}, lang={ocr_lang}, gpu={ocr_gpu}, min_conf={ocr_min_conf})")

    start_time = time.time()
    results = []
    
    if num_workers == 1:
        for item in tqdm(work_items, desc="Processing images"):
            result = process_single_image(item)
            if result is not None:
                results.append(result)
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, work_items),
                total=len(work_items),
                desc="Processing images"
            ))
            results = [r for r in results if r is not None]
    
    written_count = sum(1 for r in results if r['status'] == 'written')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = len(work_items) - len(results) if skip_existing else 0
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted processing in {elapsed_time:.1f} seconds")
    print(f"Files written: {written_count}")
    print(f"Files with errors: {error_count}")
    if skip_existing:
        print(f"Files skipped (already existed): {skipped_count}")
    print(f"Average time per image: {elapsed_time/max(1,len(work_items)):.2f} seconds")
    
    pages = [r['page'] for r in results if r['status'] == 'written']
    return pages

if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True)
    ap.add_argument('--vlm_dir', default=None, help='dir of per-page VLM JSONs (optional)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='limit processing to first N images (for testing)')
    ap.add_argument('--workers', type=int, default=None, help='number of worker processes (default: CPU count)')
    ap.add_argument('--skip-existing', action='store_true', help='skip processing images that already have output files')
    # OCR options
    ap.add_argument('--ocr', action='store_true', help='enable PaddleOCR to populate text buckets')
    ap.add_argument('--ocr-gpu', action='store_true', help='use GPU for PaddleOCR (requires paddlepaddle-gpu)')
    ap.add_argument('--ocr-lang', default='en', help='OCR language code (e.g., en, ch, fr, de, ja)')
    ap.add_argument('--ocr-min-conf', type=float, default=0.5, help='minimum OCR confidence to keep a token')
    ap.add_argument('--ocr-scope', choices=['panel','page'], default='panel', help='run OCR per panel crop or on full page')
    ap.add_argument('--ocr-angle-thr', type=float, default=15.0, help='angle threshold (deg) to treat slanted text as SFX outside balloons')
    ap.add_argument('--ocr-sfx-size-frac', type=float, default=0.08, help='fraction of panel height above which outside-balloon text is SFX')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    if args.limit:
        print(f"TEST MODE: Processing only {args.limit} images")
    else:
        print("FULL PROCESSING MODE: Processing all images")
    
    pages = build_pages_from_coco_multi(
        args.coco, 
        vlm_pages_dir=args.vlm_dir, 
        limit=args.limit,
        num_workers=args.workers,
        output_dir=args.out,
        skip_existing=args.skip_existing,
        # OCR flags
        ocr=args.ocr,
        ocr_gpu=args.ocr_gpu,
        ocr_lang=args.ocr_lang,
        ocr_min_conf=args.ocr_min_conf,
        ocr_scope=args.ocr_scope,
        ocr_angle_thr=args.ocr_angle_thr,
        ocr_sfx_size_frac=args.ocr_sfx_size_fracc if hasattr(args, 'ocr_sfx_size_fracc') else args.ocr_sfx_size_frac
    )
    
    print(f'Processing complete! Files written incrementally to {args.out}')
```

Notes and tips
- panel vs page OCR:
  - panel OCR (default) is slower but more accurate in assigning tokens to the right panel/balloon.
  - page OCR is faster; use if throughput is critical and panel layouts are clean.
- SFX heuristics:
  - angle_thr=15 degrees and sfx_size_frac=0.08 are conservative; tweak per style (manga SFX may be more slanted).
- VLM fallback:
  - If OCR yields no tokens for a panel, we’ll use your VLM dialogue/caption for that panel’s text buckets.

Want me to add a quick visual QA script that draws panels/balloons and overlays OCR tokens colored by bucket (dialogue/narration/SFX)? It’s super useful to tune the thresholds on a few books.