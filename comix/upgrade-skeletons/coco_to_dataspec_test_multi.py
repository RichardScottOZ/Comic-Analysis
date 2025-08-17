import json, os, re
from collections import defaultdict
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

# --------- Utils (same as original)
def norm_path(p: str) -> str:
    # Normalize Windows paths for consistency
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
    inter_h = max(0, min(ay2, by2) - max(ay, by))
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
    # Z-path with row grouping by vertical overlap
    idxs = list(range(len(panel_boxes)))
    idxs.sort(key=lambda i: panel_boxes[i][1])  # sort by top
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
    # next in order
    for pos,i in enumerate(order):
        if pos < len(order)-1:
            j = order[pos+1]
            adj[i,j] = 1
    # directional neighbors
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
            if j is not None:
                adj[i,j] = 1
    # kNN fallback
    C = np.array(centers)
    for i in range(N):
        if adj[i].sum() == 0 and N > 1:
            d = np.sqrt(((C - C[i])**2).sum(axis=1))
            nn = np.argsort(d)[1:min(k_fallback+1, N)]
            for j in nn:
                adj[i,int(j)] = 1
    neighbor_dict = {}
    for i in range(N):
        neighbor_dict[i] = {
            'left': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][0] < panel_boxes[i][0]],
            'right': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][0] > panel_boxes[i][0]],
            'above': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][1] < panel_boxes[i][1]],
            'below': [int(j) for j in np.where(adj[i]==1)[0] if panel_boxes[j][1] > panel_boxes[i][1]],
        }
    # next_idx list (for training RPP-lite)
    next_idx = [-100]*N
    for pos,i in enumerate(order):
        if pos < len(order)-1:
            next_idx[i] = order[pos+1]
    return adj, neighbor_dict, next_idx

# --------- Single image processor (for multiprocessing)
def process_single_image(args):
    """Process a single image - designed for multiprocessing"""
    img_id, anns, cat_map, imginfo, vlm_pages_dir, panel_thr, panel_nms, text_thr, char_thr, balloon_thr, rtl, output_dir, skip_existing = args
    
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
            # Sometimes you have "xxx.jpg.png" in image_id; try stripping a redundant extension
            alt = re.sub(r'\.jpg\.png$', '.jpg', path, flags=re.IGNORECASE)
            alt = re.sub(r'\.png\.jpg$', '.png', alt, flags=re.IGNORECASE)
            with Image.open(alt) as im:
                W, H = im.size
            path = alt

    # Check if output file already exists (for skip-existing)
    out_name = os.path.basename(path)
    out_name = os.path.splitext(out_name)[0] + '.json'
    output_path = os.path.join(output_dir, out_name)
    
    if skip_existing and os.path.exists(output_path):
        return None  # Skip this image

    # Collect boxes per category
    panels, panel_scores = [], []
    texts, text_scores = [], []
    balloons, balloon_scores = [], []
    chars, char_scores = [], []
    faces, face_scores = [], []

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
            texts.append(b); text_scores.append(s)  # treat as text region for now
        elif name == 'character' and s >= char_thr:
            chars.append(b); char_scores.append(s)
        elif name == 'face' and s >= char_thr:
            faces.append(b); face_scores.append(s)

    if not panels:
        # fallback: pick top 1-6 from looser threshold
        panels = [a['bbox'] for a in sorted(anns, key=lambda x: -x.get('score', 0)) if cat_map.get(a['category_id'])=='panel'][:6]
        panel_scores = [1.0]*len(panels)

    # NMS on panels
    if len(panels) > 1:
        keep = nms(panels, panel_scores, iou_thr=panel_nms)
        panels = [panels[i] for i in keep]
        panel_scores = [panel_scores[i] for i in keep]

    # Clean tiny/edge panels
    clean_panels = []
    for b in panels:
        x,y,w,h = b
        if w < 32 or h < 32: continue
        if w*h < 0.01 * (W*H):  # drop super tiny fragments
            continue
        clean_panels.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
    panels = clean_panels

    if not panels:
        # Skip page if no reliable panel found
        return None

    # Assign detections to panels by center point-in-bbox
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

    # Reading order & neighbors
    order = compute_reading_order(panels, rtl=rtl)
    adj_mask, neighbor_dict, next_idx = build_adjacency(panels, order)

    # Optional: fuse VLM text per panel if provided (assume VLM numbers panels in reading order)
    vlm_text_buckets = None
    if vlm_pages_dir:
        # Find matching VLM json by file stem - handle various naming patterns
        stem = os.path.splitext(os.path.basename(path))[0]
        # Clean up common naming variations
        stem = re.sub(r'\.png$', '', stem, flags=re.IGNORECASE)
        stem = re.sub(r'\.jpg$', '', stem, flags=re.IGNORECASE)
        stem = re.sub(r'\.jpeg$', '', stem, flags=re.IGNORECASE)
        
        # Look for candidates with more flexible matching
        candidates = []
        if os.path.exists(vlm_pages_dir):
            for f in os.listdir(vlm_pages_dir):
                if f.endswith('.json'):
                    # Try exact stem match first
                    if f.startswith(stem) and f.endswith('.json'):
                        candidates.append(f)
                    # Try without any extensions
                    clean_f = re.sub(r'\.(png|jpg|jpeg)$', '', f, flags=re.IGNORECASE)
                    if clean_f == stem:
                        candidates.append(f)
                    # Try partial matching (in case of additional suffixes)
                    if stem in f and f.endswith('.json'):
                        candidates.append(f)
        
        if candidates:
            # Prefer exact matches first
            exact_matches = [c for c in candidates if c.startswith(stem + '.') or c == stem + '.json']
            if exact_matches:
                vlm_file = exact_matches[0]
            else:
                vlm_file = candidates[0]  # take first candidate
            
            try:
                with open(os.path.join(vlm_pages_dir, vlm_file), 'r', encoding='utf-8') as vf:
                    vlm = json.load(vf)
                    vlm_panels = vlm.get('panels', [])
                    
                    # Build dialogue/narration/sfx lists by panel order
                    vlm_text_buckets = []
                    for p in vlm_panels:
                        # gather text fields
                        dialog = []
                        for s in p.get('speakers', []):
                            if 'dialogue' in s and s['dialogue']:
                                dialog.append(s['dialogue'])
                        # narration heuristic: look for caption or description that likely paraphrases – optional
                        narration = []
                        cap = p.get('caption')
                        if cap: narration.append(cap)
                        # no sfx in VLM – leave empty
                        vlm_text_buckets.append({
                            'dialogue': dialog,
                            'narration': narration,
                            'sfx': []
                        })
                    # If counts mismatch, pad/trim to number of detected panels
                    if len(vlm_text_buckets) < len(panels):
                        pad = len(panels) - len(vlm_text_buckets)
                        vlm_text_buckets += [{'dialogue': [], 'narration': [], 'sfx': []}]*pad
                    elif len(vlm_text_buckets) > len(panels):
                        vlm_text_buckets = vlm_text_buckets[:len(panels)]
            except Exception:
                vlm_text_buckets = None

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
        # Aggregate text buckets: VLM first (fallback); later we'll replace with OCR-to-balloons logic
        if vlm_text_buckets is not None:
            text_bucket = vlm_text_buckets[ord_idx]
        else:
            # Without OCR, we can at least mark presence; keep empty strings to avoid hallucinating
            text_bucket = {'dialogue': [], 'narration': [], 'sfx': []}

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
            'ocr_tokens': [],  # to be filled after OCR
            'character_coords': chars_by_panel[pi],
            'face_coords': faces_by_panel[pi],
            'balloon_coords': balloons_by_panel[pi],  # useful for future OCR bucketing
            'raw_text_regions': texts_by_panel[pi],   # geometry for OCR
        }
        page['panels'].append(panel_rec)

    # Write the file immediately
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
                               output_dir: str=None, skip_existing: bool=False) -> List[Dict]:
    
    print("Loading COCO data...")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Categories
    cat_map = {}
    for c in coco.get('categories', []):
        cat_map[c['id']] = c['name']  # e.g., 1->'panel'
    
    # Group annotations per image_id
    anns_by_img = defaultdict(list)
    for a in coco.get('annotations', []):
        anns_by_img[a['image_id']].append(a)

    # Build image info map if present
    imginfo = {}
    if isinstance(coco.get('images'), list) and coco['images'] and isinstance(coco['images'][0], dict):
        for im in coco['images']:
            iid = im['id']
            p = im.get('file_name', iid)
            imginfo[iid] = {
                'path': norm_path(p),
                'width': im.get('width'),
                'height': im.get('height')
            }

    # Prepare work items
    work_items = []
    for img_id, anns in anns_by_img.items():
        work_items.append((img_id, anns, cat_map, imginfo, vlm_pages_dir, 
                          panel_thr, panel_nms, text_thr, char_thr, balloon_thr, rtl, output_dir, skip_existing))
    
    # Apply limit
    if limit is not None:
        work_items = work_items[:limit]
        print(f"DEBUG: Applied limit of {limit} images")
    
    print(f"Processing {len(work_items)} images (out of {len(anns_by_img)} total in dataset)")
    print(f"Using {num_workers or mp.cpu_count()} workers")
    if skip_existing:
        print("Skip-existing mode: Will skip files that already exist")
    
    # Process with multiprocessing
    start_time = time.time()
    results = []
    
    if num_workers == 1:
        # Single-threaded for debugging
        for item in tqdm(work_items, desc="Processing images"):
            result = process_single_image(item)
            if result is not None:
                results.append(result)
    else:
        # Multiprocessing
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, work_items),
                total=len(work_items),
                desc="Processing images"
            ))
            results = [r for r in results if r is not None]
    
    # Count results by status
    written_count = sum(1 for r in results if r['status'] == 'written')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = len(work_items) - len(results) if skip_existing else 0
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted processing in {elapsed_time:.1f} seconds")
    print(f"Files written: {written_count}")
    print(f"Files with errors: {error_count}")
    if skip_existing:
        print(f"Files skipped (already existed): {skipped_count}")
    print(f"Average time per image: {elapsed_time/len(work_items):.2f} seconds")
    
    # Return just the page data for compatibility
    pages = [r['page'] for r in results if r['status'] == 'written']
    return pages

if __name__ == '__main__':
    # Example usage:
    # python coco_to_dataspect_test_multi.py --coco detections.json --vlm_dir vlm_pages/ --out out_json/ --workers 8
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True)
    ap.add_argument('--vlm_dir', default=None, help='dir of per-page VLM JSONs (optional)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='limit processing to first N images (for testing)')
    ap.add_argument('--workers', type=int, default=None, help='number of worker processes (default: CPU count)')
    ap.add_argument('--skip-existing', action='store_true', help='skip processing images that already have output files')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Estimate processing time
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
        skip_existing=args.skip_existing
    )
    
    print(f'Processing complete! Files written incrementally to {args.out}') 