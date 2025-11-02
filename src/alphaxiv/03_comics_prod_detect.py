# comix_prod_detect.py
import os, sys, cv2, json, math, argparse, tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops

# ----- Class prompts from CoMix supplement -----
PROMPTS = {
    "panel": ["comics panels","manga panels","frames","windows"],
    "character": ["characters","comics characters","person","girl","woman","man","animal"],
    "text": ["text box","text","handwriting"],
    "face": ["face","character face","animal face","head","face with nose and mouth","person’s face"],
}
CAT_ID = {"panel":1, "character":2, "text":3, "face":4}
PHRASE_TO_CLASS = {p:k for k, lst in PROMPTS.items() for p in lst}
ALL_PHRASES = list(PHRASE_TO_CLASS.keys())
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) - inter + 1e-9
    return inter / ua

def nms_per_class(boxes, scores, labels, iou_thr=0.6):
    # boxes: Nx4 xyxy, scores: N, labels: N int
    keep_boxes, keep_scores, keep_labels = [], [], []
    for c in sorted(set(labels)):
        idx = [i for i,l in enumerate(labels) if l==c]
        if not idx: continue
        b = [boxes[i] for i in idx]
        s = [scores[i] for i in idx]
        order = np.argsort(s)[::-1]
        b = [b[i] for i in order]; s = [s[i] for i in order]
        kept = []
        while b:
            m = b.pop(0); ms = s.pop(0)
            kept.append((m, ms))
            b2, s2 = [], []
            for bb, ss in zip(b, s):
                if iou_xyxy(m, bb) < iou_thr:
                    b2.append(bb); s2.append(ss)
            b, s = b2, s2
        for m, ms in kept:
            keep_boxes.append(m); keep_scores.append(ms); keep_labels.append(c)
    return keep_boxes, keep_scores, keep_labels

def list_books(root):
    root = Path(root)
    for d in sorted(root.iterdir()):
        if d.is_dir():
            # has at least one image?
            has_img = any((d / f).suffix.lower() in IMG_EXTS for f in os.listdir(d))
            if has_img:
                yield d

def list_images(book_dir):
    for fn in sorted(os.listdir(book_dir)):
        if Path(fn).suffix.lower() in IMG_EXTS:
            yield book_dir / fn

def load_model_once(cfg_path, ckpt_path, device):
    model = load_model(cfg_path, ckpt_path)
    if device.startswith("cuda"):
        model = model.cuda()
    return model

def run_gdino_on_path(model, path, box_thr, text_thr):
    image_source, image = load_image(str(path))
    H, W = image_source.shape[:2]
    boxes, logits, phrases = predict(model, image, caption=", ".join(ALL_PHRASES),
                                     box_threshold=box_thr, text_threshold=text_thr)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    boxes_xyxy[:,[0,2]] *= W
    boxes_xyxy[:,[1,3]] *= H
    classes = [PHRASE_TO_CLASS.get(ph, None) for ph in phrases]
    scores = logits.sigmoid().tolist()
    # Filter unknown classes
    out = [(b.tolist(), s, c) for b,s,c in zip(boxes_xyxy, scores, classes) if c in CAT_ID]
    return out, (H,W)

def tile_coords(W, H, tile_size, overlap):
    tw = th = tile_size
    sx = max(1, int(math.ceil((W - overlap*tw) / (tw*(1-overlap)))))
    sy = max(1, int(math.ceil((H - overlap*th) / (th*(1-overlap)))))
    xs = [int(i*tw*(1-overlap)) for i in range(sx)]
    ys = [int(j*th*(1-overlap)) for j in range(sy)]
    if not xs or xs[-1] + tw < W: xs.append(max(0, W - tw))
    if not ys or ys[-1] + th < H: ys.append(max(0, H - th))
    for y in ys:
        for x in xs:
            yield x, y, min(W, x+tw), min(H, y+th)

def run_gdino_tiled(model, path, box_thr, text_thr, tile_size=1024, overlap=0.2):
    im = cv2.imread(str(path))
    if im is None:
        return [], (0,0)
    H, W = im.shape[:2]
    boxes_all, scores_all, labels_all = [], [], []
    # Write tiles to temp files (keeps the GDINO loader simple and robust)
    for x1,y1,x2,y2 in tile_coords(W,H,tile_size,overlap):
        crop = im[y1:y2, x1:x2]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
            cv2.imwrite(tf.name, crop)
            out, _ = run_gdino_on_path(model, tf.name, box_thr, text_thr)
        for b, s, c in out:
            # shift box back to full image coords
            bx = [b[0]+x1, b[1]+y1, b[2]+x1, b[3]+y1]
            boxes_all.append(bx); scores_all.append(s); labels_all.append(c)
    # class-aware NMS
    boxes_n, scores_n, labels_n = nms_per_class(boxes_all, scores_all, labels_all, iou_thr=0.6)
    return list(zip(boxes_n, scores_n, labels_n)), (H,W)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder with books (each a folder of pages)")
    ap.add_argument("--out",  required=True, help="Output folder")
    ap.add_argument("--cfg",  default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--ckpt", required=True, help="Path to GroundingDINO checkpoint .pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--tile", action="store_true", help="Enable simple tiling pass for tiny objects")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--viz", action="store_true", help="Save per-image visualizations")
    ap.add_argument("--max-per-image", type=int, default=500, help="Optional cap for detections per image")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model_once(args.cfg, args.ckpt, device)
    out_root = Path(args.out); ensure_dir(out_root)

    categories = [{"id":CAT_ID[k], "name":k} for k in ["panel","character","text","face"]]

    # Combined COCO (all books)
    combined = {"images": [], "annotations": [], "categories": categories.copy()}
    glob_ann_id = 1
    glob_img_id = 1

    for book_dir in tqdm(list_books(args.root), desc="Books", unit="book"):
        book_name = book_dir.name
        book_out_dir = out_root / book_name
        viz_dir = book_out_dir / "viz"
        ensure_dir(book_out_dir)
        if args.viz:
            ensure_dir(viz_dir)

        # Per-book COCO
        coco = {"images": [], "annotations": [], "categories": categories.copy()}
        ann_id = 1
        img_id = 1

        for img_path in tqdm(list_images(book_dir), desc=f"{book_name}", leave=False):
            try:
                # Run detector
                if args.tile:
                    out, (H,W) = run_gdino_tiled(model, img_path, args.box_thr, args.text_thr,
                                                 tile_size=args.tile_size, overlap=args.tile_overlap)
                else:
                    out, (H,W) = run_gdino_on_path(model, img_path, args.box_thr, args.text_thr)

                # Optionally cap extremely dense images
                if args.max_per_image and len(out) > args.max_per_image:
                    out = sorted(out, key=lambda x: x[1], reverse=True)[:args.max_per_image]

                # Relative file_name "book/page.ext" in combined, just "page.ext" in per-book
                rel_name_book = img_path.name
                rel_name_combined = f"{book_name}/{img_path.name}"

                # Record image entries
                coco["images"].append({"id": img_id, "file_name": rel_name_book, "width": W, "height": H})
                combined["images"].append({"id": glob_img_id, "file_name": rel_name_combined, "width": W, "height": H})

                # Draw visualization if requested
                if args.viz:
                    im = cv2.imread(str(img_path))
                    for (x1,y1,x2,y2), score, cls in out:
                        color = {"panel":(0,200,255),"character":(0,255,0),"text":(255,150,0),"face":(255,0,200)}[cls]
                        cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                        label = f"{cls}:{score:.2f}"
                        cv2.putText(im, label, (int(x1), max(0,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.imwrite(str(viz_dir / rel_name_book), im)

                # Annotations
                for (x1,y1,x2,y2), score, cls in out:
                    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
                    a = w*h
                    ann = {
                        "id": ann_id, "image_id": img_id,
                        "category_id": CAT_ID[cls],
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(a), "iscrowd": 0, "segmentation": [],
                        "score": float(score)  # handy for manual triage; COCO eval ignores it
                    }
                    coco["annotations"].append(ann)
                    ann2 = ann.copy()
                    ann2["id"] = glob_ann_id
                    ann2["image_id"] = glob_img_id
                    combined["annotations"].append(ann2)
                    ann_id += 1; glob_ann_id += 1

                img_id += 1; glob_img_id += 1

            except Exception as e:
                print(f"[WARN] Failed on {img_path}: {e}", file=sys.stderr)
                continue

        # Write per-book JSON
        with open(book_out_dir / "preds_coco.json", "w") as f:
            json.dump(coco, f)
        # Optionally also save a flat predictions list if your downstream prefers it
        with open(book_out_dir / "preds_list.json", "w") as f:
            flat = []
            for a in coco["annotations"]:
                flat.append({
                    "file_name": coco["images"][a["image_id"]-1]["file_name"],
                    "category_id": a["category_id"], "bbox": a["bbox"], "score": a.get("score", 1.0)
                })
            json.dump(flat, f)

    # Write combined JSON
    ensure_dir(out_root)
    with open(out_root / "combined_preds_coco.json", "w") as f:
        json.dump(combined, f)
    print("Done. Combined COCO at:", out_root / "combined_preds_coco.json")

if __name__ == "__main__":
    main()
