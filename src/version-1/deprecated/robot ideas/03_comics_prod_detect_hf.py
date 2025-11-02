# comix_prod_detect_hf.py


#Usage example:

#python comix_prod_detect_hf.py --root D:\books_root --out D:\out --model-id IDEA-Research/grounding-dino-swint-ogc --viz
#To enable tiling for tiny text/face: add --tile --tile-size 1024 --tile-overlap 0.2
#Dependencies:

#pip install transformers huggingface-hub torch torchvision opencv-python pycocotools tqdm

import os, json, math, argparse, tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

# ----- Prompts from CoMix supplement -----
PROMPTS = {
    "panel": ["comics panels","manga panels","frames","windows"],
    "character": ["characters","comics characters","person","girl","woman","man","animal"],
    "text": ["text box","text","handwriting"],
    "face": ["face","character face","animal face","head","face with nose and mouth","person’s face"],
}
PHRASE_TO_CLASS = {p:k for k, lst in PROMPTS.items() for p in lst}
CAT_ID = {"panel":1, "character":2, "text":3, "face":4}
CATEGORIES = [{"id":CAT_ID[k], "name":k} for k in ["panel","character","text","face"]]
CAPTION = ", ".join(PHRASE_TO_CLASS.keys())
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def list_books(root):
    root = Path(root)
    for d in sorted(root.iterdir()):
        if d.is_dir():
            # has at least one image?
            try:
                if any((d/f).suffix.lower() in IMG_EXTS for f in os.listdir(d)):
                    yield d
            except PermissionError:
                continue

def list_books(root):
    root = Path(root)
    # if root itself contains images, treat it as one book
    has_img = any(p.suffix.lower() in IMG_EXTS for p in root.iterdir() if p.is_file())
    if has_img:
        yield root
        return
    # otherwise, iterate subfolders
    for d in sorted(root.iterdir()):
        if d.is_dir():
            if any((d/f).suffix.lower() in IMG_EXTS for f in os.listdir(d)):
                yield d


def list_images(book_dir):
    for fn in sorted(os.listdir(book_dir)):
        if Path(fn).suffix.lower() in IMG_EXTS:
            yield book_dir / fn

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1]) - inter + 1e-9
    return inter / ua

def nms_per_class(boxes, scores, labels, iou_thr=0.6):
    keep_boxes, keep_scores, keep_labels = [], [], []
    uniq = sorted(set(labels))
    for c in uniq:
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

class HF_GroundingDINO:
    def __init__(self, model_id, device="cuda:0"):
        self.processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def infer(self, image_bgr, box_threshold=0.30, text_threshold=0.25):
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled="cuda" in self.device):
            inputs = self.processor(images=image_rgb, text=CAPTION, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs, input_ids=inputs["input_ids"],
                target_sizes=[(H, W)],
                box_threshold=box_threshold, text_threshold=text_threshold
            )[0]
        boxes = processed["boxes"].tolist()     # xyxy
        scores = processed["scores"].tolist()
        phrases = processed["phrases"]
        dets = []
        for (x1,y1,x2,y2), s, ph in zip(boxes, scores, phrases):
            cls = PHRASE_TO_CLASS.get(ph, None)
            if cls is None: 
                continue
            dets.append(((x1,y1,x2,y2), float(s), cls))
        return dets, (H, W)


    # mapping helpers once at module level
    PHRASE_TO_CLASS = {p:k for k, lst in PROMPTS.items() for p in lst}
    LOWER_MAP = {k.lower(): v for k, v in PHRASE_TO_CLASS.items()}
    CAPTION = ", ".join(PHRASE_TO_CLASS.keys())

    @torch.no_grad()
    def infer(self, image_bgr, box_threshold=0.30, text_threshold=0.25):
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled="cuda" in self.device):
            inputs = self.processor(images=image_rgb, text=CAPTION, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            # Use 'threshold' (new) and fall back to 'box_threshold' (old) if needed
            try:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs=outputs, input_ids=inputs["input_ids"],
                    target_sizes=[(H, W)],
                    threshold=box_threshold,      # new arg name
                    text_threshold=text_threshold
                )[0]
            except TypeError:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs=outputs, input_ids=inputs["input_ids"],
                    target_sizes=[(H, W)],
                    box_threshold=box_threshold,  # older transformers
                    text_threshold=text_threshold
                )[0]

        boxes = processed["boxes"].tolist()     # xyxy
        scores = processed["scores"].tolist()

        # HF returns 'text_labels' (strings) and 'labels' (indices). Prefer text_labels.
        phrases = processed.get("text_labels", None)
        if phrases is None:
            # Fallback: map indices to strings if needed (rare; most builds include text_labels)
            idxs = processed.get("labels", [])
            # As a simple fallback, treat idxs as best-effort and skip if we can't map reliably:
            phrases = [None for _ in idxs]

        dets = []
        for (x1,y1,x2,y2), s, ph in zip(boxes, scores, phrases):
            if ph is None:
                continue
            key = ph if ph in PHRASE_TO_CLASS else ph.lower()
            cls = PHRASE_TO_CLASS.get(key, LOWER_MAP.get(key))
            if cls is None:
                continue
            dets.append(((x1,y1,x2,y2), float(s), cls))
        return dets, (H, W)
        

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder with books (each a folder of pages)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base",
                    help="HF model id, e.g. IDEA-Research/grounding-dino-base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--tile", action="store_true", help="Enable simple tiling pass")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--max-per-image", type=int, default=500)
    args = ap.parse_args()

    detector = HF_GroundingDINO(args.model_id, device=args.device)

    out_root = Path(args.out); ensure_dir(out_root)
    combined = {"images": [], "annotations": [], "categories": CATEGORIES.copy()}
    glob_ann_id = 1
    glob_img_id = 1

    for book_dir in tqdm(list_books(args.root), desc="Books", unit="book"):
        book_name = book_dir.name
        book_out = out_root / book_name
        viz_dir = book_out / "viz"
        ensure_dir(book_out)
        if args.viz:
            ensure_dir(viz_dir)

        coco = {"images": [], "annotations": [], "categories": CATEGORIES.copy()}
        ann_id = 1
        img_id = 1

        for img_path in tqdm(list_images(book_dir), desc=book_name, leave=False):
            im = cv2.imread(str(img_path))
            if im is None:
                continue

            if args.tile:
                boxes_all, scores_all, labels_all = [], [], []
                H, W = im.shape[:2]
                for x1,y1,x2,y2 in tile_coords(W,H,args.tile_size,args.tile_overlap):
                    crop = im[y1:y2, x1:x2]
                    dets, _ = detector.infer(crop, args.box_thr, args.text_thr)
                    for (bx1,by1,bx2,by2), sc, cls in dets:
                        boxes_all.append([bx1+x1, by1+y1, bx2+x1, by2+y1])
                        scores_all.append(sc)
                        labels_all.append(cls)
                # class-aware NMS
                boxes_n, scores_n, labels_n = nms_per_class(boxes_all, scores_all, labels_all, iou_thr=0.6)
                dets = list(zip(boxes_n, scores_n, labels_n))
                H, W = im.shape[:2]
            else:
                dets, (H, W) = detector.infer(im, args.box_thr, args.text_thr)

            if args.max_per_image and len(dets) > args.max_per_image:
                dets = sorted(dets, key=lambda x: x[1], reverse=True)[:args.max_per_image]

            rel_book = img_path.name
            rel_comb = f"{book_name}/{img_path.name}"

            coco["images"].append({"id": img_id, "file_name": rel_book, "width": W, "height": H})
            combined["images"].append({"id": glob_img_id, "file_name": rel_comb, "width": W, "height": H})

            if args.viz:
                vis = im.copy()
                for (x1,y1,x2,y2), sc, cls in dets:
                    color = {"panel":(0,200,255),"character":(0,255,0),"text":(255,150,0),"face":(255,0,200)}[cls]
                    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                    cv2.putText(vis, f"{cls}:{sc:.2f}", (int(x1), max(0,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.imwrite(str(viz_dir / rel_book), vis)

            for (x1,y1,x2,y2), sc, cls in dets:
                w = max(0.0, x2-x1); h = max(0.0, y2-y1); a = w*h
                a1 = {
                    "id": ann_id, "image_id": img_id,
                    "category_id": CAT_ID[cls],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(a), "iscrowd": 0, "segmentation": [],
                    "score": float(sc)
                }
                a2 = a1.copy(); a2["id"] = glob_ann_id; a2["image_id"] = glob_img_id
                coco["annotations"].append(a1)
                combined["annotations"].append(a2)
                ann_id += 1; glob_ann_id += 1

            img_id += 1; glob_img_id += 1

        with open(book_out / "preds_coco.json", "w") as f:
            json.dump(coco, f)
        with open(book_out / "preds_list.json", "w") as f:
            flat = []
            for a in coco["annotations"]:
                flat.append({
                    "file_name": coco["images"][a["image_id"]-1]["file_name"],
                    "category_id": a["category_id"], "bbox": a["bbox"], "score": a.get("score", 1.0)
                })
            json.dump(flat, f)

    with open(out_root / "combined_preds_coco.json", "w") as f:
        json.dump(combined, f)
    print("Done. Combined COCO at:", out_root / "combined_preds_coco.json")

if __name__ == "__main__":
    main()
