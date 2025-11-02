# bootstrap_hf_groundingdino.py
import os, cv2, json, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

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

def list_images(root):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p

class HF_GDINO:
    def __init__(self, model_id, device="cuda:0"):
        self.proc = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def infer(self, image_bgr, box_thr=0.30, text_thr=0.25):
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled="cuda" in self.device):
            inputs = self.proc(images=image_rgb, text=CAPTION, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            out = self.proc.post_process_grounded_object_detection(
                outputs=outputs, input_ids=inputs["input_ids"],
                target_sizes=[(H, W)],
                box_threshold=box_thr, text_threshold=text_thr
            )[0]
        boxes = out["boxes"].tolist()     # xyxy
        scores = out["scores"].tolist()
        phrases = out["phrases"]
        dets = []
        for (x1,y1,x2,y2), s, ph in zip(boxes, scores, phrases):
            cls = PHRASE_TO_CLASS.get(ph, None)
            if cls is None: continue
            dets.append(((x1,y1,x2,y2), float(s), cls))
        return dets, (H, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="Directory with images for one split (e.g., images/train)")
    ap.add_argument("--out-json", required=True, help="Where to write bootstrap COCO JSON")
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    args = ap.parse_args()

    det = HF_GDINO(args.model_id, args.device)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    img_id, ann_id = 1, 1
    root = Path(args.img-dir) if hasattr(args, "img-dir") else Path(args.img_dir)  # accommodates PowerShell hyphen quirk
    img_dir = Path(str(root))  # normalize

    for path in tqdm(list_images(img_dir), desc="Bootstrap"):
        im = cv2.imread(str(path))
        if im is None: 
            continue
        dets, (H,W) = det.infer(im, args.box_thr, args.text_thr)
        rel = path.relative_to(img_dir).as_posix()
        coco["images"].append({"id": img_id, "file_name": rel, "width": W, "height": H})
        for (x1,y1,x2,y2), sc, cls in dets:
            w = max(0.0, x2-x1); h = max(0.0, y2-y1)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": CAT_ID[cls],
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w*h), "iscrowd": 0, "segmentation": [], "score": float(sc)
            })
            ann_id += 1
        img_id += 1

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(coco, f)
    print("Wrote", args.out_json)

if __name__ == "__main__":
    main()
