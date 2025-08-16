# bootstrap_gdino_to_coco.py
import os, cv2, json, itertools
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops

DATA_ROOT = r"/path/to/dataset_root"  # same OUT as above
SPLIT = "train"                        # change to "val" to prelabel val too
IMG_DIR = os.path.join(DATA_ROOT, "images", SPLIT)

MODEL_CFG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CKPT = "weights/groundingdino_swint_ogc.pth"  # update to your path

PROMPTS = {
    "panel": ["comics panels","manga panels","frames","windows"],
    "character": ["characters","comics characters","person","girl","woman","man","animal"],
    "text": ["text box","text","handwriting"],
    "face": ["face","character face","animal face","head","face with nose and mouth","person’s face"],
}
cat_id = {"panel":1, "character":2, "text":3, "face":4}
all_phrases = list(itertools.chain.from_iterable(PROMPTS.values()))
phrase_to_cat = {p:k for k, lst in PROMPTS.items() for p in lst}

def walk_images(img_dir):
    for root,_,files in os.walk(img_dir):
        for fn in sorted(files):
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                yield os.path.join(root, fn)

def main():
    model = load_model(MODEL_CFG, MODEL_CKPT)
    coco = {"images": [], "annotations": [], "categories": [
        {"id":1,"name":"panel"}, {"id":2,"name":"character"},
        {"id":3,"name":"text"}, {"id":4,"name":"face"}
    ]}
    ann_id = 1
    img_id = 1

    for fpath in walk_images(IMG_DIR):
        # file_name relative to split root (book/page-001.jpg)
        file_name = os.path.relpath(fpath, IMG_DIR).replace("\\","/")
        im = cv2.imread(fpath)
        if im is None: 
            continue
        h, w = im.shape[:2]
        coco["images"].append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        # run GDINO
        image_source, image = load_image(fpath)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=", ".join(all_phrases),
            box_threshold=0.30,
            text_threshold=0.25
        )
        # convert rel cxcywh to absolute xyxy
        bxyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        bxyxy[:, [0,2]] *= w
        bxyxy[:, [1,3]] *= h

        for (x1,y1,x2,y2), ph, score in zip(bxyxy.tolist(), phrases, logits.sigmoid().tolist()):
            name = phrase_to_cat.get(ph, None)
            if name is None: 
                continue
            x, y, bw, bh = x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id[name],
                "bbox": [x, y, bw, bh],
                "area": bw*bh,
                "iscrowd": 0,
                "segmentation": [],
                "score": float(score)  # some tools keep this; exporters will drop it
            })
            ann_id += 1
        img_id += 1

    out = os.path.join(DATA_ROOT, "annotations")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{SPLIT}_bootstrap_coco.json"), "w") as f:
        json.dump(coco, f)
    print("Wrote", os.path.join(out, f"{SPLIT}_bootstrap_coco.json"))

if __name__ == "__main__":
    main()
