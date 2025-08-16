# quick_profile.py
import time, random, os, pathlib
from comix_prod_detect import run_gdino_on_path, run_gdino_tiled, load_model_once

ROOT = "/path/to/books_root"
SAMPLE_PAGES = 300
TILED = False
CKPT = "/path/to/groundingdino_swint_ogc.pth"
CFG  = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

def list_imgs(root):
    for d,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                yield os.path.join(d, f)

def main():
    imgs = list(list_imgs(ROOT))
    random.seed(0)
    sample = random.sample(imgs, min(SAMPLE_PAGES, len(imgs)))
    model = load_model_once(CFG, CKPT, "cuda:0")
    # warmup
    _ = (run_gdino_tiled if TILED else run_gdino_on_path)(model, sample[0], 0.30, 0.25)
    # timing
    t0 = time.time()
    for p in sample:
        _ = (run_gdino_tiled if TILED else run_gdino_on_path)(model, p, 0.30, 0.25)
    dt = time.time() - t0
    print(f"Processed {len(sample)} pages in {dt:.1f}s => {len(sample)/dt:.2f} pages/sec")

if __name__ == "__main__":
    main()
