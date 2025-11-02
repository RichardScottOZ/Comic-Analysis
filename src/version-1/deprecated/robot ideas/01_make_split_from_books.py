# make_split_from_books.py
import os, shutil, random, pathlib

ROOT = r"/path/to/your/books_root"  # the folder shown in your screenshot
OUT  = r"/path/to/dataset_root"     # where we create images/train,val
VAL_FRACTION = 0.15                  # change as you wish
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

random.seed(0)

books = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT,d))]
# keep only folders with at least 1 image
def has_image(dp):
    for fn in os.listdir(dp):
        if os.path.splitext(fn)[1].lower() in IMG_EXTS: return True
    return False
books = [b for b in books if has_image(os.path.join(ROOT,b))]

# split by book
val_n = max(1, int(len(books)*VAL_FRACTION))
val_books = set(random.sample(books, val_n))
splits = {"train": [b for b in books if b not in val_books],
          "val":   [b for b in books if b in val_books]}

for split, blist in splits.items():
    for b in blist:
        src_dir = os.path.join(ROOT, b)
        dst_dir = os.path.join(OUT, "images", split, b)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in sorted(os.listdir(src_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS: 
                continue
            src = os.path.join(src_dir, fn)
            dst = os.path.join(dst_dir, fn)
            try:
                os.symlink(src, dst)
            except OSError:
                # fallback to copying on Windows without admin
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

print("Train books:", len(splits["train"]), "Val books:", len(splits["val"]))
print("Dataset images root:", os.path.join(OUT, "images"))
