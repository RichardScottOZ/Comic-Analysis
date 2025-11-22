import os, json, torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, SiglipImageProcessor
from sentence_transformers import SentenceTransformer
from utils.env_paths import paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = os.environ.get("PSS_FP16", "1") == "1"

cfg = paths()
books_root = Path(cfg["root_dir"])
# Adjust annotation path according to existing version layout
ann_path = Path(cfg["data_dir"]) / "v1" / "comics_train.json"
visual_out = Path(cfg["precompute_dir"]) / "visual"
text_out = Path(cfg["precompute_dir"]) / "text"
visual_out.mkdir(parents=True, exist_ok=True)
text_out.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("PSS_VIS_MODEL", "google/siglip-so400m-patch14-384")
TEXT_MODEL_ID = os.environ.get("PSS_TEXT_MODEL", "Qwen/Qwen3-Embedding-0.6B")

print(f"Loading visual backbone {MODEL_ID}")
backbone = AutoModel.from_pretrained(MODEL_ID).eval().to(DEVICE)
processor = (SiglipImageProcessor.from_pretrained(MODEL_ID)
             if "siglip" in MODEL_ID else AutoProcessor.from_pretrained(MODEL_ID))
print(f"Loading text embedding model {TEXT_MODEL_ID}")
text_model = SentenceTransformer(TEXT_MODEL_ID, device=str(DEVICE))

def load_ann():
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ocr(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data.get("OCRResult", {}))
    except Exception:
        return ""

def visual_feats_batch(img_paths):
    images = [Image.open(p).convert("RGB") for p in img_paths]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=USE_FP16, dtype=torch.float16 if USE_FP16 else torch.float32):
        out = backbone(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:,0,:]
    return feats.detach()

def text_feats_batch(texts):
    embeddings = text_model.encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
    return torch.from_numpy(embeddings)

def process_book(book):
    book_id = book["hash_code"]
    book_dir = books_root / book_id
    if not book_dir.is_dir():
        print(f"Missing directory for book {book_id}")
        return
    img_files = sorted([p for p in book_dir.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg"}])
    ocr_files = [p.with_suffix(".json") for p in img_files]
    if not img_files:
        return
    v_save = visual_out / f"{book_id}.pt"
    t_save = text_out / f"{book_id}.pt"
    if v_save.exists() and t_save.exists():
        return
    visual_batches, text_batches = [], []
    BATCH = int(os.environ.get("PSS_PRECOMP_BATCH", "32"))
    for i in range(0, len(img_files), BATCH):
        batch_imgs = img_files[i:i+BATCH]
        feats_v = visual_feats_batch(batch_imgs)
        visual_batches.append(feats_v.cpu())
        texts = [load_ocr(p) for p in ocr_files[i:i+BATCH]]
        feats_t = text_feats_batch(texts)
        text_batches.append(feats_t)
    visual_tensor = torch.cat(visual_batches, dim=0)
    text_tensor = torch.cat(text_batches, dim=0)
    torch.save(visual_tensor, v_save)
    torch.save(text_tensor, t_save)
    print(f"Saved {book_id}: visual {visual_tensor.shape} text {text_tensor.shape}")

def main():
    ann = load_ann()
    for book in tqdm(ann, desc="Books"):
        process_book(book)

if __name__ == "__main__":
    main()
