"""
Classification-only inference script for CoSMo PSS pipeline.

Loads precomputed visual and text embeddings and runs the BookBERT classifier
(CoSMo v4, available at richardscottoz/cosmo-v4 on HuggingFace).
"""
import os, json, torch
from pathlib import Path
from tqdm import tqdm
from torch.nn.functional import softmax
from utils.env_paths import paths
from models.book_bert import BookBERTMultimodal2

cfg = paths()
precomp = Path(cfg["precompute_dir"]) / "visual", Path(cfg["precompute_dir"]) / "text"
visual_dir, text_dir = precomp
ann_path = Path(cfg["data_dir"]) / "v1" / "comics_val.json"
checkpoint_dir = Path(cfg["checkpoint_dir"])  # ensure existing model
out_dir = Path(cfg["output_dir"]) / "labels"
out_dir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = os.environ.get("PSS_CLASSIFIER_CKPT", str(checkpoint_dir / "best_BookBERT.pt"))

# Hyperparameters (must match training)
NUM_ATTENTION = int(os.environ.get("PSS_NUM_HEADS", "4"))
NUM_LAYERS = int(os.environ.get("PSS_NUM_LAYERS", "4"))
DROPOUT = float(os.environ.get("PSS_DROPOUT", "0.4"))
HIDDEN = int(os.environ.get("PSS_HIDDEN", "256"))
VIS_DIM = int(os.environ.get("PSS_VIS_DIM", "1152"))
TXT_DIM = int(os.environ.get("PSS_TXT_DIM", "1024"))
NUM_CLASSES = int(os.environ.get("PSS_NUM_CLASSES", "9"))
BERT_INPUT_DIM = int(os.environ.get("PSS_BERT_INPUT", "768"))
PROJECTION_DIM = int(os.environ.get("PSS_PROJ_DIM", "1024"))
POSITIONAL = os.environ.get("PSS_POSITIONAL", "absolute")

def load_ann():
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)

def fuse_book(book_id):
    v_path = visual_dir / f"{book_id}.pt"
    t_path = text_dir / f"{book_id}.pt"
    if not v_path.exists() or not t_path.exists():
        return None
    v = torch.load(v_path)
    t = torch.load(t_path)
    if len(v) != len(t):
        print(f"Length mismatch for {book_id}")
        return None
    return torch.cat([v, t], dim=1)

def main():
    model = BookBERTMultimodal2(
        textual_feature_dim=TXT_DIM,
        visual_feature_dim=VIS_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN,
        num_attention_heads=NUM_ATTENTION,
        bert_input_dim=BERT_INPUT_DIM,
        projection_dim=PROJECTION_DIM,
        num_hidden_layers=NUM_LAYERS,
        dropout_p=DROPOUT,
        positional_embeddings=POSITIONAL
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
    model.eval()

    ann = load_ann()
    for book in tqdm(ann, desc="Classifying books"):
        book_id = book["hash_code"]
        feats = fuse_book(book_id)
        if feats is None:
            continue
        with torch.inference_mode():
            feats = feats.to(DEVICE)
            logits = model.forward_sequence(feats)
            probs = softmax(logits, dim=-1).cpu()
        torch.save(probs, out_dir / f"{book_id}_probs.pt")

if __name__ == "__main__":
    main()
