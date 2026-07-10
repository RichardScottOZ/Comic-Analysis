"""Search utilities for the Stage 3 VLM web viewer."""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import zarr
from PIL import Image, ImageFile
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage3_panel_features_framework import PanelFeatureExtractor


_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _TOKENIZER


def load_model(checkpoint_path: str, device) -> PanelFeatureExtractor:
    """Load the Stage 3 VLM model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print("Instantiating PanelFeatureExtractor...", flush=True)
    model = PanelFeatureExtractor(
        visual_backbone="both",
        visual_fusion="attention",
        feature_dim=512,
        freeze_backbones=True,
    ).to(device)
    print("Loading state dict...", flush=True)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    print("Stage 3 model ready.", flush=True)
    return model


def load_dataset(
    zarr_path: str,
    metadata_path: str,
    manifest_path: str,
    vlm_cache_dir: str = "E:/vlm_cache",
) -> dict:
    """
    Load zarr + metadata + manifest into RAM.

    Returns:
      - metadata: list of dicts
      - image_map: dict canonical_id -> absolute_image_path
      - page_embs: (N, 512) mean-pooled page embeddings
      - panel_embs: (M, 512) flat valid panel embeddings
      - panel_meta: list[(canonical_id, panel_idx)]
    """
    print(f"Loading manifest: {manifest_path}", flush=True)
    image_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_map[row["canonical_id"]] = row["absolute_image_path"]

    print(f"Loaded {len(image_map):,} manifest rows", flush=True)
    print(f"Loading metadata: {metadata_path}", flush=True)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata):,} metadata rows", flush=True)
    print(f"Opening zarr: {zarr_path}", flush=True)
    root = zarr.open(zarr_path, mode="r")

    # Support both Stage 3 (panel_embeddings) and Stage 4 (contextualized_panels + strip_embeddings)
    is_stage4 = "contextualized_panels" in root
    if is_stage4:
        print("Detected Stage 4 zarr format.", flush=True)
        print("Reading contextualized panel embeddings into RAM...", flush=True)
        panel_embeddings = root["contextualized_panels"][:]
        print("Reading strip embeddings into RAM...", flush=True)
        strip_embeddings = root["strip_embeddings"][:]
    else:
        print("Detected Stage 3 zarr format.", flush=True)
        print("Reading panel embeddings into RAM...", flush=True)
        panel_embeddings = root["panel_embeddings"][:]
        strip_embeddings = None
    print("Reading panel masks into RAM...", flush=True)
    panel_masks = root["panel_masks"][:]

    page_embs = []
    panel_embs = []
    panel_meta = []

    print("Building page/panel indices...", flush=True)
    for i, meta in enumerate(metadata):
        valid_mask = panel_masks[i].astype(bool)
        valid_panels = panel_embeddings[i][valid_mask]

        if len(valid_panels) == 0:
            page_embs.append(np.zeros(panel_embeddings.shape[-1], dtype=np.float32))
            continue

        # Use strip embedding for page-level search if available (Stage 4), else mean-pool
        if strip_embeddings is not None:
            page_embs.append(strip_embeddings[i].astype(np.float32))
        else:
            page_embs.append(valid_panels.mean(axis=0))

        for panel_idx, emb in enumerate(valid_panels):
            panel_embs.append(emb)
            panel_meta.append((meta["canonical_id"], panel_idx))

    print(
        f"Dataset ready: {len(page_embs):,} pages, {len(panel_embs):,} panels",
        flush=True,
    )
    return {
        "metadata": metadata,
        "image_map": image_map,
        "page_embs": np.stack(page_embs).astype(np.float32),
        "panel_embs": np.stack(panel_embs).astype(np.float32),
        "panel_meta": panel_meta,
        "vlm_cache_dir": Path(vlm_cache_dir),
    }


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x)
    return str(value)


def _extract_page_number(canonical_id: str) -> str:
    page_name = canonical_id.split("/")[-1]
    matches = [
        r"page[_\- ]?(\d+)$",
        r"p(\d+)$",
        r"[_\- ](\d+)$",
    ]
    stem = Path(page_name).stem
    for pattern in matches:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1).lstrip("0") or "0"
    return ""


def _display_parts(canonical_id: str) -> tuple:
    parts = canonical_id.split("/")
    page_name = parts[-1] if parts else canonical_id
    comic_name = parts[-2] if len(parts) > 1 else canonical_id
    return comic_name, page_name, _extract_page_number(canonical_id)


def _load_vlm_json(dataset: dict, canonical_id: str) -> dict:
    path = dataset["vlm_cache_dir"] / f"{canonical_id}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_panels(vlm_data: dict) -> list:
    panels = vlm_data.get("panels") or []
    if isinstance(panels, dict):
        panels = list(panels.values())
    return panels if isinstance(panels, list) else []


def _panel_text(panel: dict) -> str:
    parts = []
    desc = _to_text(panel.get("description")).strip()
    if desc:
        parts.append(desc)
    for tc in (panel.get("text_content") or []):
        if not isinstance(tc, dict):
            continue
        txt = _to_text(tc.get("text")).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()


def _panel_dialogue_lines(panel: dict) -> list:
    lines = []
    for tc in (panel.get("text_content") or []):
        if not isinstance(tc, dict):
            continue
        txt = _to_text(tc.get("text")).strip()
        if not txt:
            continue
        speaker = _to_text(tc.get("speaker")).strip()
        if speaker:
            lines.append(f"{speaker}: {txt}")
        else:
            lines.append(txt)
    return lines


def _page_text_preview(vlm_data: dict, max_panels: int = 4, max_chars: int = 500) -> str:
    texts = []
    for panel in _get_panels(vlm_data)[:max_panels]:
        if not isinstance(panel, dict):
            continue
        panel_text = _panel_text(panel)
        if panel_text:
            texts.append(panel_text)
    preview = " | ".join(texts)
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3].rstrip() + "..."
    return preview


def _enrich_result(dataset: dict, result: dict) -> dict:
    canonical_id = result["canonical_id"]
    comic_name, page_name, page_number = _display_parts(canonical_id)
    enriched = dict(result)
    enriched["comic_name"] = comic_name
    enriched["page_name"] = page_name
    enriched["page_number"] = page_number

    vlm_data = _load_vlm_json(dataset, canonical_id)
    if "overall_summary" not in enriched or not enriched.get("overall_summary"):
        enriched["overall_summary"] = _to_text(vlm_data.get("overall_summary")).strip()

    if "panel_idx" in enriched:
        panels = _get_panels(vlm_data)
        idx = enriched["panel_idx"]
        if 0 <= idx < len(panels) and isinstance(panels[idx], dict):
            enriched["panel_text"] = _panel_text(panels[idx])
            enriched["panel_description"] = _to_text(panels[idx].get("description")).strip()
            enriched["panel_dialogue"] = _panel_dialogue_lines(panels[idx])
            enriched["panel_box"] = panels[idx].get("box_2d")
        else:
            enriched["panel_text"] = ""
            enriched["panel_description"] = ""
            enriched["panel_dialogue"] = []
            enriched["panel_box"] = None
    else:
        enriched["page_text_preview"] = _page_text_preview(vlm_data)

    return enriched


def get_panel_crop(dataset: dict, canonical_id: str, panel_idx: int):
    """Return a PIL crop for a panel result, or None if unavailable."""
    image_path = dataset["image_map"].get(canonical_id)
    if not image_path or not Path(image_path).exists():
        return None

    vlm_data = _load_vlm_json(dataset, canonical_id)
    panels = _get_panels(vlm_data)
    if panel_idx < 0 or panel_idx >= len(panels):
        return None

    panel = panels[panel_idx]
    if not isinstance(panel, dict):
        return None

    box_2d = panel.get("box_2d")
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None

    try:
        y1, x1, y2, x2 = [float(v) if v is not None else 0.0 for v in box_2d]
    except (TypeError, ValueError):
        return None

    image = Image.open(image_path).convert("RGB")
    page_w, page_h = image.size
    px1 = max(0, int(x1 / 1000.0 * page_w))
    py1 = max(0, int(y1 / 1000.0 * page_h))
    px2 = min(page_w, int(x2 / 1000.0 * page_w))
    py2 = min(page_h, int(y2 / 1000.0 * page_h))

    if px2 <= px1 or py2 <= py1:
        return None

    return image.crop((px1, py1, px2, py2))


def get_embedding_for_text(model, text_query: str, device) -> np.ndarray:
    """
    Encode text query using model.encode_text_only().

    Returns:
        (1, 512) numpy array
    """
    tokenizer = _get_tokenizer()
    enc = tokenizer(
        [text_query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        emb = model.encode_text_only(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return emb.cpu().numpy()


def get_embedding_for_image(model, image_path: str, device) -> tuple:
    """
    Encode image query with the full Stage 3 model.

    Returns:
        (panel_emb, page_emb), both (1, 512)
    """
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    tokenizer = _get_tokenizer()
    enc = tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    batch = {
        "images": image_tensor,
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "comp_feats": torch.zeros(1, 7, device=device),
        "modality_mask": torch.tensor([[1.0, 0.0, 1.0]], device=device),
    }

    with torch.no_grad():
        emb = model(batch)

    emb_np = emb.cpu().numpy()
    return emb_np, emb_np


def get_embedding_for_multimodal(model, image_path: str, text_query: str, device) -> tuple:
    """
    Encode a combined image+text query with the fused Stage 3 model.

    Returns:
        (panel_emb, page_emb), both (1, 512)
    """
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    tokenizer = _get_tokenizer()
    enc = tokenizer(
        [text_query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    batch = {
        "images": image_tensor,
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "comp_feats": torch.zeros(1, 7, device=device),
        "modality_mask": torch.tensor([[1.0, 1.0, 0.0]], device=device),
    }

    with torch.no_grad():
        emb = model(batch)

    emb_np = emb.cpu().numpy()
    return emb_np, emb_np


def cosine_search(query_emb: np.ndarray, database_embs: np.ndarray, top_k: int = 12) -> tuple:
    """
    Batched cosine similarity search.

    Returns:
        (top_indices, top_similarities)
    """
    q = query_emb.reshape(1, -1).astype(np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    chunk_size = 50000
    all_scores = np.empty(database_embs.shape[0], dtype=np.float32)

    for start in range(0, database_embs.shape[0], chunk_size):
        end = min(start + chunk_size, database_embs.shape[0])
        batch = database_embs[start:end]
        batch = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8)
        all_scores[start:end] = np.dot(batch, q.T).squeeze(-1)

    top_k = min(top_k, len(all_scores))
    top_idx = np.argpartition(all_scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(all_scores[top_idx])[::-1]]
    return top_idx, all_scores[top_idx]


def find_similar_pages(dataset: dict, query_emb: np.ndarray, top_k: int = 12) -> list:
    """Search page embeddings."""
    top_idx, top_scores = cosine_search(query_emb, dataset["page_embs"], top_k)
    results = []
    for rank, (idx, sim) in enumerate(zip(top_idx, top_scores), 1):
        meta = dataset["metadata"][int(idx)]
        results.append(
            {
                "rank": rank,
                "canonical_id": meta["canonical_id"],
                "similarity": float(sim),
                "num_panels": meta.get("num_panels", 0),
                "overall_summary": meta.get("overall_summary", ""),
            }
        )
    return [_enrich_result(dataset, r) for r in results]


def find_similar_panels(dataset: dict, query_emb: np.ndarray, top_k: int = 12) -> list:
    """Search panel embeddings."""
    top_idx, top_scores = cosine_search(query_emb, dataset["panel_embs"], top_k)
    results = []
    for rank, (idx, sim) in enumerate(zip(top_idx, top_scores), 1):
        canonical_id, panel_idx = dataset["panel_meta"][int(idx)]
        results.append(
            {
                "rank": rank,
                "canonical_id": canonical_id,
                "panel_idx": int(panel_idx),
                "similarity": float(sim),
            }
        )
    return [_enrich_result(dataset, r) for r in results]


def make_json_serializable(obj):
    """Recursively convert numpy types to Python natives."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj
