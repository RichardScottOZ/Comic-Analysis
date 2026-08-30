"""Search utilities for the Stage 3+4 VLM web viewer (v2).

Extends v1 with Stage 4 narrative strip search:
  - load_dataset_v2: loads Stage 3 zarr (query space) + Stage 4 strip embeddings (narrative index)
  - find_similar_strips: given a canonical_id, find pages with similar narrative context
"""

# Re-export query-encoding and search helpers from v1
from search_utils_vlm import (  # noqa: F401
    _enrich_result,
    cosine_search,
    find_similar_pages,
    find_similar_panels,
    get_embedding_for_image,
    get_embedding_for_multimodal,
    get_embedding_for_text,
    get_panel_crop,
    load_model,
    make_json_serializable,
)

import csv
import json
from pathlib import Path

import numpy as np
import zarr


def load_dataset_v2(
    zarr_path: str,
    metadata_path: str,
    manifest_path: str,
    stage4_zarr_path: str,
    vlm_cache_dir: str = "E:/vlm_cache",
    chunk_size: int = 2000,
) -> dict:
    """
    Load Stage 3 zarr (for query-space search) AND Stage 4 strip embeddings
    (for narrative similarity search).

    Reads the Stage 3 panel array in chunks to avoid allocating the full
    ~30 GB array in one shot.

    Returns a dataset dict compatible with all v1 search functions, plus:
      - strip_embs: (N, 512) Stage 4 strip embeddings, one per page
      - canonical_to_idx: canonical_id -> row index
    """
    # --- manifest ---
    print(f"Loading manifest: {manifest_path}", flush=True)
    image_map = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_map[row["canonical_id"]] = row["absolute_image_path"]
    print(f"Loaded {len(image_map):,} manifest rows", flush=True)

    # --- metadata ---
    print(f"Loading metadata: {metadata_path}", flush=True)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata):,} metadata rows", flush=True)

    # --- Stage 3 zarr: page embeddings only (mean-pool on the fly) ---
    # v2 does not build a panel search index to save RAM. Use v1 for panel search.
    print(f"Opening Stage 3 zarr: {zarr_path}", flush=True)
    s3_root = zarr.open(zarr_path, mode="r")
    panel_array = s3_root["panel_embeddings"]
    mask_array  = s3_root["panel_masks"]

    n_total = panel_array.shape[0]
    feature_dim = panel_array.shape[2]

    if len(metadata) > n_total:
        print(f"⚠️ Truncating metadata {len(metadata):,} → {n_total:,} to match zarr.", flush=True)
        metadata = metadata[:n_total]

    n_rows = len(metadata)
    page_embs = np.zeros((n_rows, feature_dim), dtype=np.float32)

    print(f"Building page index (chunked, chunk={chunk_size})...", flush=True)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        emb_chunk  = panel_array[start:end]
        mask_chunk = mask_array[start:end]
        for local_i in range(end - start):
            valid = mask_chunk[local_i].astype(bool)
            valid_panels = emb_chunk[local_i][valid]
            if len(valid_panels) > 0:
                page_embs[start + local_i] = valid_panels.mean(axis=0)
        if start % 100000 == 0:
            print(f"  processed {end:,}/{n_rows:,} rows", flush=True)

    # No panel index in v2
    panel_embs = np.zeros((0, feature_dim), dtype=np.float32)
    panel_meta = []
    canonical_to_idx = {meta["canonical_id"]: i for i, meta in enumerate(metadata)}
    panel_to_idx     = {}

    print(f"Stage 3 page index ready: {n_rows:,} pages (no panel index in v2)", flush=True)

    # --- Stage 4 strip embeddings ---
    print(f"Loading Stage 4 strip embeddings: {stage4_zarr_path}", flush=True)
    s4_root = zarr.open(stage4_zarr_path, mode="r")
    if "strip_embeddings" not in s4_root:
        raise ValueError(f"No 'strip_embeddings' found in {stage4_zarr_path}")

    strip_embs = s4_root["strip_embeddings"][:n_rows].astype(np.float32)
    print(f"Stage 4 strip embeddings ready: {strip_embs.shape[0]:,} pages", flush=True)

    return {
        "metadata":        metadata,
        "image_map":       image_map,
        "page_embs":       page_embs,
        "panel_embs":      panel_embs,
        "panel_meta":      panel_meta,
        "canonical_to_idx": canonical_to_idx,
        "panel_to_idx":    panel_to_idx,
        "strip_embs":      strip_embs,
        "vlm_cache_dir":   Path(vlm_cache_dir),
    }


def find_similar_strips(dataset: dict, seed_canonical_id: str, top_k: int = 12) -> list:
    """
    Given a canonical_id already in the dataset, find pages with the most
    similar Stage 4 strip (narrative context) embedding.

    The seed page itself is excluded from results.
    """
    canonical_to_idx = dataset.get("canonical_to_idx", {})
    if seed_canonical_id not in canonical_to_idx:
        return []

    seed_idx   = canonical_to_idx[seed_canonical_id]
    strip_embs = dataset["strip_embs"]

    if seed_idx >= strip_embs.shape[0]:
        return []

    query_emb = strip_embs[seed_idx : seed_idx + 1]       # (1, 512)
    top_idx, top_scores = cosine_search(query_emb, strip_embs, top_k=top_k + 1)

    results = []
    rank = 1
    for idx, sim in zip(top_idx, top_scores):
        idx = int(idx)
        if idx == seed_idx:
            continue
        meta = dataset["metadata"][idx]
        results.append({
            "rank":            rank,
            "canonical_id":    meta["canonical_id"],
            "similarity":      float(sim),
            "num_panels":      meta.get("num_panels", 0),
            "overall_summary": meta.get("overall_summary", ""),
        })
        rank += 1
        if rank > top_k:
            break

    return [_enrich_result(dataset, r) for r in results]

