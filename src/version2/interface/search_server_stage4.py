"""Flask web UI for Stage 4 Semantic search using contextualized page/panel embeddings."""

import os
import sys
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_utils_vlm import (
    find_similar_pages,
    find_similar_panels,
    get_embedding_for_image,
    get_embedding_for_multimodal,
    get_embedding_for_text,
    get_panel_crop,
    load_dataset,
    load_model,
    make_json_serializable,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

# Default to Stage 4 embeddings
ZARR_PATH = os.environ.get("STAGE4_SEARCH_ZARR", "E:/stage4_embeddings.zarr")
if not os.path.exists(ZARR_PATH) and os.path.exists("E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"):
    ZARR_PATH = "E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"

METADATA_PATH = Path(
    os.environ.get("STAGE4_SEARCH_METADATA", str(REPO_ROOT / "stage4_metadata.json"))
)
MANIFEST_PATH = Path(
    os.environ.get(
        "STAGE4_SEARCH_MANIFEST",
        str(REPO_ROOT / "manifests" / "master_manifest_20251229.csv"),
    )
)
CHECKPOINT_PATH = Path(
    os.environ.get(
        "STAGE4_SEARCH_CHECKPOINT",
        str(REPO_ROOT / "checkpoints" / "stage3_vlm" / "best_model_vlm.pt"),
    )
)
VLM_CACHE_DIR = os.environ.get("STAGE4_SEARCH_VLM_CACHE", "E:/vlm_cache")


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

Path(app.static_folder).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Initializing Stage 4 Semantic Search Server...")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"• Device:        {DEVICE}")
print(f"• Zarr path:     {ZARR_PATH}")
print(f"• Metadata path: {METADATA_PATH}")
print(f"• VLM checkpoint: {CHECKPOINT_PATH} (Query Encoder)")
print(f"• VLM cache:     {VLM_CACHE_DIR}")

# Load Stage 3 model to encode queries
MODEL = load_model(str(CHECKPOINT_PATH), DEVICE)

# Load dataset using Zarr and metadata
# Since search_utils_vlm.load_dataset automatically detects contextualized_panels, it will load Stage 4
DATASET = load_dataset(
    str(ZARR_PATH),
    str(METADATA_PATH),
    str(MANIFEST_PATH),
    vlm_cache_dir=str(VLM_CACHE_DIR),
)
print("=" * 60)
print("Stage 4 Initialization complete.")
print("=" * 60)


@app.route("/")
def index():
    return render_template("index_stage4.html")


@app.route("/search", methods=["POST"])
def search():
    query_type = request.form.get("query_type")
    search_mode = request.form.get("search_mode")

    try:
        if query_type == "image":
            if "image_query" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            image_file = request.files["image_query"]
            if not image_file or image_file.filename == "":
                return jsonify({"error": "No image file selected"}), 400

            temp_path = Path(app.static_folder) / "temp_query.jpg"
            image_file.save(temp_path)
            panel_emb, page_emb = get_embedding_for_image(MODEL, str(temp_path), DEVICE)
            query_emb = page_emb if search_mode == "page" else panel_emb

        elif query_type == "multimodal":
            if "image_query" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            image_file = request.files["image_query"]
            if not image_file or image_file.filename == "":
                return jsonify({"error": "No image file selected"}), 400

            text_query = request.form.get("text_query", "").strip()
            if not text_query:
                return jsonify({"error": "No multimodal text provided"}), 400

            temp_path = Path(app.static_folder) / "temp_query.jpg"
            image_file.save(temp_path)
            panel_emb, page_emb = get_embedding_for_multimodal(
                MODEL, str(temp_path), text_query, DEVICE
            )
            query_emb = page_emb if search_mode == "page" else panel_emb

        elif query_type == "text":
            text_query = request.form.get("text_query", "").strip()
            if not text_query:
                return jsonify({"error": "No text query provided"}), 400
            query_emb = get_embedding_for_text(MODEL, text_query, DEVICE)

        else:
            return jsonify({"error": "Invalid query type"}), 400

        # Perform cosine similarity search against contextualized panels/strips
        if search_mode == "page":
            results = find_similar_pages(DATASET, query_emb, top_k=12)
        elif search_mode == "panel":
            results = find_similar_panels(DATASET, query_emb, top_k=12)
        else:
            return jsonify({"error": "Invalid search mode"}), 400

        return jsonify(make_json_serializable(results))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {e}"}), 500


@app.route("/image/<path:canonical_id>")
def serve_image(canonical_id):
    image_path = DATASET["image_map"].get(canonical_id)
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(image_path)


@app.route("/panel_image/<path:canonical_id>/<int:panel_idx>")
def serve_panel_image(canonical_id, panel_idx):
    panel_image = get_panel_crop(DATASET, canonical_id, panel_idx)
    if panel_image is None:
        return jsonify({"error": "Panel image not found"}), 404

    buf = BytesIO()
    panel_image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    # Serve on port 5004 for Stage 4
    app.run(host="127.0.0.1", debug=False, port=5004)
