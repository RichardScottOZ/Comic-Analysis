# search_server.py

print("DEBUG: Loading search_server.py from: " + __file__)
print("DEBUG: Loading search_server.py from: " + __file__)
print("DEBUG: Loading search_server.py from: " + __file__)
print("DEBUG: Loading search_server.py from: " + __file__)
import sys
import os
import json
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import torch

# Add the specific directory containing the framework module to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'benchmarks/detections/openrouter'))
sys.path.insert(0, module_path)

from search_utils import (
    load_zarr_dataset, 
    load_model, 
    load_manifest,
    get_embedding_for_image, 
    get_embedding_for_text,
    find_similar_pages,
    find_similar_panels,
    keyword_search_panels,
    page_id_search,
    get_embedding_by_page_id,
    make_json_serializable,
    _bytes_to_str,
    get_zarr_embedding_by_manifest_path
)

# --- Configuration ---
# Multiple embedding dataset configurations
EMBEDDING_CONFIGS = {
    'fused': {
        'name': 'Fused (Vision + Text + Composition)',
        'zarr_path': "E:\\calibre3\\combined_embeddings.zarr",
        'description': 'Full multimodal embeddings with all modalities fused'
    },
    'vision': {
        'name': 'Vision Only',
        'zarr_path': "E:\\calibre3_vision\\combined_embeddings.zarr",
        'description': 'Pure vision-based embeddings (images only)'
    },
    'text': {
        'name': 'Text Only',
        'zarr_path': "E:\\calibre3_text\\combined_embeddings.zarr",
        'description': 'Pure text-based embeddings (text only)'
    },
    'comp': {
        'name': 'Composition Only',
        'zarr_path': "E:\\calibre3_comp\\combined_embeddings.zarr",
        'description': 'Pure compositional embeddings (layout only)'
    }
}

CHECKPOINT_PATH = "C:\\Users\\Richard\\OneDrive\\GIT\\CoMix\\closure_lite_output\\calibre_perfect_simple_denoise_context\\best_checkpoint.pth"
IMAGE_ROOT = "E:\\CalibreComics_extracted"
MANIFEST_PATH = "perfect_match_training\\calibre_dataspec_final_perfect_list.txt"

# --- Initialization ---
app = Flask(__name__, template_folder='templates')

if not os.path.exists('static'):
    os.makedirs('static')

print("Initializing server...")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load all available datasets
DATASETS = {}
for key, config in EMBEDDING_CONFIGS.items():
    zarr_path = config['zarr_path']
    if os.path.exists(zarr_path):
        print(f"Loading {config['name']} dataset from {zarr_path}...")
        try:
            DATASETS[key] = load_zarr_dataset(zarr_path)
            print(f"  ✓ Loaded {config['name']}")
        except Exception as e:
            print(f"  ✗ Failed to load {config['name']}: {e}")
    else:
        print(f"  ⚠ Skipping {config['name']} - dataset not found at {zarr_path}")

if not DATASETS:
    raise RuntimeError("No embedding datasets could be loaded!")

# Default to fused if available, otherwise use first available
DEFAULT_DATASET_KEY = 'fused' if 'fused' in DATASETS else list(DATASETS.keys())[0]
print(f"Default dataset: {EMBEDDING_CONFIGS[DEFAULT_DATASET_KEY]['name']}")

print(f"Loading model from {CHECKPOINT_PATH}...")
MODEL = load_model(CHECKPOINT_PATH, DEVICE)

print(f"Loading manifest from {MANIFEST_PATH}...")
MANIFEST_DATA = load_manifest(MANIFEST_PATH)

print(f"Initialization complete. Loaded {len(DATASETS)} dataset(s). Server is ready.")

# --- Helper Functions ---
def add_metadata_to_results(results, dataset):
    for item in results:
        if item.get('manifest_path'):
            item['manifest_basename'] = os.path.basename(item['manifest_path'])
        if 'text_snippet' not in item and item.get('page_id') and 'panel_id' in item:
            try:
                text_val = dataset['text_content'].sel(page_id=item['page_id'], panel_id=item['panel_id']).values
                text_str = _bytes_to_str(text_val)
                if hasattr(text_str, 'tolist'): text_str = text_str.tolist()
                if isinstance(text_str, list) and len(text_str) > 0: text_str = text_str[0]
                item['text_snippet'] = str(text_str)
            except Exception as e:
                item['text_snippet'] = "[Text not available]"
    return results

def process_results(results):
    for item in results:
        json_manifest_path = item.get('manifest_path', '')
        item['image_url'] = None
        try:
            if not json_manifest_path or not os.path.exists(json_manifest_path):
                continue
            with open(json_manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            image_path_from_json = data.get('page_image_path') or data.get('image_path')
            item['raw_image_path'] = image_path_from_json
            if not image_path_from_json: continue
            norm_image_path = os.path.normpath(image_path_from_json)
            norm_image_root = os.path.normpath(IMAGE_ROOT)
            if norm_image_path.lower().startswith(norm_image_root.lower()):
                relative_path = norm_image_path[len(norm_image_root):].lstrip('\\/')
                item['image_url'] = f"/images/{relative_path.replace(os.sep, '/')}"
        except Exception as e:
            print(f"Error processing manifest {json_manifest_path}: {e}")
    return results

# --- Routes ---
@app.route('/')
def index():
    # Pass dataset configurations to template
    available_datasets = {
        key: {
            'name': config['name'],
            'description': config['description'],
            'available': key in DATASETS
        }
        for key, config in EMBEDDING_CONFIGS.items()
    }
    return render_template('index.html', 
                         datasets=available_datasets, 
                         default_dataset=DEFAULT_DATASET_KEY)

@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type')
    search_mode = request.form.get('search_mode')
    dataset_key = request.form.get('dataset', DEFAULT_DATASET_KEY)
    
    # Validate dataset selection
    if dataset_key not in DATASETS:
        return jsonify({"error": f"Invalid dataset selection: {dataset_key}"}), 400
    
    dataset = DATASETS[dataset_key]
    dataset_name = EMBEDDING_CONFIGS[dataset_key]['name']
    print(f"Using dataset: {dataset_name}")
    
    results = []
    try:
        if query_type == 'image':
            if 'image_query' not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            image_file = request.files['image_query']
            if not image_file or image_file.filename == '':
                return jsonify({"error": "No image file selected"}), 400

            temp_path = os.path.join("static", "temp_query.jpg")
            image_file.save(temp_path)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return jsonify({"error": "Failed to save uploaded file. It might be empty."}), 500

            print(f"Saved image file: {temp_path}, size: {os.path.getsize(temp_path)} bytes")

            try:
                from PIL import Image
                with Image.open(temp_path) as img:
                    img.verify()
                print(f"PIL successfully verified image: {temp_path}")
            except Exception as e:
                print(f"PIL failed to open or verify image {temp_path}: {e}")
                return jsonify({"error": f"Uploaded file is not a valid image or is corrupted. PIL error: {e}"}), 400

            try:
                panel_embedding, page_embedding = get_embedding_for_image(MODEL, temp_path, DEVICE)
            except Exception as e:
                print(f"Failed to process image {temp_path}: {e}")
                return jsonify({"error": f"Could not process image. It may be corrupt or in an unsupported format. Details: {e}"}), 400

            if search_mode == 'page':
                results = find_similar_pages(dataset, page_embedding, top_k=12)
            else: # panel
                results = find_similar_panels(dataset, panel_embedding, top_k=12)
        
        elif query_type == 'text':
            text_query = request.form.get('text_query')
            if not text_query: return jsonify({"error": "No text query provided"}), 400
            if search_mode == 'semantic':
                query_embedding = get_embedding_for_text(MODEL, text_query, DEVICE)
                results = find_similar_panels(dataset, query_embedding, top_k=12)
            else: # keyword
                results = keyword_search_panels(dataset, text_query, top_k=50)

        elif query_type == 'page_id':
            text_query = request.form.get('text_query')
            if not text_query: return jsonify({"error": "No Page ID query provided"}), 400
            results = page_id_search(MANIFEST_DATA, text_query, top_k=100)
            # For Page ID search, the result is the manifest record itself, which needs processing.
            # We will just display the canonical_id and path for now.
            return jsonify(make_json_serializable(results))

        elif query_type == 'embedding':
            page_id = request.form.get('page_id')
            if not page_id: return jsonify({"error": "No page_id provided for embedding search"}), 400
            query_embedding = get_embedding_by_page_id(dataset, page_id)
            
            if search_mode == 'page':
                results = find_similar_pages(dataset, query_embedding, top_k=12, query_manifest_path=page_id)
            else: # panel
                results = find_similar_panels(dataset, query_embedding, top_k=12)

        else:
            return jsonify({"error": "Invalid query type"}), 400

        results_with_metadata = add_metadata_to_results(results, dataset)
        processed_results = process_results(results_with_metadata)
        serializable_results = make_json_serializable(processed_results)
        
        # Add dataset info to response
        serializable_results.insert(0, {
            'dataset_info': {
                'key': dataset_key,
                'name': dataset_name,
                'description': EMBEDDING_CONFIGS[dataset_key]['description']
            }
        })
        
        return jsonify(serializable_results)

    except Exception as e:
        print(f"An error occurred during search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_ROOT, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)