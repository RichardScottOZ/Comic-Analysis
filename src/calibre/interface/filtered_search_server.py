# filtered_search_server.py
"""
Filtered similarity search server.
Allows users to first filter pages by page_id or text search,
then perform similarity search within that filtered subset.
"""

import sys
import os
import json
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
    cosine_similarity_search
)

# --- Configuration ---
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

print("Initializing filtered search server...")
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

DEFAULT_DATASET_KEY = 'fused' if 'fused' in DATASETS else list(DATASETS.keys())[0]
print(f"Default dataset: {EMBEDDING_CONFIGS[DEFAULT_DATASET_KEY]['name']}")

print(f"Loading model from {CHECKPOINT_PATH}...")
MODEL = load_model(CHECKPOINT_PATH, DEVICE)

print(f"Loading manifest from {MANIFEST_PATH}...")
MANIFEST_DATA = load_manifest(MANIFEST_PATH)

print(f"Initialization complete. Loaded {len(DATASETS)} dataset(s). Server is ready.")

# --- Helper Functions ---

def find_similar_pages_filtered(ds, query_embedding, filtered_indices, top_k=12, query_manifest_path=None):
    """
    Find similar pages within a filtered subset of indices.
    
    Args:
        ds: Zarr dataset
        query_embedding: Query embedding vector
        filtered_indices: List/array of indices to search within
        top_k: Number of results to return
        query_manifest_path: Optional path to ensure query page is included
    
    Returns:
        List of result dictionaries with similarity scores
    """
    if len(filtered_indices) == 0:
        return []
    
    # Get embeddings only for filtered indices
    filtered_embeddings = ds['page_embeddings'].values[filtered_indices]
    
    # Adjust top_k if filtered set is smaller
    actual_top_k = min(top_k, len(filtered_indices))
    
    # Find similar pages within filtered set
    similar_local_indices, similarities = cosine_similarity_search(query_embedding, filtered_embeddings, actual_top_k)
    
    # Map local indices back to global indices
    similar_global_indices = filtered_indices[similar_local_indices]
    
    # Build results
    results = []
    for i, global_idx in enumerate(similar_global_indices):
        results.append({
            'rank': i + 1,
            'page_id': ds['page_id'].values[global_idx],
            'manifest_path': ds['manifest_path'].values[global_idx],
            'similarity': similarities[i],
            'global_index': int(global_idx)
        })
    
    return make_json_serializable(results)


def get_filtered_indices_from_page_id(ds, query_text):
    """
    Get indices of pages matching page_id query.
    
    Returns:
        numpy array of indices
    """
    manifest_paths = ds['manifest_path'].values
    query_lower = query_text.lower()
    
    matching_indices = []
    for idx, path in enumerate(manifest_paths):
        path_str = str(path) if not isinstance(path, str) else path
        if isinstance(path_str, (bytes, bytearray)):
            path_str = path_str.decode('utf-8', errors='ignore')
        
        if query_lower in path_str.lower():
            matching_indices.append(idx)
    
    return np.array(matching_indices, dtype=np.int64)


def get_filtered_indices_from_text(ds, query_text):
    """
    Get indices of pages containing text query.
    
    Returns:
        numpy array of indices
    """
    text_content = ds['text_content'].values
    query_lower = query_text.lower()
    
    matching_indices = []
    for page_idx in range(len(ds['page_id'])):
        # Check all panels for this page
        page_text = text_content[page_idx]
        for panel_text in page_text:
            text_str = _bytes_to_str(panel_text)
            if isinstance(text_str, (list, tuple)) and len(text_str) > 0:
                text_str = text_str[0]
            text_str = str(text_str)
            
            if query_lower in text_str.lower():
                matching_indices.append(page_idx)
                break  # Found match in this page, move to next page
    
    return np.array(matching_indices, dtype=np.int64)


def add_metadata_to_results(results, dataset):
    """Add additional metadata to search results."""
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
    """Process results to add image URLs."""
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
    """Main page with filtered search interface."""
    available_datasets = {
        key: {
            'name': config['name'],
            'description': config['description'],
            'available': key in DATASETS
        }
        for key, config in EMBEDDING_CONFIGS.items()
    }
    return render_template('filtered_index.html', 
                         datasets=available_datasets, 
                         default_dataset=DEFAULT_DATASET_KEY)


@app.route('/filter', methods=['POST'])
def filter_pages():
    """
    First step: Filter pages based on page_id or text search.
    Returns list of matching pages.
    """
    filter_type = request.form.get('filter_type')  # 'page_id' or 'text'
    filter_query = request.form.get('filter_query')
    dataset_key = request.form.get('dataset', DEFAULT_DATASET_KEY)
    
    if dataset_key not in DATASETS:
        return jsonify({"error": f"Invalid dataset selection: {dataset_key}"}), 400
    
    dataset = DATASETS[dataset_key]
    
    try:
        if filter_type == 'page_id':
            filtered_indices = get_filtered_indices_from_page_id(dataset, filter_query)
        elif filter_type == 'text':
            filtered_indices = get_filtered_indices_from_text(dataset, filter_query)
        else:
            return jsonify({"error": "Invalid filter type"}), 400
        
        # Build result list with basic info
        results = []
        for idx in filtered_indices:
            results.append({
                'index': int(idx),
                'page_id': dataset['page_id'].values[idx],
                'manifest_path': dataset['manifest_path'].values[idx]
            })
        
        # Add metadata and process for images
        results = add_metadata_to_results(results, dataset)
        results = process_results(results)
        
        response = {
            'filter_type': filter_type,
            'filter_query': filter_query,
            'total_matches': len(results),
            'results': make_json_serializable(results),
            'dataset_key': dataset_key,
            'dataset_name': EMBEDDING_CONFIGS[dataset_key]['name']
        }
        
        print(f"Filter '{filter_query}' ({filter_type}) returned {len(results)} matches")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in filter_pages: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    """
    Second step: Perform similarity search within filtered subset.
    Requires filtered_indices from previous filter step.
    """
    data = request.get_json()
    
    filtered_indices = data.get('filtered_indices', [])
    query_source = data.get('query_source')  # 'page_id', 'image', 'text'
    query_data = data.get('query_data')  # varies based on source
    dataset_key = data.get('dataset', DEFAULT_DATASET_KEY)
    top_k = data.get('top_k', 12)
    
    if dataset_key not in DATASETS:
        return jsonify({"error": f"Invalid dataset selection: {dataset_key}"}), 400
    
    dataset = DATASETS[dataset_key]
    
    try:
        # Convert filtered_indices to numpy array
        filtered_indices = np.array(filtered_indices, dtype=np.int64)
        
        if len(filtered_indices) == 0:
            return jsonify({
                "error": "No pages in filtered set",
                "results": [],
                "total_filtered": 0
            })
        
        # Get query embedding based on source
        query_embedding = None
        query_manifest_path = None
        
        if query_source == 'page_id':
            # Query by selecting a page from filtered set
            query_manifest_path = query_data
            query_embedding = get_embedding_by_page_id(dataset, query_manifest_path)
            
        elif query_source == 'image':
            # Query by uploading an image
            # Note: This requires handling file upload differently
            return jsonify({"error": "Image upload not yet implemented for filtered search"}), 501
            
        elif query_source == 'text':
            # Query by semantic text search
            query_text = query_data
            query_embedding = get_embedding_for_text(MODEL, query_text, DEVICE)
        
        else:
            return jsonify({"error": "Invalid query source"}), 400
        
        # Perform similarity search within filtered set
        results = find_similar_pages_filtered(
            dataset, 
            query_embedding, 
            filtered_indices, 
            top_k=top_k,
            query_manifest_path=query_manifest_path
        )
        
        # Add metadata and process
        results = add_metadata_to_results(results, dataset)
        results = process_results(results)
        
        response = {
            'query_source': query_source,
            'total_filtered': len(filtered_indices),
            'total_results': len(results),
            'results': make_json_serializable(results),
            'dataset_key': dataset_key,
            'dataset_name': EMBEDDING_CONFIGS[dataset_key]['name']
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in similarity_search: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the image root directory."""
    return send_from_directory(IMAGE_ROOT, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Different port from other servers
