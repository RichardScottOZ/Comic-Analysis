# panel_viz_server.py
"""
Enhanced search server with panel-level visualization
Combines similarity search with detailed panel analysis similar to demo scripts
"""

import sys
import os
import json
import csv
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64

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
if not os.path.exists('static/panel_viz'):
    os.makedirs('static/panel_viz')

print("Initializing panel visualization server...")
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

def get_page_data_from_zarr(manifest_path, dataset_key='fused'):
    """
    Extract page data from Zarr dataset including panel embeddings, coordinates, etc.
    """
    try:
        dataset = DATASETS[dataset_key]
        
        # Normalize manifest path
        manifest_path_norm = os.path.normpath(manifest_path).replace(os.sep, '\\').lower()
        
        # Get page_id from manifest
        all_page_ids = dataset['page_id'].values
        all_manifest_paths = dataset['manifest_path'].values
        
        # Find matching page_id
        page_idx = None
        for idx, (pid, mpath) in enumerate(zip(all_page_ids, all_manifest_paths)):
            mpath_str = _bytes_to_str(mpath)
            if hasattr(mpath_str, 'tolist'):
                mpath_str = mpath_str.tolist()
            if isinstance(mpath_str, list) and len(mpath_str) > 0:
                mpath_str = mpath_str[0]
            mpath_str = str(mpath_str)
            
            mpath_norm = os.path.normpath(mpath_str).replace(os.sep, '\\').lower()
            if mpath_norm == manifest_path_norm:
                page_idx = idx
                page_id = _bytes_to_str(pid)
                if hasattr(page_id, 'tolist'):
                    page_id = page_id.tolist()
                if isinstance(page_id, list) and len(page_id) > 0:
                    page_id = page_id[0]
                break
        
        if page_idx is None:
            return None
        
        # Extract data for this page
        panel_embeddings = dataset['panel_embeddings'][page_idx].values  # Shape: (max_panels, embedding_dim)
        panel_mask = dataset['panel_mask'][page_idx].values  # Shape: (max_panels,)
        panel_coordinates = dataset['panel_coordinates'][page_idx].values  # Shape: (max_panels, 4)
        attention_weights = dataset['attention_weights'][page_idx].values  # Shape: (max_panels,)
        page_embedding = dataset['page_embeddings'][page_idx].values  # Shape: (embedding_dim,)
        text_content = dataset['text_content'][page_idx].values  # Shape: (max_panels,)
        
        # Filter to actual panels (not padding)
        num_panels = int(panel_mask.sum())
        
        page_data = {
            'page_id': str(page_id),
            'manifest_path': manifest_path,
            'num_panels': num_panels,
            'panel_embeddings': panel_embeddings[:num_panels],  # (num_panels, embedding_dim)
            'panel_coordinates': panel_coordinates[:num_panels],  # (num_panels, 4)
            'attention_weights': attention_weights[:num_panels],  # (num_panels,)
            'page_embedding': page_embedding,  # (embedding_dim,)
            'text_content': [_bytes_to_str(text_content[i]) for i in range(num_panels)]
        }
        
        return page_data
        
    except Exception as e:
        print(f"Error extracting page data from Zarr: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_panel_visualization(page_data, manifest_path, dataset_key='fused'):
    """
    Create a comprehensive panel visualization similar to demo scripts
    Returns base64 encoded image
    """
    try:
        # Load the page JSON to get image path
        with open(manifest_path, 'r', encoding='utf-8') as f:
            page_json = json.load(f)
        
        img_path = page_json.get('page_image_path') or page_json.get('image_path')
        if not img_path:
            return None
        
        # Ensure absolute path
        if not os.path.isabs(img_path):
            img_path = os.path.join(IMAGE_ROOT, img_path)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        W_orig, H_orig = img.size
        
        # Get panel data
        panel_embeddings = torch.from_numpy(page_data['panel_embeddings']).float()
        panel_coords = page_data['panel_coordinates']
        attention_weights = page_data['attention_weights']
        num_panels = page_data['num_panels']
        
        # Resize image if too large to prevent memory issues
        # CRITICAL: Keep this small to avoid matplotlib memory allocation errors
        max_dim = 600  # Keep small to prevent "bad allocation" errors
        resize_scale = 1.0
        if max(W_orig, H_orig) > max_dim:
            resize_scale = max_dim / max(W_orig, H_orig)
            new_w = int(W_orig * resize_scale)
            new_h = int(H_orig * resize_scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            W, H = new_w, new_h
            print(f"Resized image from {W_orig}x{H_orig} to {W}x{H} (scale={resize_scale:.3f})")
        else:
            W, H = W_orig, H_orig
            print(f"Image size {W}x{H} is within limits")
        
        # Clean up any existing figures to free memory
        plt.close('all')
        import gc
        gc.collect()
        
        # Create figure with subplots (reduced size to prevent memory issues)
        # Use smaller DPI and figure size to reduce memory footprint
        # CRITICAL: Keep figsize small - matplotlib allocates memory based on figsize * dpi
        figsize = (18, 10)  # inches - matches demo_compare_models.py
        dpi = 80  # Reasonable DPI
        pixel_w = int(figsize[0] * dpi)
        pixel_h = int(figsize[1] * dpi)
        print(f"Creating figure: {figsize} inches @ {dpi} DPI = {pixel_w}x{pixel_h} pixels")
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Use GridSpec matching demo_compare_models.py layout
        gs = fig.add_gridspec(2, 3)
        
        # 1. Main image with panels - spans both rows, first column
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(img)
        ax1.set_title('Panel Detection & Layout', fontsize=12, fontweight='bold', pad=10)
        
        # Draw panels
        colors = plt.cm.Set3(np.linspace(0, 1, num_panels))
        
        # Debug: Check coordinate format
        print(f"Panel coordinates sample (first panel): {panel_coords[0]}")
        print(f"Original image dimensions: W_orig={W_orig}, H_orig={H_orig}")
        print(f"Display image dimensions: W={W}, H={H}, resize_scale={resize_scale}")
        
        for i in range(num_panels):
            coords = panel_coords[i]
            
            # Coordinates from zarr are in PIXEL space [x, y, w, h] relative to ORIGINAL image
            # They need to be scaled to the resized display image
            x_orig, y_orig, w_orig, h_orig = coords
            
            # Scale to display coordinates using resize_scale
            x = x_orig * resize_scale
            y = y_orig * resize_scale
            w = w_orig * resize_scale
            h = h_orig * resize_scale
            
            print(f"Panel {i}: orig=[{x_orig:.1f}, {y_orig:.1f}, {w_orig:.1f}, {h_orig:.1f}] -> display=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
            
            # Draw panel box
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor=colors[i], facecolor='none', alpha=0.8)
            ax1.add_patch(rect)
            
            # Add panel number
            ax1.text(x + 5, y + 15, str(i), fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor=colors[i], alpha=0.9))
        
        ax1.set_xlim(0, W)
        ax1.set_ylim(H, 0)
        ax1.axis('off')
        
        # 2. Panel similarity heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        if num_panels > 1:
            normalized_panels = F.normalize(panel_embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_panels, normalized_panels.t()).cpu().numpy()
            im = ax2.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax2.set_title(f'Panel Similarity Matrix\n({EMBEDDING_CONFIGS[dataset_key]["name"]})', 
                         fontsize=10, fontweight='bold', pad=10)
            ax2.set_xlabel('Panel Index', fontsize=9)
            ax2.set_ylabel('Panel Index', fontsize=9)
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.text(0.5, 0.5, 'Single Panel', ha='center', va='center', fontsize=12)
            ax2.set_title('Panel Similarity', fontsize=10, fontweight='bold', pad=10)
        ax2.set_aspect('auto')
        
        # 3. Attention weights
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(range(num_panels), attention_weights, color=colors)
        ax3.set_title('Panel Attention Weights', fontsize=10, fontweight='bold', pad=10)
        ax3.set_xlabel('Panel Index', fontsize=9)
        ax3.set_ylabel('Attention Weight', fontsize=9)
        ax3.set_xticks(range(num_panels))
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Panel embeddings UMAP
        ax4 = fig.add_subplot(gs[1, 1])
        if num_panels > 1:
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(2, num_panels - 1),
                    min_dist=0.3,
                    spread=1.0,
                    metric='cosine'
                )
                embeddings_2d = reducer.fit_transform(panel_embeddings.cpu().numpy())
                
                scatter = ax4.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                    c=range(num_panels), cmap='Set3', s=100, alpha=0.7)
                ax4.set_title('Panel Embeddings (UMAP)', fontsize=10, fontweight='bold', pad=10)
                ax4.set_xlabel('UMAP 1', fontsize=9)
                ax4.set_ylabel('UMAP 2', fontsize=9)
                ax4.grid(alpha=0.3)
                
                # Add panel labels
                for i, (x, y) in enumerate(embeddings_2d):
                    ax4.annotate(f'P{i}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontweight='bold', fontsize=8)
                    
            except ImportError:
                ax4.text(0.5, 0.5, 'UMAP not installed\nInstall with:\npip install umap-learn',
                        ha='center', va='center', fontsize=9)
                ax4.set_title('Panel Embeddings (UMAP)', fontsize=10, fontweight='bold', pad=10)
            except Exception as e:
                ax4.text(0.5, 0.5, f'UMAP failed:\n{str(e)[:50]}...',
                        ha='center', va='center', fontsize=9)
                ax4.set_title('Panel Embeddings (UMAP)', fontsize=10, fontweight='bold', pad=10)
        else:
            ax4.text(0.5, 0.5, f'Need 2+ panels for UMAP\n(has {num_panels})',
                    ha='center', va='center', fontsize=10)
            ax4.set_title('Panel Embeddings (UMAP)', fontsize=10, fontweight='bold', pad=10)
        ax4.set_aspect('auto')
        
        # 5. Panel text preview
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        ax5.set_title('Panel Text Content', fontsize=10, fontweight='bold', pad=10)
        
        # Show text for each panel
        text_display = []
        for i in range(min(num_panels, 8)):  # Show up to 8 panels
            text_val = page_data['text_content'][i]
            if isinstance(text_val, (list, np.ndarray)):
                text_val = text_val[0] if len(text_val) > 0 else ""
            text_str = str(text_val)[:80]  # Truncate to 80 chars
            text_display.append(f"P{i}: {text_str}")
        
        text_content = "\n".join(text_display)
        ax5.text(0.05, 0.95, text_content, transform=ax5.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                wrap=True)
        
        # No tight_layout needed with GridSpec - already properly spaced
        # Match demo_compare_models.py save approach
        fig.tight_layout()
        
        # Convert to base64 with minimal memory usage
        buf = io.BytesIO()
        try:
            # Match demo_compare_models.py save approach
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            print(f"Successfully created visualization image")
        finally:
            buf.close()
            plt.close(fig)
            plt.close('all')  # Ensure all figures are closed
        
        # Clean up
        del img, panel_embeddings, attention_weights, panel_coords
        import gc
        gc.collect()  # Force garbage collection
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating panel visualization: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        plt.close('all')
        return None

def compare_datasets_visualization(page_data_dict, manifest_path):
    """
    Create a comparison visualization across multiple datasets (e.g., fused vs vision)
    page_data_dict: {'fused': page_data, 'vision': page_data}
    """
    try:
        # Load the page JSON to get image path
        with open(manifest_path, 'r', encoding='utf-8') as f:
            page_json = json.load(f)
        
        img_path = page_json.get('page_image_path') or page_json.get('image_path')
        if not img_path:
            return None
        
        if not os.path.isabs(img_path):
            img_path = os.path.join(IMAGE_ROOT, img_path)
        
        if not os.path.exists(img_path):
            return None
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        W, H = img.size
        
        # Resize image if too large to prevent memory issues
        max_dim = 2000
        if max(W, H) > max_dim:
            scale = max_dim / max(W, H)
            new_w = int(W * scale)
            new_h = int(H * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            W, H = new_w, new_h
        
        num_datasets = len(page_data_dict)
        
        # Create figure with comparison layout (reduced size to prevent memory issues)
        fig = plt.figure(figsize=(8 * num_datasets, 10))
        
        dataset_keys = list(page_data_dict.keys())
        
        for col_idx, dataset_key in enumerate(dataset_keys):
            page_data = page_data_dict[dataset_key]
            panel_embeddings = torch.from_numpy(page_data['panel_embeddings']).float()
            panel_coords = page_data['panel_coordinates']
            attention_weights = page_data['attention_weights']
            num_panels = page_data['num_panels']
            
            # Panel similarity heatmap for this dataset
            ax = plt.subplot(2, num_datasets, col_idx + 1)
            if num_panels > 1:
                normalized_panels = F.normalize(panel_embeddings, p=2, dim=1)
                similarity_matrix = torch.mm(normalized_panels, normalized_panels.t()).cpu().numpy()
                im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'{EMBEDDING_CONFIGS[dataset_key]["name"]}\nPanel Similarity',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Panel Index')
                ax.set_ylabel('Panel Index')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'Single Panel', ha='center', va='center', fontsize=14)
                ax.set_title(f'{EMBEDDING_CONFIGS[dataset_key]["name"]}\nPanel Similarity',
                           fontsize=12, fontweight='bold')
            
            # Attention weights for this dataset
            ax = plt.subplot(2, num_datasets, num_datasets + col_idx + 1)
            colors = plt.cm.Set3(np.linspace(0, 1, num_panels))
            bars = ax.bar(range(num_panels), attention_weights, color=colors)
            ax.set_title(f'{EMBEDDING_CONFIGS[dataset_key]["name"]}\nAttention Weights',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Panel Index')
            ax.set_ylabel('Attention Weight')
            ax.set_xticks(range(num_panels))
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout(pad=1.0)
        
        # Convert to base64 with reduced DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
        
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Routes ---

@app.route('/')
def index():
    available_datasets = {
        key: {
            'name': config['name'],
            'description': config['description'],
            'available': key in DATASETS
        }
        for key, config in EMBEDDING_CONFIGS.items()
    }
    return render_template('panel_viz_index.html',
                         datasets=available_datasets,
                         default_dataset=DEFAULT_DATASET_KEY)

@app.route('/search', methods=['POST'])
def search():
    """Standard search endpoint (page-level similarity)"""
    query_type = request.form.get('query_type')
    dataset_key = request.form.get('dataset', DEFAULT_DATASET_KEY)
    
    if dataset_key not in DATASETS:
        return jsonify({"error": f"Invalid dataset selection: {dataset_key}"}), 400
    
    dataset = DATASETS[dataset_key]
    dataset_name = EMBEDDING_CONFIGS[dataset_key]['name']
    print(f"Using dataset: {dataset_name}")
    
    results = []
    try:
        if query_type == 'page_id':
            search_term = request.form.get('page_id_query', '').strip()
            if not search_term:
                return jsonify({"error": "No search term provided"}), 400
            
            results = page_id_search(MANIFEST_DATA, search_term, top_k=50)
        
        elif query_type == 'text':
            text_query = request.form.get('text_query', '').strip()
            if not text_query:
                return jsonify({"error": "No text query provided"}), 400
            
            page_embedding = get_embedding_for_text(MODEL, text_query, DEVICE)
            results = find_similar_pages(dataset, page_embedding, top_k=12)
        
        else:
            return jsonify({"error": f"Unknown query type: {query_type}"}), 400
        
        # Process results to add image URLs
        for item in results:
            # The manifest can have either 'manifest_path' or 'canonical_id' key
            json_manifest_path = item.get('manifest_path') or item.get('canonical_id', '')
            
            # If json_manifest_path is relative, make it absolute
            if json_manifest_path and not os.path.isabs(json_manifest_path):
                repo_root = os.path.dirname(os.path.abspath(__file__))
                json_manifest_path = os.path.join(repo_root, json_manifest_path)
            
            item['manifest_path'] = json_manifest_path  # Ensure this key exists
            item['image_url'] = None
            try:
                if not json_manifest_path or not os.path.exists(json_manifest_path):
                    print(f"Warning: JSON manifest not found: {json_manifest_path}")
                    continue
                with open(json_manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                image_path_from_json = data.get('page_image_path') or data.get('image_path')
                item['raw_image_path'] = image_path_from_json
                if not image_path_from_json:
                    print(f"Warning: No image path in JSON: {json_manifest_path}")
                    continue
                norm_image_path = os.path.normpath(image_path_from_json)
                norm_image_root = os.path.normpath(IMAGE_ROOT)
                if norm_image_path.lower().startswith(norm_image_root.lower()):
                    relative_path = norm_image_path[len(norm_image_root):].lstrip('\\/')
                    item['image_url'] = f"/images/{relative_path.replace(os.sep, '/')}"
                else:
                    print(f"Warning: Image path not under IMAGE_ROOT: {image_path_from_json}")
            except Exception as e:
                print(f"Error processing manifest {json_manifest_path}: {e}")
        
        results = make_json_serializable(results)
        
        return jsonify({
            "results": results,
            "dataset_used": dataset_name,
            "dataset_key": dataset_key
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/visualize_panel', methods=['POST'])
def visualize_panel():
    """Generate panel-level visualization for a specific page"""
    print("="*70)
    print("🔬 VISUALIZE_PANEL ENDPOINT HIT!")
    print("="*70)
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        manifest_path = data.get('manifest_path')
        dataset_key = data.get('dataset', DEFAULT_DATASET_KEY)
        compare_datasets = data.get('compare_datasets', False)
        print(f"Manifest path: {manifest_path}")
        print(f"Dataset key: {dataset_key}")
        print(f"Compare datasets: {compare_datasets}")
        
        if not manifest_path:
            return jsonify({"error": "No manifest path provided"}), 400
        
        if dataset_key not in DATASETS:
            return jsonify({"error": f"Invalid dataset: {dataset_key}"}), 400
        
        # Get page data from Zarr
        page_data = get_page_data_from_zarr(manifest_path, dataset_key)
        if page_data is None:
            return jsonify({"error": "Could not extract page data from Zarr"}), 404
        
        if compare_datasets and 'fused' in DATASETS and 'vision' in DATASETS:
            # Generate comparison visualization
            page_data_dict = {
                'fused': get_page_data_from_zarr(manifest_path, 'fused'),
                'vision': get_page_data_from_zarr(manifest_path, 'vision')
            }
            if None in page_data_dict.values():
                return jsonify({"error": "Could not extract page data for comparison"}), 404
            
            img_base64 = compare_datasets_visualization(page_data_dict, manifest_path)
        else:
            # Generate single dataset visualization
            img_base64 = create_panel_visualization(page_data, manifest_path, dataset_key)
        
        if img_base64 is None:
            return jsonify({"error": "Could not generate visualization"}), 500
        
        return jsonify({
            "visualization": img_base64,
            "page_data": {
                'num_panels': int(page_data['num_panels']),
                'dataset': EMBEDDING_CONFIGS[dataset_key]['name']
            }
        })
        
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_ROOT, filename)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Panel Visualization Server Ready!")
    print("="*70)
    print(f"📊 Loaded {len(DATASETS)} embedding dataset(s)")
    print(f"🔧 Using model from: {CHECKPOINT_PATH}")
    print(f"📁 Image root: {IMAGE_ROOT}")
    print("\n🌐 Access the interface at: http://127.0.0.1:5000")
    print("\n📍 Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule} {list(rule.methods)}")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
