# search_utils.py

import os
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
import csv
from typing import List, Dict, Tuple

from closure_lite_simple_framework import ClosureLiteSimple
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

# --- Core Data Loading ---

def load_model(checkpoint_path: str, device: torch.device, num_heads: int = 4, temperature: float = 0.1):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ClosureLiteSimple(d=384, num_heads=num_heads, temperature=temperature).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.eval()
    return model

def load_zarr_dataset(zarr_path: str) -> xr.Dataset:
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr dataset not found at: {zarr_path}")
    return xr.open_zarr(zarr_path)

def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Loads a manifest from a file.
    Handles .csv files or .txt files (treating each line as a canonical_id).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
    
    manifest_data = []
    file_ext = os.path.splitext(manifest_path)[1].lower()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        if file_ext == '.csv':
            return list(csv.DictReader(f))
        elif file_ext == '.txt':
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    manifest_data.append({'canonical_id': stripped_line})
            return manifest_data
        else:
            raise TypeError(f"Unsupported manifest file type: {file_ext}")

# --- Type Conversion & JSON Serialization ---

def _bytes_to_str(v):
    if isinstance(v, (bytes, bytearray)):
        try: return v.decode('utf-8')
        except Exception: return v.decode('latin-1', errors='ignore')
    return v

def make_json_serializable(obj):
    if isinstance(obj, (np.ndarray, np.generic)): return obj.tolist()
    if isinstance(obj, (bytes, bytearray)): return _bytes_to_str(obj)
    if isinstance(obj, dict): return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [make_json_serializable(v) for v in obj]
    if hasattr(obj, 'item'): return obj.item()
    return obj

# --- Embedding Generation ---

def get_embedding_for_image(model, image_path, device):
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.ConvertImageDtype(torch.float32)])
    img = Image.open(image_path).convert('RGB')
    batch = {
        'images': tf(img).unsqueeze(0).unsqueeze(0).to(device),
        'input_ids': torch.zeros((1, 1, 128), dtype=torch.long).to(device),
        'attention_mask': torch.zeros((1, 1, 128), dtype=torch.long).to(device),
        'comp_feats': torch.zeros((1, 1, 7), dtype=torch.float32).to(device),
        'panel_mask': torch.ones((1, 1), dtype=torch.bool).to(device),
    }
    with torch.no_grad():
        P_flat = model.atom(images=batch['images'].flatten(0,1), input_ids=batch['input_ids'].flatten(0,1), attention_mask=batch['attention_mask'].flatten(0,1), comp_feats=batch['comp_feats'].flatten(0,1))
        P = P_flat.view(1, 1, -1)
        E_page, _ = model.han.panels_to_page(P, batch['panel_mask'])
    print(f"DEBUG: Image embedding (E_page) shape: {E_page.shape}")
    print(f"DEBUG: Image embedding (E_page) sample (first 5 elements): {E_page.cpu().numpy()[0, 0, :5]}")
    return E_page.cpu().numpy()

def get_embedding_for_text(model, text_query: str, device):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tok = tokenizer(text_query, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
    dummy_image = torch.zeros((1, 3, 224, 224), device=device)
    dummy_comp = torch.zeros((1, 7), device=device)
    with torch.no_grad():
        fused_embedding = model.atom(images=dummy_image, input_ids=tok['input_ids'], attention_mask=tok['attention_mask'], comp_feats=dummy_comp)
    return fused_embedding.cpu().numpy()

# --- Search Functions ---

def cosine_similarity_search(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if query_embedding.ndim == 1: query_embedding = np.expand_dims(query_embedding, axis=0)
    query_norm = F.normalize(torch.from_numpy(query_embedding), p=2, dim=1)
    embeddings_norm = F.normalize(torch.from_numpy(embeddings), p=2, dim=1)

    print(f"DEBUG: cosine_similarity_search - query_norm shape: {query_norm.shape}, embeddings_norm shape: {embeddings_norm.shape}")
    print(f"DEBUG: cosine_similarity_search - query_norm sample (first 5): {query_norm[0, :5]}")
    print(f"DEBUG: cosine_similarity_search - embeddings_norm sample (first row, first 5): {embeddings_norm[0, :5]}")

    similarities = torch.mm(query_norm, embeddings_norm.t()).squeeze(0)

    print(f"DEBUG: cosine_similarity_search - Similarities shape: {similarities.shape}")
    print(f"DEBUG: cosine_similarity_search - Similarities max: {similarities.max()}, min: {similarities.min()}, mean: {similarities.mean()}")

    top_values, top_indices = torch.topk(similarities, top_k)
    return top_indices.numpy(), top_values.numpy()

def find_similar_pages(ds: xr.Dataset, query_embedding: np.ndarray, top_k: int = 12) -> List[Dict]:
    print(f"DEBUG: Type of ds['page_embeddings']: {type(ds['page_embeddings'])}")
    print(f"DEBUG: Shape of ds['page_embeddings']: {ds['page_embeddings'].shape}")
    all_embeddings = ds['page_embeddings'].values
    # Ensure all_embeddings has the correct shape for page embeddings (num_pages, embedding_dim)
    if all_embeddings.ndim == 3: # This indicates it might be (page_id, panel_id, embedding_dim)
        # If it's 3D, we need to flatten it to (total_panels, embedding_dim) or take page-level average
        # For now, let's assume it's a direct error and it should be 2D (num_pages, embedding_dim)
        # This might indicate a deeper issue in Zarr dataset creation or loading if page_embeddings is 3D
        raise ValueError("ds['page_embeddings'].values returned a 3D array. Expected 2D (num_pages, embedding_dim) for page embeddings.")
    elif all_embeddings.ndim != 2:
        raise ValueError(f"ds['page_embeddings'].values returned {all_embeddings.ndim}D array. Expected 2D (num_pages, embedding_dim).")

    print(f"DEBUG: Shape of all_embeddings before passing to cosine_similarity_search: {all_embeddings.shape}")
    similar_indices, similarities = cosine_similarity_search(query_embedding, all_embeddings, top_k)
    similar_indices, similarities = cosine_similarity_search(query_embedding, all_embeddings, top_k)
    results = []
    for i, idx in enumerate(similar_indices):
        results.append({
            'rank': i + 1,
            'page_id': ds['page_id'].values[idx],
            'manifest_path': ds['manifest_path'].values[idx],
            'similarity': similarities[i]
        })
    return make_json_serializable(results)

def find_similar_panels(ds: xr.Dataset, query_embedding: np.ndarray, top_k: int = 12) -> List[Dict]:
    panel_embeddings_3d = ds['panel_embeddings'].values
    panel_mask_3d = ds['panel_mask'].values
    valid_indices_3d = np.argwhere(panel_mask_3d)
    valid_embeddings_2d = panel_embeddings_3d[panel_mask_3d]
    similar_indices_1d, similarities = cosine_similarity_search(query_embedding, valid_embeddings_2d, top_k)
    results = []
    for i, flat_idx in enumerate(similar_indices_1d):
        page_idx, panel_idx = valid_indices_3d[flat_idx]
        results.append({
            'rank': i + 1,
            'page_id': ds['page_id'].values[page_idx],
            'panel_id': int(panel_idx),
            'manifest_path': ds['manifest_path'].values[page_idx],
            'panel_coords': ds['panel_coordinates'].values[page_idx, panel_idx],
            'similarity': similarities[i]
        })
    return make_json_serializable(results)

def keyword_search_panels(ds: xr.Dataset, text_query: str, top_k: int = 50) -> List[Dict]:
    results = []
    query_lower = text_query.lower()
    for page_idx, page_id in enumerate(ds.page_id.values):
        panel_texts = ds['text_content'].values[page_idx]
        panel_mask = ds['panel_mask'].values[page_idx]
        for panel_idx, text in enumerate(panel_texts):
            if not panel_mask[panel_idx]: continue
            text_str = _bytes_to_str(text)
            if query_lower in text_str.lower():
                results.append({
                    'page_id': page_id,
                    'panel_id': int(panel_idx),
                    'manifest_path': ds['manifest_path'].values[page_idx],
                    'panel_coords': ds['panel_coordinates'].values[page_idx, panel_idx],
                    'text_snippet': text_str
                })
                if len(results) >= top_k: return make_json_serializable(results)
    return make_json_serializable(results)

# --- New Multi-Stage Search Functions ---

def page_id_search(manifest_data: List[Dict], query: str, top_k: int = 50) -> List[Dict]:
    """Finds pages in the manifest by matching a query against the canonical_id."""
    results = []
    query_lower = query.lower()
    for record in manifest_data:
        if query_lower in record.get('canonical_id', '').lower():
            results.append(record)
            if len(results) >= top_k:
                break
    return make_json_serializable(results)

def get_embedding_by_page_id(ds: xr.Dataset, page_id: str) -> np.ndarray:
    """Retrieves a pre-calculated page embedding from the Zarr dataset by its page_id."""
    try:
        # The incoming page_id is actually a manifest_path
        target_manifest_path = page_id

        manifest_vals = ds['manifest_path'].values.tolist()
        idx = None
        try:
            idx = manifest_vals.index(target_manifest_path)
        except ValueError:
            # Try normalized path match
            import os as _os
            key = _os.path.normcase(_os.path.normpath(target_manifest_path))
            for i, mv in enumerate(manifest_vals):
                try:
                    if _os.path.normcase(_os.path.normpath(str(mv))) == key:
                        idx = i
                        break
                except Exception:
                    continue
        
        if idx is None:
            raise ValueError(f"Manifest path '{target_manifest_path}' not found in Zarr dataset.")
        
        # Retrieve the corresponding page embedding
        page_embedding = ds['page_embeddings'].values[idx]
        print(f"DEBUG: Retrieved page_embedding shape: {page_embedding.shape}")
        print(f"DEBUG: Retrieved page_embedding sample (first 5 elements): {page_embedding[:5]}")
        return np.expand_dims(page_embedding, axis=0)
    except Exception as e:
        print(f"Error retrieving embedding for manifest path '{target_manifest_path}': {e}")
        raise
