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
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    print(f"DEBUG: Raw image loaded - Mode: {img.mode}, Size: {img.size}")
    # Print a few pixel values from the raw image to check diversity
    if img.mode == 'RGB':
        pixels = list(img.getdata())
        print(f"DEBUG: Raw image pixel sample (first 5): {pixels[:5]}")
        print(f"DEBUG: Raw image pixel sample (last 5): {pixels[-5:]}")

    # Replicate the batch structure used during Zarr embedding generation
    # For a single image, we treat it as a batch of 1 page with 1 panel
    B = 1 # Batch size
    N = 1 # Number of panels per page


    batch = {
        'images': tf(img).to(device).unsqueeze(0).unsqueeze(0), # (B, N, C, H, W)
        'panel_mask': torch.ones((B, N), dtype=torch.bool).to(device),
    }
    print(f"DEBUG: Transformed image tensor shape: {batch['images'].shape}")
    print(f"DEBUG: Transformed image tensor mean: {batch['images'].mean():.4f}, std: {batch['images'].std():.4f}")
    print(f"DEBUG: Transformed image tensor sample (first 5 elements): {batch['images'].flatten()[:5]}")
    print(f"DEBUG: Transformed image tensor sample (last 5 elements): {batch['images'].flatten()[-5:]}")    

    with torch.no_grad():
        # Flatten images for vision encoder (B*N, C, H, W)
        images_flat = batch['images'].flatten(0,1)

        # 1. Get pure vision embedding for the panel
        V = model.atom.vision(images_flat) # (B*N, D)

        # Manually construct P_flat to force vision dominance, bypassing GatedFusion
        # This is a temporary measure to confirm GatedFusion is the issue.
        P_flat = V
        print(f"DEBUG: P_flat (forced vision) sample (first 5 elements): {P_flat.cpu().numpy().flatten()[:5]}")

        # Now, create a multi-panel context for P, even for a single image query
        max_panels = 12 # As defined in closure_lite_dataset.py
        embedding_dim = P_flat.shape[-1] # D=384

        # Initialize P with zeros for max_panels
        P = torch.zeros((B, max_panels, embedding_dim), device=device)
        # Place the single image's P_flat into the first panel slot
        P[:, 0, :] = P_flat

        # Create a panel mask: True for the first panel, False for the rest
        panel_mask = torch.zeros((B, max_panels), dtype=torch.bool, device=device)
        panel_mask[:, 0] = True

        # 2. Pass through page-level understanding
        E_page, _ = model.han.panels_to_page(P, panel_mask)

    return P_flat.cpu().numpy(), E_page.cpu().numpy()

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
    similarities = torch.mm(query_norm, embeddings_norm.t()).squeeze(0)
    top_values, top_indices = torch.topk(similarities, top_k)
    return top_indices.numpy(), top_values.numpy()

def find_similar_pages(ds: xr.Dataset, query_embedding: np.ndarray, top_k: int = 12, query_manifest_path: str = None) -> List[Dict]:
    all_embeddings = ds['page_embeddings'].values
    # Ensure all_embeddings has the correct shape for page embeddings (num_pages, embedding_dim)
    if all_embeddings.ndim == 3: # This indicates it might be (page_id, panel_id, embedding_dim)
        # If it's 3D, we need to flatten it to (total_panels, embedding_dim) or take page-level average
        # For now, let's assume it's a direct error and it should be 2D (num_pages, embedding_dim)
        # This might indicate a deeper issue in Zarr dataset creation or loading if page_embeddings is 3D
        raise ValueError("ds['page_embeddings'].values returned a 3D array. Expected 2D (num_pages, embedding_dim) for page embeddings.")
    elif all_embeddings.ndim != 2:
        raise ValueError(f"ds['page_embeddings'].values returned {all_embeddings.ndim}D array. Expected 2D (num_pages, embedding_dim).")

    similar_indices, similarities = cosine_similarity_search(query_embedding, all_embeddings, top_k)
    
    # Debug: if query_manifest_path provided, find its index and ensure it's included in results
    query_idx = None
    if query_manifest_path is not None:
        manifest_vals = ds['manifest_path'].values
        import os as _os
        
        # Try exact match first
        try:
            manifest_list = manifest_vals.tolist() if hasattr(manifest_vals, 'tolist') else list(manifest_vals)
            query_idx = manifest_list.index(query_manifest_path)
            print(f"DEBUG: Found query page by exact match at idx={query_idx}, manifest='{ds['manifest_path'].values[query_idx]}'")
        except (ValueError, AttributeError):
            pass
        
        # Try normalized path match
        if query_idx is None:
            key = _os.path.normcase(_os.path.normpath(query_manifest_path)).lower()
            for i, mv in enumerate(manifest_vals):
                try:
                    mv_str = str(mv) if not isinstance(mv, str) else mv
                    if isinstance(mv_str, (bytes, bytearray)):
                        mv_str = mv_str.decode('utf-8', errors='ignore')
                    mv_key = _os.path.normcase(_os.path.normpath(mv_str)).lower()
                    if mv_key == key:
                        query_idx = i
                        print(f"DEBUG: Found query page by normalized match at idx={query_idx}, manifest='{ds['manifest_path'].values[query_idx]}'")
                        break
                except Exception:
                    continue
        
        # Try suffix matching (last 3 path components) if normalized match failed
        if query_idx is None:
            target_parts = _os.path.normpath(query_manifest_path).split(_os.sep)
            target_signature = tuple(target_parts[-3:]) if len(target_parts) >= 3 else tuple(target_parts)
            target_signature_lower = tuple(p.lower() for p in target_signature)
            
            for i, mv in enumerate(manifest_vals):
                try:
                    mv_str = str(mv) if not isinstance(mv, str) else mv
                    if isinstance(mv_str, (bytes, bytearray)):
                        mv_str = mv_str.decode('utf-8', errors='ignore')
                    mv_parts = _os.path.normpath(mv_str).split(_os.sep)
                    mv_signature = tuple(mv_parts[-3:]) if len(mv_parts) >= 3 else tuple(mv_parts)
                    mv_signature_lower = tuple(p.lower() for p in mv_signature)
                    
                    if target_signature_lower == mv_signature_lower:
                        query_idx = i
                        print(f"DEBUG: Found query page by suffix match at idx={query_idx}, target_sig={target_signature}, zarr_manifest='{ds['manifest_path'].values[query_idx]}'")
                        break
                except Exception:
                    continue
        
        if query_idx is None:
            print(f"WARNING: Could not find query_manifest_path '{query_manifest_path}' in dataset")
        
        if query_idx is not None:
            # Check if query_idx is in the top results
            if query_idx not in similar_indices:
                print(f"DEBUG: Query page (idx={query_idx}) not in top {top_k} results, adding it as rank 1")
                # Calculate actual similarity for the query page
                query_sim = float(cosine_similarity_search(query_embedding, all_embeddings[query_idx:query_idx+1], 1)[1][0])
                print(f"DEBUG: Query page similarity: {query_sim:.6f}")
                # Insert query page as first result, shift others down
                similar_indices = np.concatenate([[query_idx], similar_indices[:-1]])
                similarities = np.concatenate([[query_sim], similarities[:-1]])
            else:
                result_pos = np.where(similar_indices == query_idx)[0][0]
                print(f"DEBUG: Query page found at rank {result_pos + 1} with similarity {similarities[result_pos]:.6f}")
                # If not already at position 0, move it there
                if result_pos != 0:
                    print(f"DEBUG: Moving query page from rank {result_pos + 1} to rank 1")
                    # Save the similarity value before deleting
                    removed_sim = similarities[result_pos]
                    # Remove from current position
                    similar_indices = np.delete(similar_indices, result_pos)
                    similarities = np.delete(similarities, result_pos)
                    # Insert at front with the correct similarity
                    similar_indices = np.concatenate([[query_idx], similar_indices])
                    similarities = np.concatenate([[removed_sim], similarities])
    
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
    """Retrieves a pre-calculated page embedding from the Zarr dataset by its page_id (which is a manifest_path)."""
    return get_zarr_embedding_by_manifest_path(ds, page_id)

def get_zarr_embedding_by_manifest_path(ds: xr.Dataset, manifest_path: str) -> np.ndarray:
    """Retrieves a pre-calculated page embedding from the Zarr dataset by its manifest_path.
    This function uses the same robust lookup logic as the Zarr generation script.
    
    CRITICAL: The Zarr generation script stores manifest paths in a normalized, lowercased form:
    os.path.normcase(os.path.normpath(str(json_path))).lower()
    
    We must apply the same normalization here for lookups to succeed.
    """
    try:
        target_manifest_path = manifest_path

        manifest_vals = ds['manifest_path'].values.tolist()
        idx = None
        
        # First try: exact match (for backwards compatibility)
        try:
            idx = manifest_vals.index(target_manifest_path)
            print(f"DEBUG: Found exact match for '{target_manifest_path}' at idx={idx}")
        except ValueError:
            pass
        
        # Second try: normalized path match - CRITICAL: must lowercase like generation script does
        if idx is None:
            import os as _os
            # CRITICAL: Apply same normalization as generate_embeddings_zarr_claude.py line 687:
            # key_norm = os.path.normcase(os.path.normpath(str(json_path))).lower()
            key = _os.path.normcase(_os.path.normpath(target_manifest_path)).lower()
            for i, mv in enumerate(manifest_vals):
                try:
                    mv_str = str(mv) if not isinstance(mv, str) else mv
                    # Decode bytes if necessary
                    if isinstance(mv_str, (bytes, bytearray)):
                        mv_str = mv_str.decode('utf-8', errors='ignore')
                    # Normalize and lowercase for comparison
                    mv_key = _os.path.normcase(_os.path.normpath(mv_str)).lower()
                    if mv_key == key:
                        idx = i
                        print(f"DEBUG: Found normalized match for '{target_manifest_path}' at idx={idx}")
                        break
                except Exception:
                    continue
        
        # Third try: case-insensitive match on original path (for when original casing is preserved)
        if idx is None:
            target_lower = target_manifest_path.lower()
            for i, mv in enumerate(manifest_vals):
                try:
                    mv_str = str(mv) if not isinstance(mv, str) else mv
                    if isinstance(mv_str, (bytes, bytearray)):
                        mv_str = mv_str.decode('utf-8', errors='ignore')
                    if mv_str.lower() == target_lower:
                        idx = i
                        print(f"DEBUG: Found case-insensitive match for '{target_manifest_path}' at idx={idx}")
                        break
                except Exception:
                    continue
        
        # Fourth try: suffix matching (for when directory structure differs)
        # This handles cases where the same JSON filename exists in different directory structures
        # e.g., "Humble Comics Bundle.../sheena_queenofthejungle_vol2/..." vs "sheena queenofthejungle vol2 - Unknown/..."
        # CRITICAL: Use at least 3 path components to avoid false matches on generic filenames
        if idx is None:
            import os as _os
            # Extract the significant path components (last 3 parts: grandparent/parent/filename)
            target_parts = _os.path.normpath(target_manifest_path).split(_os.sep)
            # Get last 3 parts to be more specific and avoid false matches
            target_signature = tuple(target_parts[-3:]) if len(target_parts) >= 3 else tuple(target_parts)
            target_signature_lower = tuple(p.lower() for p in target_signature)
            
            for i, mv in enumerate(manifest_vals):
                try:
                    mv_str = str(mv) if not isinstance(mv, str) else mv
                    if isinstance(mv_str, (bytes, bytearray)):
                        mv_str = mv_str.decode('utf-8', errors='ignore')
                    mv_parts = _os.path.normpath(mv_str).split(_os.sep)
                    mv_signature = tuple(mv_parts[-3:]) if len(mv_parts) >= 3 else tuple(mv_parts)
                    mv_signature_lower = tuple(p.lower() for p in mv_signature)
                    
                    # Compare signatures (case-insensitive exact match on last 3 components)
                    if target_signature_lower == mv_signature_lower:
                        idx = i
                        print(f"DEBUG: Found match using suffix matching (3 components) - target: {target_signature}, zarr: {mv_signature}, idx={idx}")
                        break
                except Exception:
                    continue
        
        if idx is None:
            # Debug: print what we have vs what we're looking for
            import os as _os
            print(f"DEBUG: Failed to find '{target_manifest_path}'")
            print(f"DEBUG: Normalized target: '{_os.path.normcase(_os.path.normpath(target_manifest_path)).lower()}'")
            print(f"DEBUG: Target lowercase: '{target_manifest_path.lower()}'")
            target_parts = _os.path.normpath(target_manifest_path).split(_os.sep)
            target_signature = tuple(target_parts[-3:]) if len(target_parts) >= 3 else tuple(target_parts)
            print(f"DEBUG: Target signature (last 3 parts): {target_signature}")
            print(f"DEBUG: Sample manifest_paths in Zarr (first 5):")
            for i in range(min(5, len(manifest_vals))):
                mv_str = str(manifest_vals[i]) if not isinstance(manifest_vals[i], str) else manifest_vals[i]
                if isinstance(mv_str, (bytes, bytearray)):
                    mv_str = mv_str.decode('utf-8', errors='ignore')
                print(f"  [{i}]: '{mv_str}'")
            
            # Also search for partial matches to help diagnose
            print(f"DEBUG: Searching for any paths containing the filename '{_os.path.basename(target_manifest_path)}':")
            found_count = 0
            for i, mv in enumerate(manifest_vals):
                mv_str = str(mv) if not isinstance(mv, str) else mv
                if isinstance(mv_str, (bytes, bytearray)):
                    mv_str = mv_str.decode('utf-8', errors='ignore')
                if _os.path.basename(target_manifest_path).lower() in mv_str.lower():
                    print(f"  [{i}]: '{mv_str}'")
                    found_count += 1
                    if found_count >= 5:
                        break
            
            raise ValueError(f"Manifest path '{target_manifest_path}' not found in Zarr dataset.")
        
        # Retrieve the corresponding page embedding
        page_embedding = ds['page_embeddings'].values[idx]
        print(f"DEBUG: Retrieved embedding for idx={idx}, shape={page_embedding.shape}, L2 norm={np.linalg.norm(page_embedding):.6f}")
        result = np.expand_dims(page_embedding, axis=0)
        print(f"DEBUG: Returning embedding with shape={result.shape}")
        return result
    except Exception as e:
        print(f"Error retrieving embedding for manifest path '{target_manifest_path}': {e}")
        raise
