"""
Query interface for Zarr-based comic embeddings
Supports similarity search, filtering, and analysis
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import numbers

def load_zarr_dataset(zarr_path: str) -> xr.Dataset:
    """Load Zarr dataset"""
    ds = xr.open_zarr(zarr_path)
    return ds


def _bytes_to_str(v):
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode('utf-8')
        except Exception:
            return v.decode('latin-1', errors='ignore')
    return v


def _make_json_serializable(obj):
    """Recursively convert numpy arrays/scalars and other non-JSON types to Python native types."""
    # Lazy import to avoid hard dependency in some contexts
    try:
        import numpy as _np
    except Exception:
        _np = None

    # Handle numpy arrays
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()

    # Numpy scalar types (int64, float32, bool_ etc.)
    if _np is not None and isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
        return obj.item()

    # Python numeric types
    if isinstance(obj, numbers.Number):
        return obj

    # Bytes -> str
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode('utf-8')
        except Exception:
            return obj.decode('latin-1', errors='ignore')

    # Dict -> convert values
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    # List/tuple -> convert elements
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]

    # Fallback: try to convert numpy scalar via .item()
    try:
        if hasattr(obj, 'item') and not isinstance(obj, str):
            return obj.item()
    except Exception:
        pass

    return obj

def cosine_similarity_search(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Find most similar embeddings using cosine similarity"""
    
    # Normalize embeddings
    query_norm = F.normalize(torch.from_numpy(query_embedding).unsqueeze(0), p=2, dim=1)
    embeddings_norm = F.normalize(torch.from_numpy(embeddings), p=2, dim=1)
    
    # Calculate similarities
    similarities = torch.mm(query_norm, embeddings_norm.t()).squeeze(0)
    
    # Get top-k indices and values
    top_values, top_indices = torch.topk(similarities, top_k)
    
    return top_indices.numpy(), top_values.numpy()

def find_similar_pages(ds: xr.Dataset, query_page_id: str, top_k: int = 10, 
                      use_panel_embeddings: bool = False, panel_idx: int = 0) -> Dict:
    """Find pages similar to a query page"""
    
    # Get query embedding
    if use_panel_embeddings:
        query_embedding = ds['panel_embeddings'].sel(page_id=query_page_id, panel_id=panel_idx).values
    else:
        query_embedding = ds['page_embeddings'].sel(page_id=query_page_id).values
    
    # Get all embeddings
    if use_panel_embeddings:
        all_embeddings = ds['panel_embeddings'].sel(panel_id=panel_idx).values
    else:
        all_embeddings = ds['page_embeddings'].values
    
    # Find similar pages
    similar_indices, similarities = cosine_similarity_search(query_embedding, all_embeddings, top_k)
    
    # Get page IDs
    similar_page_ids = [ _bytes_to_str(ds['page_id'].values[int(i)]) for i in similar_indices ]
    
    # Create results
    results = []
    for i, (page_id, similarity) in enumerate(zip(similar_page_ids, similarities)):
        page_info = {
            'rank': i + 1,
            'page_id': page_id,
            'similarity': float(similarity),
            'source': ds['source'].sel(page_id=page_id).values,
            'series': ds['series'].sel(page_id=page_id).values,
            'volume': ds['volume'].sel(page_id=page_id).values,
            'issue': ds['issue'].sel(page_id=page_id).values,
            'page_num': ds['page_num'].sel(page_id=page_id).values
        }
        results.append(page_info)
    
    return {
        'query_page_id': query_page_id,
        'query_type': 'panel' if use_panel_embeddings else 'page',
        'panel_idx': panel_idx if use_panel_embeddings else None,
        'results': results
    }

def find_similar_covers(ds: xr.Dataset, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
    """Find covers similar to a query embedding"""
    
    # For now, assume single-panel pages are covers
    # In the future, we could add a 'page_type' coordinate
    single_panel_pages = ds.sel(panel_id=0).where(ds['panel_mask'].sel(panel_id=0) == True)
    
    # Get page embeddings for single-panel pages
    cover_embeddings = single_panel_pages['page_embeddings'].values
    cover_page_ids = single_panel_pages['page_id'].values
    
    # Remove NaN values
    valid_mask = ~np.isnan(cover_embeddings).any(axis=1)
    cover_embeddings = cover_embeddings[valid_mask]
    cover_page_ids = cover_page_ids[valid_mask]
    
    # Find similar covers
    similar_indices, similarities = cosine_similarity_search(query_embedding, cover_embeddings, top_k)
    
    # Get page IDs
    similar_page_ids = cover_page_ids[similar_indices]
    
    # Create results
    results = []
    for i, (page_id, similarity) in enumerate(zip(similar_page_ids, similarities)):
        page_info = {
            'rank': i + 1,
            'page_id': page_id,
            'similarity': float(similarity),
            'source': ds['source'].sel(page_id=page_id).values,
            'series': ds['series'].sel(page_id=page_id).values,
            'volume': ds['volume'].sel(page_id=page_id).values,
            'issue': ds['issue'].sel(page_id=page_id).values,
            'page_num': ds['page_num'].sel(page_id=page_id).values
        }
        results.append(page_info)
    
    return {
        'query_type': 'cover',
        'results': results
    }

def filter_pages(ds: xr.Dataset, filters: Dict[str, Union[str, List[str]]]) -> xr.Dataset:
    """Filter pages based on criteria"""
    
    filtered_ds = ds
    
    for key, value in filters.items():
        if key == 'source':
            if isinstance(value, str):
                filtered_ds = filtered_ds.where(filtered_ds['source'] == value, drop=True)
            else:
                filtered_ds = filtered_ds.where(filtered_ds['source'].isin(value), drop=True)
        elif key == 'series':
            if isinstance(value, str):
                filtered_ds = filtered_ds.where(filtered_ds['series'] == value, drop=True)
            else:
                filtered_ds = filtered_ds.where(filtered_ds['series'].isin(value), drop=True)
        elif key == 'volume':
            if isinstance(value, str):
                filtered_ds = filtered_ds.where(filtered_ds['volume'] == value, drop=True)
            else:
                filtered_ds = filtered_ds.where(filtered_ds['volume'].isin(value), drop=True)
        elif key == 'issue':
            if isinstance(value, str):
                filtered_ds = filtered_ds.where(filtered_ds['issue'] == value, drop=True)
            else:
                filtered_ds = filtered_ds.where(filtered_ds['issue'].isin(value), drop=True)
    
    return filtered_ds

def analyze_series(ds: xr.Dataset, series_name: str) -> Dict:
    """Analyze a specific comic series"""
    
    # Filter to series
    series_ds = ds.sel(series=series_name)
    
    if len(series_ds.page_id) == 0:
        return {'error': f'Series "{series_name}" not found'}
    
    # Get basic statistics
    total_pages = len(series_ds.page_id)
    sources = series_ds['source'].values
    volumes = series_ds['volume'].values
    issues = series_ds['issue'].values
    
    # Count by source
    amazon_count = np.sum(sources == 'amazon')
    calibre_count = np.sum(sources == 'calibre')
    
    # Count by volume
    unique_volumes = np.unique(volumes)
    volume_counts = {vol: np.sum(volumes == vol) for vol in unique_volumes}
    
    # Count by issue
    unique_issues = np.unique(issues)
    issue_counts = {iss: np.sum(issues == iss) for iss in unique_issues}
    
    return {
        'series_name': series_name,
        'total_pages': int(total_pages),
        'amazon_pages': int(amazon_count),
        'calibre_pages': int(calibre_count),
        'volumes': volume_counts,
        'issues': issue_counts,
        'unique_volumes': len(unique_volumes),
        'unique_issues': len(unique_issues)
    }

def find_pages_by_text(ds: xr.Dataset, search_text: str, case_sensitive: bool = False) -> List[Dict]:
    """Find pages containing specific text"""
    results = []

    def _text_to_str(text_v):
        try:
            if isinstance(text_v, np.ndarray):
                if text_v.size == 0:
                    return ''
                elif text_v.size == 1:
                    v = text_v.item()
                    return _bytes_to_str(v) if isinstance(v, (bytes, bytearray)) else str(v)
                else:
                    parts = []
                    for x in text_v.flatten():
                        parts.append(_bytes_to_str(x) if isinstance(x, (bytes, bytearray)) else str(x))
                    return ' '.join([p for p in parts if p])
            else:
                return _bytes_to_str(text_v) if isinstance(text_v, (bytes, bytearray)) else str(text_v)
        except Exception:
            return str(text_v)

    search_text_clean = search_text if case_sensitive else search_text.lower()

    for page_id in ds.page_id.values:
        # Get text content for this page (may be array-like)
        try:
            text_content = ds['text_content'].sel(page_id=page_id).values
        except Exception:
            # Missing text_content for this page
            continue

        # Iterate panels safely
        for panel_idx, raw_text in enumerate(text_content):
            text = _text_to_str(raw_text)
            if not text:
                continue

            text_clean = text if case_sensitive else text.lower()
            if search_text_clean in text_clean:
                page_info = {
                    'page_id': _bytes_to_str(page_id),
                    'panel_idx': int(panel_idx),
                    'text': text,
                    'source': ds['source'].sel(page_id=page_id).values,
                    'series': ds['series'].sel(page_id=page_id).values,
                    'volume': ds['volume'].sel(page_id=page_id).values,
                    'issue': ds['issue'].sel(page_id=page_id).values,
                    'page_num': ds['page_num'].sel(page_id=page_id).values
                }
                results.append(page_info)

    return results


def _normalize_page_id(s: str) -> str:
    """Normalize page_id strings for substring matching: bytes->str, lower, unify slashes."""
    if s is None:
        return ''
    s = _bytes_to_str(s)
    try:
        s = str(s)
    except Exception:
        pass
    return s.strip().lower().replace('\\', '/')


def find_page_ids_by_substr(ds: xr.Dataset, substr: str) -> List[str]:
    """Return a list of page_id strings that contain the substring (case-insensitive)."""
    if substr is None:
        return []
    norm_q = _normalize_page_id(substr)
    matches = []
    for pid in ds.page_id.values:
        pid_str = _bytes_to_str(pid)
        if norm_q in _normalize_page_id(pid_str):
            matches.append(pid_str)
    return matches


def _as_scalar(v, dtype=float, default=None):
    """Coerce a possibly-array value to a Python scalar.

    - If v is numpy array: empty -> default, size==1 -> item(), size>1 -> try first non-nan or mean.
    - If v is bytes -> decode.
    - Otherwise try to cast to dtype.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None

    if v is None:
        return default

    # Bytes
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode('utf-8')
        except Exception:
            return v.decode('latin-1', errors='ignore')

    # Numpy arrays / array-like
    try:
        if _np is not None and isinstance(v, _np.ndarray):
            if v.size == 0:
                return default
            if v.size == 1:
                try:
                    return dtype(v.item())
                except Exception:
                    return default
            # multiple values: prefer first non-nan, otherwise mean
            flat = v.flatten()
            for x in flat:
                try:
                    if not (_np.isnan(x) if _np.issubdtype(type(x), _np.floating) else False):
                        return dtype(x)
                except Exception:
                    try:
                        return dtype(x)
                    except Exception:
                        continue
            # fallback to mean if numeric
            try:
                return dtype(_np.nanmean(flat))
            except Exception:
                try:
                    return dtype(flat[0])
                except Exception:
                    return default
    except Exception:
        pass

    # Generic coercion
    try:
        return dtype(v)
    except Exception:
        try:
            return v
        except Exception:
            return default

def get_page_details(ds: xr.Dataset, page_id: str, include_embeddings: bool = False) -> Dict:
    """Get detailed information about a specific page"""
    
    try:
        page_data = ds.sel(page_id=page_id)
        
        # Get basic info
        info = {
            'page_id': page_id,
            'source': page_data['source'].values,
            'series': page_data['series'].values,
            'volume': page_data['volume'].values,
            'issue': page_data['issue'].values,
            'page_num': page_data['page_num'].values
        }
        
        # Get panel information
        panel_mask = np.asarray(page_data['panel_mask'].values)
        # Squeeze and ensure 1-D boolean mask
        panel_mask = np.squeeze(panel_mask)
        if panel_mask.ndim == 0:
            panel_mask = np.array([bool(panel_mask)])
        panel_mask = panel_mask.astype(bool)

        num_panels = int(np.sum(panel_mask))

        panels = []
        # Use the mask length as the panel count (cap to 12 if necessary)
        max_panels = panel_mask.shape[0] if panel_mask.shape[0] > 0 else 12
        for i in range(max_panels):
            # Coerce the mask element to a single boolean safely
            try:
                mask_el = np.asarray(panel_mask[i])
            except Exception:
                mask_el = panel_mask[i]

            if getattr(mask_el, 'size', None) is None:
                # Not array-like, just coerce to bool
                try:
                    is_present = bool(mask_el)
                except Exception:
                    is_present = False
            else:
                # If array-like: empty -> False, single-element -> its truth, multi -> any()
                if mask_el.size == 0:
                    is_present = False
                elif mask_el.size == 1:
                    try:
                        is_present = bool(mask_el.item())
                    except Exception:
                        is_present = bool(mask_el.flatten()[0])
                else:
                    is_present = bool(np.any(mask_el))

            if not is_present:
                continue

            # Robustly extract text content for the panel
            text_v = page_data['text_content'].sel(panel_id=i).values
            try:
                if isinstance(text_v, np.ndarray):
                    if text_v.size == 0:
                        text_str = ''
                    elif text_v.size == 1:
                        text_str = _bytes_to_str(text_v.item())
                    else:
                        parts = [_bytes_to_str(x) for x in text_v.flatten()]
                        text_str = ' '.join([p for p in parts if p])
                else:
                    text_str = _bytes_to_str(text_v) if isinstance(text_v, (bytes, bytearray)) else str(text_v)
            except Exception:
                text_str = str(text_v)

            att_v = page_data['attention_weights'].sel(panel_id=i).values
            ro_v = page_data['reading_order'].sel(panel_id=i).values
            panel_info = {
                'panel_idx': i,
                'coordinates': page_data['panel_coordinates'].sel(panel_id=i).values.tolist(),
                'text': text_str,
                'attention_weight': _as_scalar(att_v, dtype=float, default=0.0),
                'reading_order': _as_scalar(ro_v, dtype=float, default=0.0)
            }
            panels.append(panel_info)
        
        info['num_panels'] = int(num_panels)
        info['panels'] = panels
        
        # Get embeddings (optional)
        if include_embeddings:
            info['page_embedding'] = page_data['page_embeddings'].values.tolist()
            info['panel_embeddings'] = page_data['panel_embeddings'].values.tolist()
        
        return info
        
    except KeyError:
        return {'error': f'Page "{page_id}" not found'}

def main():
    parser = argparse.ArgumentParser(description='Query Zarr-based comic embeddings')
    parser.add_argument('--zarr_path', type=str, required=True,
                       help='Path to Zarr dataset')
    parser.add_argument('--query_type', type=str, choices=['similar', 'covers', 'filter', 'analyze', 'text', 'details'],
                       help='Type of query to perform')
    parser.add_argument('--query_page_id', type=str, default=None,
                       help='Page ID for similarity search')
    parser.add_argument('--query_page_substr', type=str, default=None,
                       help='Substring to search for in page_id labels (case-insensitive). If multiple matches found, details for all matches are returned by default; use --match_index to select one).')
    parser.add_argument('--match_index', type=int, default=None,
                       help='If multiple substring matches are found, select the match at this 0-based index and return details for it.')
    parser.add_argument('--query_embedding', type=str, default=None,
                       help='Path to query embedding file (for cover search)')
    parser.add_argument('--search_text', type=str, default=None,
                       help='Text to search for')
    parser.add_argument('--series_name', type=str, default=None,
                       help='Series name for analysis')
    parser.add_argument('--filters', type=str, default=None,
                       help='Filters as JSON string (e.g., \'{"source": "amazon", "series": "batman"}\')')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--use_panel_embeddings', action='store_true',
                       help='Use panel embeddings instead of page embeddings')
    parser.add_argument('--panel_idx', type=int, default=0,
                       help='Panel index for panel-based queries')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--include_embeddings', action='store_true',
                       help='Include embeddings in details output (default: False). Use this for debugging; omitting reduces output size.')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.zarr_path}")
    ds = load_zarr_dataset(args.zarr_path)
    print(f"Dataset loaded: {len(ds.page_id)} pages")

    # If a substring lookup was requested, resolve to page_id(s)
    if args.query_page_substr and not args.query_page_id:
        matches = find_page_ids_by_substr(ds, args.query_page_substr)
        if len(matches) == 0:
            print(f"No page_id matches found for substring: {args.query_page_substr}")
            raise SystemExit(1)
        elif len(matches) == 1:
            # Unambiguous: use this page id
            args.query_page_id = matches[0]
            print(f"Substring matched a single page_id: {args.query_page_id}")
        else:
            # Multiple matches
            if args.match_index is not None:
                if args.match_index < 0 or args.match_index >= len(matches):
                    print(f"--match_index {args.match_index} out of range (0..{len(matches)-1})")
                    raise SystemExit(1)
                selected = matches[args.match_index]
                args.query_page_id = selected
                print(f"Selected match index {args.match_index}: {selected}")
            else:
                # By default return details for all matches when query_type is details
                if args.query_type == 'details':
                    results = [get_page_details(ds, pid, include_embeddings=args.include_embeddings) for pid in matches]
                    # Output and return
                    if args.output_file:
                        with open(args.output_file, 'w') as f:
                            json.dump(_make_json_serializable(results), f, indent=2)
                        print(f"Results saved to: {args.output_file}")
                    else:
                        print(json.dumps(_make_json_serializable(results), indent=2))
                    return
                else:
                    # For other query types just return the matching page_id list
                    results = {'matches': matches}
                    if args.output_file:
                        with open(args.output_file, 'w') as f:
                            json.dump(_make_json_serializable(results), f, indent=2)
                        print(f"Results saved to: {args.output_file}")
                    else:
                        print(json.dumps(_make_json_serializable(results), indent=2))
                    return
    
    # Perform query
    if args.query_type == 'similar':
        if not args.query_page_id:
            print("Error: --query_page_id required for similarity search")
            return
        
        results = find_similar_pages(
            ds, 
            args.query_page_id, 
            args.top_k, 
            args.use_panel_embeddings, 
            args.panel_idx
        )
        
    elif args.query_type == 'covers':
        if not args.query_embedding:
            print("Error: --query_embedding required for cover search")
            return
        
        query_embedding = np.load(args.query_embedding)
        results = find_similar_covers(ds, query_embedding, args.top_k)
        
    elif args.query_type == 'filter':
        if not args.filters:
            print("Error: --filters required for filtering")
            return
        filters = json.loads(args.filters)
        filtered_ds = filter_pages(ds, filters)
        
        results = {
            'filters': filters,
            'total_pages': len(filtered_ds.page_id),
            'page_ids': filtered_ds.page_id.values.tolist()
        }
        
    elif args.query_type == 'analyze':
        if not args.series_name:
            print("Error: --series_name required for analysis")
            return
        
        results = analyze_series(ds, args.series_name)
        
    elif args.query_type == 'text':
        if not args.search_text:
            print("Error: --search_text required for text search")
            return
        
        results = find_pages_by_text(ds, args.search_text)
        
    elif args.query_type == 'details':
        if not args.query_page_id:
            print("Error: --query_page_id required for page details")
            return
        
        results = get_page_details(ds, args.query_page_id)
        
    else:
        print("Error: Unknown query type")
        return
    
    # Output results
    if args.output_file:
        # Ensure everything is JSON serializable (convert numpy types / arrays)
        serializable_results = _make_json_serializable(results)
        with open(args.output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    else:
        print(json.dumps(_make_json_serializable(results), indent=2))

if __name__ == "__main__":
    main()
