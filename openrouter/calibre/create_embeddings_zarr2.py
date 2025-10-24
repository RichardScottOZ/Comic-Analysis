"""
Generate embeddings for comic datasets and store in Zarr format
Supports both Amazon and CalibreComics datasets with standardized naming
"""

import os
import re
import json
import torch
import numpy as np
import xarray as xr
import zarr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import time
import tempfile

from closure_lite_simple_framework import ClosureLiteSimple
# Use the same dataloader factory the trainer uses so loading behavior matches
try:
    # Local import (script sits next to the training script) — prefer this so
    # python module path issues don't block execution when run from the repo root.
    from train_closure_lite_with_list import create_dataloader_from_list
except Exception:
    # Fall back to package-style import if available in PYTHONPATH
    from benchmarks.detections.openrouter.train_closure_lite_with_list import create_dataloader_from_list

def standardize_path(original_path: str, source: str) -> str:
    """Convert various path formats to standardized naming"""
    
    path_parts = Path(original_path).parts
    
    if source == 'amazon':
        # E:\amazon\Batman The Dark Knight Detective v03\Batman The Dark Knight Detective v03 - p001.jpg
        # -> amazon_batman_dark_knight_detective_v03_001_p001
        
        if len(path_parts) < 2:
            return f"amazon_unknown_unknown_000_p000"
            
        series_dir = path_parts[-2]  # "Batman The Dark Knight Detective v03"
        filename = path_parts[-1]    # "Batman The Dark Knight Detective v03 - p001.jpg"
        
        # Extract series, volume, issue, page
        series = clean_series_name(series_dir)
        volume = extract_volume(series_dir)
        issue = extract_issue(filename)
        page = extract_page(filename)
        
    elif source == 'calibre':
        # E:\CalibreComics\Justice League (2016-2018)\Justice League (2016-2018) 012 - p019.jpg
        # -> calibre_justice_league_2016_012_p019
        
        if len(path_parts) < 2:
            return f"calibre_unknown_unknown_000_p000"
            
        series_dir = path_parts[-2]  # "Justice League (2016-2018)"
        filename = path_parts[-1]    # "Justice League (2016-2018) 012 - p019.jpg"
        
        # Extract series, volume, issue, page
        series = clean_series_name(series_dir)
        volume = extract_volume(series_dir)
        issue = extract_issue(filename)
        page = extract_page(filename)
    
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return f"{source}_{series}_{volume}_{issue}_{page}"

def clean_series_name(name: str) -> str:
    """Clean and standardize series names"""
    # Remove volume information first to avoid duplication
    name_no_volume = re.sub(r'\s+(v\d+|\(\d{4}-\d{4}\)|\d{4})', '', name)
    
    # Remove special characters, normalize spaces
    cleaned = re.sub(r'[^\w\s]', '', name_no_volume)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned.lower()

def extract_volume(name: str) -> str:
    """Extract volume information"""
    # Look for patterns like "v03", "2016-2018", etc.
    # Remove the volume info from the name to avoid duplication
    volume_match = re.search(r'(v\d+|\(\d{4}-\d{4}\)|\d{4})', name)
    return volume_match.group(1) if volume_match else 'unknown'

def extract_issue(filename: str) -> str:
    """Extract issue number"""
    issue_match = re.search(r'(\d{3})', filename)
    return issue_match.group(1) if issue_match else '000'

def extract_page(filename: str) -> str:
    """Extract page number"""
    page_match = re.search(r'p(\d{3})', filename)
    return f"p{page_match.group(1)}" if page_match else 'p000'

def extract_series(page_id: str) -> str:
    """Extract series from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:-3])  # Everything between source and volume
    return 'unknown'

def extract_volume_from_id(page_id: str) -> str:
    """Extract volume from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-3]  # Volume is third from end
    return 'unknown'

def extract_issue_from_id(page_id: str) -> str:
    """Extract issue from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-2]  # Issue is second from end
    return '000'

def extract_page_from_id(page_id: str) -> str:
    """Extract page from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-1]  # Page is last
    return 'p000'

def load_model(checkpoint_path: str, device: torch.device, num_heads: int = 4, temperature: float = 0.1):
    """Load the trained simple model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ClosureLiteSimple(d=384, num_heads=num_heads, temperature=temperature).to(device)
    # Allow loading checkpoints with extra heads (e.g., context/denoise) by ignoring unexpected keys
    incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # Log any missing or unexpected keys for visibility
    if getattr(incompatible, 'missing_keys', None):
        print(f"[load_model] Missing keys when loading checkpoint: {incompatible.missing_keys}")
    if getattr(incompatible, 'unexpected_keys', None):
        print(f"[load_model] Ignored unexpected keys from checkpoint: {incompatible.unexpected_keys}")
    model.eval()
    return model

def process_batch(model, batch, device):
    """Process a batch and extract embeddings"""
    with torch.no_grad():
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get model outputs
        B, N, _, _, _ = batch['images'].shape
        images = batch['images'].flatten(0, 1)
        input_ids = batch['input_ids'].flatten(0, 1)
        attention_mask = batch['attention_mask'].flatten(0, 1)
        comp_feats = batch['comp_feats'].flatten(0, 1)
        
        # 1. Panel Analysis (raw embeddings)
        P_flat = model.atom(images, input_ids, attention_mask, comp_feats)
        P = P_flat.view(B, N, -1)
        
        # 2. Page-level Understanding
        E_page, attention_weights = model.han.panels_to_page(P, batch['panel_mask'])
        
        # 3. Reading Order Prediction
        logits_neighbors = model.next_head(P)
        
        return {
            'panel_embeddings': P.cpu().numpy(),
            'page_embeddings': E_page.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy(),
            'reading_order': logits_neighbors.cpu().numpy(),
            'panel_mask': batch['panel_mask'].cpu().numpy(),
            'next_idx': batch['next_idx'].cpu().numpy(),
            'original_pages': batch['original_page'],
            'json_files': batch['json_file'],
            'json_paths': batch.get('json_path', [])
        }

def create_zarr_dataset(output_path: str, amazon_data: List[Dict], calibre_data: List[Dict] = None):
    """Create Zarr dataset with standardized coordinates"""
    
    print("Creating standardized coordinates...")
    
    # Combine all page IDs
    # Use normalized full manifest path as the canonical page_id to avoid
    # lossy collisions (training uses exact paths). We still compute a
    # standardized id for series/volume extraction for human-readable coords.
    all_page_ids = []  # canonical normalized manifest paths (os.path.normcase(normpath))
    all_sources = []
    all_series = []
    all_volumes = []
    all_issues = []
    all_pages = []
    all_manifest_paths = []  # original raw manifest paths (as provided)
    
    # Process Amazon data (each entry may carry an inferred 'source')
    for page_data in amazon_data:
        src = page_data.get('source', 'amazon')
        raw_path = str(page_data.get('path'))
        # Canonical page id: normalized full manifest path
        norm_path = os.path.normcase(os.path.normpath(raw_path))
        all_page_ids.append(norm_path)
        all_manifest_paths.append(raw_path)
        all_sources.append(src)
        # For human-readable series/volume extraction, keep using standardized id
        std_id = standardize_path(raw_path, src)
        all_series.append(extract_series(std_id))
        all_volumes.append(extract_volume_from_id(std_id))
        all_issues.append(extract_issue_from_id(std_id))
        all_pages.append(extract_page_from_id(std_id))
    
    # Process CalibreComics data if provided
    if calibre_data:
        for page_data in calibre_data:
            raw_path = str(page_data.get('path'))
            norm_path = os.path.normcase(os.path.normpath(raw_path))
            all_page_ids.append(norm_path)
            all_manifest_paths.append(raw_path)
            all_sources.append('calibre')
            std_id = standardize_path(raw_path, 'calibre')
            all_series.append(extract_series(std_id))
            all_volumes.append(extract_volume_from_id(std_id))
            all_issues.append(extract_issue_from_id(std_id))
            all_pages.append(extract_page_from_id(std_id))
    
    print(f"Total pages: {len(all_page_ids)}")
    print(f"Amazon pages: {len(amazon_data)}")
    if calibre_data:
        print(f"CalibreComics pages: {len(calibre_data)}")
    
    # Create coordinates
    coords = {
        'page_id': all_page_ids,
        'manifest_path': ('page_id', all_manifest_paths),
        'source': ('page_id', all_sources),
        'series': ('page_id', all_series),
        'volume': ('page_id', all_volumes),
        'issue': ('page_id', all_issues),
        'page_num': ('page_id', all_pages),
        'panel_id': range(12),  # Max panels
        'embedding_dim': range(384),
        'coord_dim': range(4)  # x, y, width, height
    }
    
    # Create XArray dataset
    ds = xr.Dataset(coords=coords)
    
    # Add data variables with proper shapes
    n_pages = len(all_page_ids)
    max_panels = 12
    embedding_dim = 384
    
    ds['panel_embeddings'] = (['page_id', 'panel_id', 'embedding_dim'], 
                              np.zeros((n_pages, max_panels, embedding_dim), dtype=np.float32))
    ds['page_embeddings'] = (['page_id', 'embedding_dim'], 
                             np.zeros((n_pages, embedding_dim), dtype=np.float32))
    ds['attention_weights'] = (['page_id', 'panel_id'], 
                               np.zeros((n_pages, max_panels), dtype=np.float32))
    ds['reading_order'] = (['page_id', 'panel_id'], 
                           np.zeros((n_pages, max_panels), dtype=np.float32))
    ds['panel_coordinates'] = (['page_id', 'panel_id', 'coord_dim'], 
                               np.zeros((n_pages, max_panels, 4), dtype=np.float32))
    # Initialize text content as empty strings to avoid mixed-type object arrays
    # (zeros would create ints and later string assignments produce mixed types
    # which break xarray->zarr persistence).
    # Use a fixed-width unicode dtype for text_content to avoid mixed-type
    # object arrays which can break xarray->zarr persistence. 512 chars per
    # panel should be more than enough for OCR/caption snippets; adjust if
    # needed.
    max_text_len = 512
    ds['text_content'] = (['page_id', 'panel_id'], 
                          np.full((n_pages, max_panels), '', dtype=f'<U{max_text_len}'))
    ds['panel_mask'] = (['page_id', 'panel_id'], 
                        np.zeros((n_pages, max_panels), dtype=bool))
    
    # Configure Zarr backend with compression
    encoding = {
        'panel_embeddings': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12, 384)  # Chunk by 1000 pages
        },
        'page_embeddings': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 384)
        },
        'attention_weights': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'reading_order': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'panel_coordinates': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12, 4)
        },
        'text_content': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'panel_mask': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        }
    }
    
    # Add global attributes
    ds.attrs['model_name'] = 'CLOSURE-Lite-Simple'
    ds.attrs['embedding_dim'] = 384
    ds.attrs['max_panels'] = 12
    ds.attrs['created_date'] = str(Path().cwd())
    ds.attrs['amazon_pages'] = len(amazon_data)
    if calibre_data:
        ds.attrs['calibre_pages'] = len(calibre_data)
    
    # Save to Zarr
    print(f"Saving to Zarr: {output_path}")
    ds.to_zarr(output_path, encoding=encoding, mode='w')
    
    return ds


def _acquire_lock(lock_path: str, timeout: int = 60):
    """Simple file-based lock (creates path exclusively)."""
    start = time.time()
    while True:
        try:
            # exclusive creation
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(str(os.getpid()))
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(0.5)


def _release_lock(lock_path: str):
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def load_ledger(ledger_path: str) -> Dict[str, int]:
    ledger = {}
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r', encoding='utf-8') as lf:
                for line in lf:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    # rec expected {key: index}
                    if isinstance(rec, dict):
                        for k, v in rec.items():
                            ledger[k] = int(v)
        except Exception:
            # fallback: try to read as a single JSON dict
            try:
                with open(ledger_path, 'r', encoding='utf-8') as lf:
                    d = json.load(lf)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            ledger[k] = int(v)
            except Exception:
                pass
    return ledger


def save_ledger_atomic(ledger_path: str, ledger: Dict[str, int]):
    tmp = ledger_path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as lf:
            json.dump(ledger, lf, ensure_ascii=False)
        os.replace(tmp, ledger_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def generate_embeddings(model, dataloader, device, output_path: str, amazon_data: List[Dict], calibre_data: List[Dict] = None, args=None):
    """Generate embeddings for all pages and save to Zarr"""
    
    # Determine if we're in incremental append mode
    incremental_mode = args and getattr(args, 'incremental', False)
    ledger_path = None
    if incremental_mode:
        # Expect an existing Zarr at output_path
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Incremental mode requested but Zarr not found at {output_path}")
        print(f"Opening existing Zarr for incremental update: {output_path}")
        # Open existing dataset
        ds = xr.open_zarr(output_path)
        # Ledger path default
        if args and getattr(args, 'ledger_path', None):
            ledger_path = args.ledger_path
        else:
            ledger_path = os.path.join(Path(output_path).parent, 'page_id_ledger.jsonl')
        # Load or build ledger
        ledger = load_ledger(ledger_path)
        if not ledger:
            # Build ledger from existing dataset coordinates
            try:
                manifest_paths = list(ds['manifest_path'].values.tolist())
                for idx, raw in enumerate(manifest_paths):
                    try:
                        key = os.path.normcase(os.path.normpath(str(raw))).lower()
                        ledger[key] = idx
                    except Exception:
                        continue
                save_ledger_atomic(ledger_path, ledger)
                print(f"Built ledger with {len(ledger)} entries at: {ledger_path}")
            except Exception as e:
                print(f"Warning: failed to build ledger from existing Zarr: {e}")
        # Open zarr group for direct writes/resizing
        zgroup = zarr.open_group(output_path, mode='r+')
        original_n = int(zgroup['page_embeddings'].shape[0])
        print(f"Existing pages in Zarr: {original_n}")
    else:
        # Create Zarr dataset
        ds = create_zarr_dataset(output_path, amazon_data, calibre_data)

    
    print("Generating embeddings...")
    
    # Process batches
    page_idx = 0
    
    # Build a fast lookup from the Zarr dataset's stored manifest_path -> list(indices).
    # Keep all indices for a given normalized manifest path so the generator can
    # map multiple training-list entries (if present) to distinct dataset rows.
    manifest_path_to_page_idxs = {}
    try:
        manifest_paths = list(ds['manifest_path'].values.tolist())
        for idx, raw in enumerate(manifest_paths):
            try:
                key = os.path.normcase(os.path.normpath(str(raw))).lower()
                manifest_path_to_page_idxs.setdefault(key, []).append(idx)
            except Exception:
                pass
            # Also map a common variant where analysis paths were replaced with extracted
            try:
                alt = str(raw).replace('_analysis', '_extracted')
                key_alt = os.path.normcase(os.path.normpath(alt)).lower()
                manifest_path_to_page_idxs.setdefault(key_alt, []).append(idx)
            except Exception:
                pass
    except Exception:
        manifest_path_to_page_idxs = {}

    # If incremental mode, prepare helpers to resize underlying zarr arrays
    if incremental_mode:
        # zgroup and original_n were set earlier
        current_size = original_n
        new_assigned = 0
        lock_path = os.path.join(Path(output_path).parent, '.zarr_update_lock')

        def ensure_size(target_n):
            nonlocal current_size
            if target_n <= current_size:
                return
            # Optionally acquire lock while resizing
            if args and getattr(args, 'concurrency_safe', False):
                _acquire_lock(lock_path, timeout=120)
            try:
                # Resize data vars
                var_resize_map = {
                    'panel_embeddings': (target_n, ) + tuple(zgroup['panel_embeddings'].shape[1:]),
                    'page_embeddings': (target_n, ) + tuple(zgroup['page_embeddings'].shape[1:]),
                    'attention_weights': (target_n, ) + tuple(zgroup['attention_weights'].shape[1:]),
                    'reading_order': (target_n, ) + tuple(zgroup['reading_order'].shape[1:]),
                    'panel_coordinates': (target_n, ) + tuple(zgroup['panel_coordinates'].shape[1:]),
                    'text_content': (target_n, ) + tuple(zgroup['text_content'].shape[1:]),
                    'panel_mask': (target_n, ) + tuple(zgroup['panel_mask'].shape[1:])
                }
                for v, newshape in var_resize_map.items():
                    try:
                        zgroup[v].resize(newshape)
                    except Exception as e:
                        print(f"Warning: failed to resize var {v}: {e}")
                # Resize coords
                coord_vars = ['page_id', 'manifest_path', 'source', 'series', 'volume', 'issue', 'page_num']
                for c in coord_vars:
                    try:
                        zgroup[c].resize((target_n,))
                    except Exception:
                        pass
                current_size = target_n
            finally:
                if args and getattr(args, 'concurrency_safe', False):
                    _release_lock(lock_path)

    # Debug containers
    mapping_failures = []
    empty_page_records = []
    # Track indices already written to detect collisions (multiple inputs mapping to same index)
    written_index_map = {}  # page_index -> first json_file
    mapping_collisions = []
    # Audit every input JSON -> resolved index or explicit reason for skip
    audit_records = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Process batch
        results = process_batch(model, batch, device)
        
        batch_size = len(batch['original_page'])
        
        for i in range(batch_size):
            # Get page data
            page_data = results['original_pages'][i]
            json_file = results['json_files'][i]
            json_path = None
            try:
                json_path = results.get('json_paths', [])[i]
            except Exception:
                json_path = None

            # Determine page index using exact normalized path matching only.
            # Policy: do NOT use basename fallbacks or standardize heuristics that
            # can collapse distinct JSONs. Try exact normalized path first, then
            # try an '_analysis' -> '_extracted' variant. If multiple dataset
            # rows exist for the same normalized path, pick the first unwritten
            # index so multiple identical training entries map to distinct rows.
            page_id = None
            page_index = None
            try:
                candidates = []
                if json_path:
                    key = os.path.normcase(os.path.normpath(str(json_path))).lower()
                    candidates = manifest_path_to_page_idxs.get(key, [])
                # Try variant replacement (analysis -> extracted)
                if not candidates and json_path:
                    alt = str(json_path).replace('_analysis', '_extracted')
                    key_alt = os.path.normcase(os.path.normpath(alt)).lower()
                    candidates = manifest_path_to_page_idxs.get(key_alt, [])

                if candidates:
                    # Choose the first candidate index that hasn't been written yet
                    chosen = None
                    for c in candidates:
                        if c not in written_index_map:
                            chosen = c
                            break
                    if chosen is None:
                        # All candidate rows already written; treat as collision/skip
                        first = candidates[0]
                        mapping_collisions.append({
                            'candidate_indices': candidates,
                            'page_index': int(first),
                            'page_id': str(ds['page_id'].values.tolist()[int(first)]) if len(ds['page_id'].values.tolist()) > int(first) else None,
                            'first_json': written_index_map.get(first),
                            'incoming_json': str(json_file)
                        })
                        audit_records.append({
                            'json_file': str(json_file),
                            'json_path': str(json_path),
                            'status': 'skipped_collision',
                            'reason': 'all_candidate_indices_taken',
                            'candidate_indices': candidates
                        })
                        continue
                    else:
                        page_index = chosen
                        try:
                            page_id = ds.page_id.values.tolist()[page_index]
                        except Exception:
                            page_id = None
                else:
                    # No exact mapping found.
                    if incremental_mode:
                        # Assign a new index at the end of the existing store.
                        try:
                            key_norm = os.path.normcase(os.path.normpath(str(json_path))).lower() if json_path else None
                        except Exception:
                            key_norm = None
                        if key_norm is None:
                            mapping_failures.append({
                                'json_file': str(json_file),
                                'json_path': str(json_path)
                            })
                            audit_records.append({
                                'json_file': str(json_file),
                                'json_path': str(json_path),
                                'status': 'skipped_no_mapping',
                                'reason': 'no_exact_match'
                            })
                            print(f"Warning: could not determine page_id for JSON '{json_path or json_file}', skipping (no exact match)...")
                            continue

                        new_idx = current_size + new_assigned
                        # Prepare zarr arrays to accommodate the new index
                        ensure_size(new_idx + 1)
                        # Update in-memory maps
                        manifest_path_to_page_idxs.setdefault(key_norm, []).append(new_idx)
                        ledger[key_norm] = new_idx
                        new_assigned += 1
                        # Persist ledger periodically
                        try:
                            if new_assigned % 100 == 0:
                                save_ledger_atomic(ledger_path, ledger)
                        except Exception:
                            pass

                        page_index = new_idx
                        page_id = key_norm
                        # We'll write directly to zarr arrays below (incremental path)
                        # record audit as would_write for now; final status updated after write
                        audit_records.append({
                            'json_file': str(json_file),
                            'json_path': str(json_path),
                            'status': 'assigned_new_index',
                            'page_index': int(page_index),
                            'page_id': str(page_id)
                        })
                    else:
                        # Non-incremental: record diagnostic info if requested
                        if args and getattr(args, 'debug_manifest', False):
                            attempts = []
                            try:
                                if json_path:
                                    attempts.append({'normalized': os.path.normcase(os.path.normpath(str(json_path))).lower()})
                                    attempts.append({'alt': os.path.normcase(os.path.normpath(str(json_path).replace('_analysis', '_extracted'))).lower()})
                            except Exception:
                                pass
                            mapping_failures.append({
                                'json_file': str(json_file),
                                'json_path': str(json_path),
                                'attempts': attempts
                            })
                        audit_records.append({
                            'json_file': str(json_file),
                            'json_path': str(json_path),
                            'status': 'skipped_no_mapping',
                            'reason': 'no_exact_match'
                        })
                        print(f"Warning: could not determine page_id for JSON '{json_path or json_file}', skipping (no exact match)...")
                        continue
            except Exception:
                page_index = None
            
            # Extract data
            panel_embeddings = results['panel_embeddings'][i]  # Shape: (12, 384)
            page_embedding = results['page_embeddings'][i]     # Shape: (384,)
            attention_weights = results['attention_weights'][i] # Shape: (12,)
            reading_order = results['reading_order'][i]        # Shape: (12,)
            panel_mask = results['panel_mask'][i]              # Shape: (12,)
            next_idx = results['next_idx'][i]                  # Shape: (12,)
            
            # Extract panel coordinates and text. Prefer normalized panels attached
            # by the dataset under '_normalized_panels', but fall back to any
            # existing 'panels' key for compatibility.
            panel_coords = np.zeros((12, 4), dtype=np.float32)
            # Use matching unicode dtype as the dataset to avoid mixed types
            text_content = np.full((12,), '', dtype=f'<U{512}')
            panels_list = []
            if isinstance(page_data, dict):
                panels_list = page_data.get('_normalized_panels') or page_data.get('panels') or []
            else:
                # In case page_data is some other structure, try to be robust
                try:
                    panels_list = list(page_data.get('panels', []))
                except Exception:
                    panels_list = []

            for j, panel in enumerate(panels_list):
                if j < 12:  # Max panels
                    # Defensive access in case normalization missed fields
                    coords = panel.get('panel_coords') if isinstance(panel, dict) else None
                    if coords is None:
                        continue
                    panel_coords[j] = coords
                    text_content[j] = str(panel.get('text', '')) if isinstance(panel, dict) else ''
            
            # Normalize shapes and reduce any logits/matrices into 1-D indices where needed
            try:
                # attention_weights: if matrix -> reduce to diag or mean across last axis
                aw = np.array(attention_weights)
                if aw.ndim == 2 and aw.shape[0] == aw.shape[1]:
                    # square matrix: take row-wise max as proxy
                    aw_vec = np.argmax(aw, axis=1).astype(np.float32)
                else:
                    aw_vec = aw.astype(np.float32)
            except Exception:
                aw_vec = np.zeros((12,), dtype=np.float32)

            try:
                ro = np.array(reading_order)
                if ro.ndim == 2 and ro.shape[0] == ro.shape[1]:
                    # logits or adjacency matrix -> choose argmax per row as next-panel index
                    ro_vec = np.argmax(ro, axis=1).astype(np.int64)
                elif ro.ndim == 1:
                    ro_vec = ro.astype(np.int64)
                else:
                    # Fallback: flatten and take argmax per block of size 12
                    ro_flat = ro.flatten()
                    if ro_flat.size >= 12:
                        ro_vec = ro_flat[:12].astype(np.int64)
                    else:
                        ro_vec = np.zeros((12,), dtype=np.int64)
            except Exception:
                ro_vec = np.zeros((12,), dtype=np.int64)

            # First-writer-wins policy: if this page_index was already written by
            # a different JSON, skip the write to avoid overwriting valid data.
            jf_str = str(json_file)
            if page_index in written_index_map:
                first = written_index_map[page_index]
                if first == jf_str:
                    # Redundant write from the same JSON — skip to avoid duplicate work
                    if args and getattr(args, 'verbose', False):
                        print(f"[SKIP] redundant write for idx={page_index} json={jf_str}")
                    # audit
                    audit_records.append({
                        'json_file': jf_str,
                        'json_path': str(json_path),
                        'status': 'skipped_redundant',
                        'reason': 'redundant_same_json',
                        'page_index': int(page_index)
                    })
                    continue
                else:
                    # Collision from a different JSON — record and skip overwrite
                    mapping_collisions.append({
                        'page_index': int(page_index),
                        'page_id': page_id,
                        'first_json': first,
                        'second_json': jf_str
                    })
                    if args and getattr(args, 'verbose', False):
                        print(f"[COLLISION] idx={page_index} first={first} second={jf_str} (skipping overwrite)")
                    # Do not overwrite the existing row
                    audit_records.append({
                        'json_file': jf_str,
                        'json_path': str(json_path),
                        'status': 'skipped_collision',
                        'reason': 'first_writer_wins',
                        'page_index': int(page_index),
                        'first_json': first
                    })
                    continue
            else:
                # Reserve this index for the current JSON and write
                written_index_map[page_index] = jf_str

            # Store in dataset (handle incremental direct zarr writes vs in-memory ds)
            jf_str = str(json_file)
            if incremental_mode:
                try:
                    # Infer simple source and human-readable ids
                    try:
                        src = 'calibre' if 'calibre' in (str(json_path or json_file).lower()) else 'amazon'
                    except Exception:
                        src = 'amazon'
                    try:
                        std_id = standardize_path(json_path or json_file, src)
                        series = extract_series(std_id)
                        vol = extract_volume_from_id(std_id)
                        issue = extract_issue_from_id(std_id)
                        page_num = extract_page_from_id(std_id)
                    except Exception:
                        series = ''
                        vol = ''
                        issue = '000'
                        page_num = 'p000'

                    # Write numeric arrays directly into zarr arrays
                    zgroup['panel_embeddings'][page_index, :panel_embeddings.shape[0], :panel_embeddings.shape[1]] = panel_embeddings
                    zgroup['page_embeddings'][page_index, :page_embedding.shape[0]] = page_embedding
                    zgroup['attention_weights'][page_index, :aw_vec.shape[0]] = aw_vec
                    zgroup['reading_order'][page_index, :ro_vec.shape[0]] = ro_vec
                    zgroup['panel_coordinates'][page_index, :panel_coords.shape[0], :panel_coords.shape[1]] = panel_coords
                    # text_content: ensure correct dtype/shape
                    try:
                        tc_arr = np.asarray(text_content, dtype=zgroup['text_content'].dtype)
                    except Exception:
                        tc_arr = np.asarray(text_content, dtype='U')
                    zgroup['text_content'][page_index, :tc_arr.shape[0]] = tc_arr
                    zgroup['panel_mask'][page_index, :len(panel_mask)] = panel_mask

                    # Update coords
                    try:
                        zgroup['page_id'][page_index] = str(page_id)
                    except Exception:
                        pass
                    try:
                        zgroup['manifest_path'][page_index] = str(json_path or json_file)
                    except Exception:
                        pass
                    try:
                        zgroup['source'][page_index] = src
                    except Exception:
                        pass
                    try:
                        zgroup['series'][page_index] = series
                        zgroup['volume'][page_index] = vol
                        zgroup['issue'][page_index] = issue
                        zgroup['page_num'][page_index] = page_num
                    except Exception:
                        pass

                    # mark written
                    written_index_map[page_index] = jf_str
                    # update ledger for this key if present
                    try:
                        if ledger_path and key_norm:
                            save_ledger_atomic(ledger_path, ledger)
                    except Exception:
                        pass

                    # record audit for successful write
                    audit_records.append({
                        'json_file': jf_str,
                        'json_path': str(json_path),
                        'status': 'written',
                        'page_index': int(page_index),
                        'page_id': str(page_id)
                    })
                except Exception as e:
                    print(f"Warning: incremental write failed for idx={page_index} json={jf_str}: {e}")
                    audit_records.append({
                        'json_file': jf_str,
                        'json_path': str(json_path),
                        'status': 'write_failed',
                        'reason': str(e)
                    })
                    continue
            else:
                ds['panel_embeddings'][page_index] = panel_embeddings
                ds['page_embeddings'][page_index] = page_embedding
                ds['attention_weights'][page_index] = aw_vec
                ds['reading_order'][page_index] = ro_vec
                ds['panel_coordinates'][page_index] = panel_coords
                ds['text_content'][page_index] = text_content
                ds['panel_mask'][page_index] = panel_mask

            # Per-page diagnostics: compute panel_mask sum and embedding L2 norm
            try:
                pm_arr = np.array(panel_mask)
                pm_sum = int(pm_arr.sum())
            except Exception:
                pm_sum = None
            try:
                emb_norm = float(np.linalg.norm(page_embedding))
            except Exception:
                emb_norm = None

            if args and getattr(args, 'verbose', False):
                try:
                    log_csv = os.path.join(args.output_dir, 'per_page_stats.csv')
                    if not os.path.exists(log_csv):
                        with open(log_csv, 'w', encoding='utf-8') as fh:
                            fh.write('json_file,page_id,page_index,panel_mask_sum,embedding_l2\n')
                    with open(log_csv, 'a', encoding='utf-8') as fh:
                        fh.write(f'"{json_file}","{page_id}",{page_index},{pm_sum},{emb_norm}\n')
                except Exception as e:
                    print(f"Warning: failed to write per-page stats: {e}")
                print(f"[PAGE] json={json_file} page_id={page_id} idx={page_index} mask_sum={pm_sum} emb_l2={emb_norm}")

            # record audit for successful write
            audit_records.append({
                'json_file': str(json_file),
                'json_path': str(json_path),
                'status': 'written',
                'page_index': int(page_index),
                'page_id': str(page_id),
                'panel_mask_sum': pm_sum,
                'embedding_l2': emb_norm
            })

            page_idx += 1
    
    print(f"Generated embeddings for {page_idx} pages")
    # Persist the in-memory Dataset to the Zarr store so assignments made
    # during batch processing are flushed to disk. create_zarr_dataset wrote
    # an initial empty Zarr, but xarray Dataset modifications don't always
    # update the on-disk store unless explicitly saved.
    try:
        ds.to_zarr(output_path, mode='w')
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"Warning: failed to persist dataset to Zarr: {e}")
        # Attempt a best-effort recovery: coerce any object-typed data vars to
        # fixed-width unicode and try again. This addresses cases where some
        # variables accidentally contain mixed native types (str/int) which
        # prevent xarray from inferring a single dtype for zarr storage.
        try:
            coerced = []
            for var in list(ds.data_vars):
                try:
                    arr = ds[var].values
                except Exception:
                    arr = None
                if arr is None:
                    continue
                if getattr(arr, 'dtype', None) == object:
                    try:
                        # Convert to unicode with conservative max length
                        flat = np.asarray(arr, dtype='U')
                        ds[var].values = flat
                        coerced.append(var)
                        print(f"[persist-fix] coerced variable '{var}' to unicode")
                    except Exception as e2:
                        print(f"[persist-fix] failed to coerce variable '{var}': {e2}")
            # Also coerce any coords that are object-typed
            coerced_coords = []
            try:
                for c in list(ds.coords):
                    try:
                        carr = ds.coords[c].values
                    except Exception:
                        carr = None
                    if carr is None:
                        continue
                    if getattr(carr, 'dtype', None) == object:
                        try:
                            carr_u = np.asarray(carr, dtype='U')
                            ds.coords[c].values = carr_u
                            coerced_coords.append(c)
                            print(f"[persist-fix] coerced coord '{c}' to unicode")
                        except Exception as ecoord:
                            print(f"[persist-fix] failed to coerce coord '{c}': {ecoord}")
            except Exception:
                pass

            if coerced or coerced_coords:
                print(f"Retrying to_zarr after coercing variables: {coerced}")
                ds.to_zarr(output_path, mode='w')
                print(f"Saved to: {output_path} (after coercion)")
            else:
                print("No object-typed variables found to coerce; final persist failed.")
        except Exception as e3:
            print(f"Error: final persist after coercion failed: {e3}")
            # As a last resort, dump each variable to a compressed .npz file so
            # the user can at least recover the arrays without relying on zarr.
            try:
                alt_dir = Path(output_path).parent / (Path(output_path).stem + '_partial_recover')
                os.makedirs(alt_dir, exist_ok=True)
                for var in ds.data_vars:
                    try:
                        np.savez_compressed(str(alt_dir / f"{var}.npz"), ds[var].values)
                    except Exception as e4:
                        print(f"Failed to dump var {var}: {e4}")
                print(f"Wrote partial dump to {alt_dir}")
            except Exception as e5:
                print(f"Critical: failed to write partial dump: {e5}")
    
    # After writing embeddings, optionally gather diagnostics about empty page rows
    try:
        page_embs = np.array(ds['page_embeddings'].values)
        norms = np.linalg.norm(page_embs, axis=1)
        valid_mask = norms > 1e-6
        valid_count = int(np.sum(valid_mask))
        total_pages = int(page_embs.shape[0]) if page_embs.size else 0
        zero_count = total_pages - valid_count

        if args and getattr(args, 'debug_empty', False):
            # Inspect a sample of empty pages and write details
            empty_indices = np.where(~valid_mask)[0]
            sample_n = min(getattr(args, 'debug_sample_n', 10), len(empty_indices))
            for idx in empty_indices[:sample_n]:
                try:
                    page_id = str(ds['page_id'].values.tolist()[int(idx)])
                except Exception:
                    page_id = None
                try:
                    manifest_path = str(ds['manifest_path'].values.tolist()[int(idx)])
                except Exception:
                    manifest_path = None
                record = {'index': int(idx), 'page_id': page_id, 'manifest_path': manifest_path}
                # Try to load the JSON and inspect panels
                try:
                    if manifest_path and os.path.exists(manifest_path):
                        with open(manifest_path, 'r', encoding='utf-8') as jf:
                            j = json.load(jf)
                        panels = j.get('panels') if isinstance(j, dict) else None
                        record['num_panels_in_json'] = len(panels) if isinstance(panels, list) else None
                        # count panels with coords
                        if isinstance(panels, list):
                            coords_count = 0
                            for p in panels:
                                if isinstance(p, dict) and p.get('panel_coords'):
                                    coords_count += 1
                            record['panels_with_coords'] = coords_count
                    else:
                        record['num_panels_in_json'] = None
                except Exception as e:
                    record['json_load_error'] = str(e)

                # Panel mask from dataset
                try:
                    pm = ds['panel_mask'].values[int(idx)].tolist()
                    record['panel_mask'] = pm
                    record['panel_mask_sum'] = int(np.sum(np.array(pm, dtype=int)))
                except Exception:
                    pass

                empty_page_records.append(record)

            # Save diagnostics to output_dir
            try:
                mf_path = os.path.join(Path(output_path).parent, 'mapping_failures.jsonl')
                with open(mf_path, 'w', encoding='utf-8') as mf:
                    for item in mapping_failures:
                        mf.write(json.dumps(item) + '\n')
                ed_path = os.path.join(Path(output_path).parent, 'empty_pages_debug.jsonl')
                with open(ed_path, 'w', encoding='utf-8') as ef:
                    for item in empty_page_records:
                        ef.write(json.dumps(item) + '\n')

                # Save mapping collisions if any
                collisions_path = os.path.join(Path(output_path).parent, 'mapping_collisions.jsonl')
                if mapping_collisions:
                    with open(collisions_path, 'w', encoding='utf-8') as cf:
                        for item in mapping_collisions:
                            cf.write(json.dumps(item) + '\n')
                    print(f"Wrote debug files: {mf_path}, {ed_path}, {collisions_path}")
                else:
                    print(f"Wrote debug files: {mf_path}, {ed_path}")
            except Exception as e:
                print(f"Warning: failed to write debug files: {e}")

        # Write the audit file mapping each input JSON to its resolved action.
        try:
            audit_path = os.path.join(Path(output_path).parent, 'audit_mapping.jsonl')
            with open(audit_path, 'w', encoding='utf-8') as af:
                for r in audit_records:
                    af.write(json.dumps(r) + '\n')
            print(f"Wrote audit file: {audit_path}")
        except Exception as e:
            print(f"Warning: failed to write audit file: {e}")

        return ds
    except Exception:
        return ds

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings and store in Zarr format')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--amazon_json_list', type=str, required=True,
                       help='Path to Amazon JSON list file')
    parser.add_argument('--amazon_image_root', type=str, required=True,
                       help='Root directory for Amazon images')
    parser.add_argument('--calibre_json_list', type=str, default=None,
                       help='Path to CalibreComics JSON list file (optional)')
    parser.add_argument('--calibre_image_root', type=str, default=None,
                       help='Root directory for CalibreComics images (optional)')
    parser.add_argument('--output_dir', type=str, default='embeddings_output',
                       help='Output directory for Zarr files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Attention temperature')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='Do not run model inference; perform mapping-only audit using the provided manifest lists')
    parser.add_argument('--incremental', action='store_true', default=False,
                       help='Open existing Zarr and append new pages instead of creating a fresh dataset')
    parser.add_argument('--ledger_path', type=str, default=None,
                       help='Optional path to a ledger file (key->index mapping). Defaults to <output_dir>/page_id_ledger.jsonl')
    parser.add_argument('--concurrency_safe', action='store_true', default=False,
                       help='Acquire a simple file lock when appending to an existing Zarr to avoid concurrent writers')
    parser.add_argument('--allow_overwrite', action='store_true', default=False,
                       help='If set, allow incoming inputs to overwrite existing indices (dangerous)')
    parser.add_argument('--save_mean', action='store_true', default=False,
                       help='Save computed mean page embedding to a .npy file')
    parser.add_argument('--debug_manifest', action='store_true', default=False,
                       help='Collect and write mapping attempts for JSONs that fail to map to dataset rows')
    parser.add_argument('--debug_empty', action='store_true', default=False,
                       help='Collect diagnostics for pages that result in empty page embeddings')
    parser.add_argument('--debug_sample_n', type=int, default=10,
                       help='Number of sample empty pages to inspect when --debug_empty is set')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print per-page diagnostics (panel_mask_sum, embedding norm) and save CSV to output dir')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='If set, allow overwriting an existing output Zarr. Otherwise the script will abort if the target exists.')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device, args.num_heads, args.temperature)
    
    # Load Amazon data
    print("Loading Amazon data...")
    amazon_data = []
    with open(args.amazon_json_list, 'r', encoding='utf-8') as f:
        for line in f:
            json_path = line.strip()
            if json_path:
                # Infer source from path so we correctly standardize IDs later
                lp = json_path.lower()
                if 'calibre' in lp or 'calibrecomics' in lp or 'calibre_comics' in lp or 'calibrecomics_analysis' in lp:
                    src = 'calibre'
                else:
                    src = 'amazon'
                amazon_data.append({'path': json_path, 'source': src})
    # Preserve the full training list exactly as provided — do NOT de-duplicate
    # here. The user requested the generator must use the exact training list
    # and paths so we keep order and duplicates intact.

    if args.max_samples:
        amazon_data = amazon_data[:args.max_samples]

    # Write inputs_used.jsonl for full auditability (exact training list + order)
    try:
        inputs_used_path = os.path.join(args.output_dir, 'inputs_used.jsonl')
        with open(inputs_used_path, 'w', encoding='utf-8') as iu:
            for i, item in enumerate(amazon_data):
                record = {'index': i, 'path': item.get('path'), 'source': item.get('source', 'amazon')}
                iu.write(json.dumps(record) + '\n')
        print(f"Wrote inputs_used manifest: {inputs_used_path}")
    except Exception as e:
        print(f"Warning: failed to write inputs_used.jsonl: {e}")
    
    # Load CalibreComics data if provided
    calibre_data = None
    if args.calibre_json_list and args.calibre_image_root:
        print("Loading CalibreComics data...")
        calibre_data = []
        with open(args.calibre_json_list, 'r', encoding='utf-8') as f:
            for line in f:
                json_path = line.strip()
                if json_path:
                    calibre_data.append({'path': json_path, 'source': 'calibre'})
        
        if args.max_samples:
            calibre_data = calibre_data[:args.max_samples]

        # Append Calibre inputs to inputs_used.jsonl for auditability
        try:
            inputs_used_path = os.path.join(args.output_dir, 'inputs_used.jsonl')
            with open(inputs_used_path, 'a', encoding='utf-8') as iu:
                base_idx = 0
                try:
                    # count existing entries to maintain indices
                    with open(inputs_used_path, 'r', encoding='utf-8') as riu:
                        base_idx = sum(1 for _ in riu)
                except Exception:
                    base_idx = 0
                for j, item in enumerate(calibre_data):
                    record = {'index': base_idx + j, 'path': item.get('path'), 'source': item.get('source', 'calibre')}
                    iu.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"Warning: failed to append calibre inputs to inputs_used.jsonl: {e}")
    
    # Create dataloader
    print("Creating dataloader using training loader...")
    # If dry_run is set, skip creating the heavy dataloader and model.
    dataloader = None
    if not args.dry_run:
        dataloader = create_dataloader_from_list(
            args.amazon_json_list,
            args.amazon_image_root,
            batch_size=args.batch_size,
            num_workers=0,
            max_samples=args.max_samples
        )
    
    # Generate embeddings
    output_path = os.path.join(args.output_dir, 'combined_embeddings.zarr')
    # Safety: avoid accidentally overwriting an existing Zarr unless user explicitly permits it.
    if os.path.exists(output_path) and not getattr(args, 'overwrite', False):
        print(f"Error: output Zarr already exists at: {output_path}")
        print("If you want to overwrite it, re-run with the --overwrite flag.")
        return
    # If dry_run, perform mapping-only audit and exit without running model inference
    if args.dry_run:
        print("Dry-run mode: performing mapping-only audit (no model inference)")
        # Build the dataset-like manifest lists in memory (same canonicalization used by create_zarr_dataset)
        all_page_ids = []
        all_manifest_paths = []
        all_sources = []
        if amazon_data:
            for page_data in amazon_data:
                raw_path = str(page_data.get('path'))
                norm_path = os.path.normcase(os.path.normpath(raw_path))
                all_page_ids.append(norm_path)
                all_manifest_paths.append(raw_path)
                all_sources.append(page_data.get('source', 'amazon'))
        if calibre_data:
            for page_data in calibre_data:
                raw_path = str(page_data.get('path'))
                norm_path = os.path.normcase(os.path.normpath(raw_path))
                all_page_ids.append(norm_path)
                all_manifest_paths.append(raw_path)
                all_sources.append('calibre')

        # Build mapping from normalized manifest path -> list of dataset indices
        manifest_path_to_page_idxs = {}
        for idx, raw in enumerate(all_manifest_paths):
            try:
                key = os.path.normcase(os.path.normpath(str(raw))).lower()
                manifest_path_to_page_idxs.setdefault(key, []).append(idx)
            except Exception:
                pass
            try:
                alt = str(raw).replace('_analysis', '_extracted')
                key_alt = os.path.normcase(os.path.normpath(alt)).lower()
                manifest_path_to_page_idxs.setdefault(key_alt, []).append(idx)
            except Exception:
                pass

        # Perform mapping-only pass over inputs in exact order
        inputs = list(amazon_data)
        if calibre_data:
            inputs += list(calibre_data)

        written_index_map = {}
        mapping_collisions = []
        mapping_failures = []
        audit_records = []

        for i, item in enumerate(inputs):
            jf = str(item.get('path') or item.get('json') or item.get('file') or '')
            json_path = jf
            candidates = []
            if json_path:
                key = os.path.normcase(os.path.normpath(str(json_path))).lower()
                candidates = manifest_path_to_page_idxs.get(key, [])
            if not candidates and json_path:
                alt = str(json_path).replace('_analysis', '_extracted')
                key_alt = os.path.normcase(os.path.normpath(alt)).lower()
                candidates = manifest_path_to_page_idxs.get(key_alt, [])

            if candidates:
                chosen = None
                for c in candidates:
                    if c not in written_index_map:
                        chosen = c
                        break
                if chosen is None:
                    # All candidate indices taken -> collision
                    first = candidates[0]
                    mapping_collisions.append({'candidate_indices': candidates, 'page_index': int(first), 'first_json': written_index_map.get(first), 'incoming_json': jf})
                    audit_records.append({'json_file': jf, 'json_path': json_path, 'status': 'skipped_collision', 'reason': 'all_candidate_indices_taken', 'candidate_indices': candidates})
                else:
                    written_index_map[chosen] = jf
                    audit_records.append({'json_file': jf, 'json_path': json_path, 'status': 'would_write', 'page_index': int(chosen), 'page_id': all_page_ids[chosen]})
            else:
                mapping_failures.append({'json_file': jf, 'json_path': json_path})
                audit_records.append({'json_file': jf, 'json_path': json_path, 'status': 'skipped_no_mapping', 'reason': 'no_exact_match'})

        # Write audit files and summaries
        try:
            audit_path = os.path.join(args.output_dir, 'audit_mapping.jsonl')
            with open(audit_path, 'w', encoding='utf-8') as af:
                for r in audit_records:
                    af.write(json.dumps(r) + '\n')
            print(f"Wrote audit file: {audit_path}")
        except Exception as e:
            print(f"Warning: failed to write audit file during dry-run: {e}")

        try:
            mf_path = os.path.join(args.output_dir, 'mapping_failures.jsonl')
            with open(mf_path, 'w', encoding='utf-8') as mf:
                for item in mapping_failures:
                    mf.write(json.dumps(item) + '\n')
            if mapping_collisions:
                collisions_path = os.path.join(args.output_dir, 'mapping_collisions.jsonl')
                with open(collisions_path, 'w', encoding='utf-8') as cf:
                    for item in mapping_collisions:
                        cf.write(json.dumps(item) + '\n')
        except Exception as e:
            print(f"Warning: failed to write mapping debug files during dry-run: {e}")

        print(f"Dry-run summary: inputs={len(inputs)} would_write={len(written_index_map)} failures={len(mapping_failures)} collisions={len(mapping_collisions)}")
        return

    ds = generate_embeddings(model, dataloader, device, output_path, amazon_data, calibre_data, args)
    
    print("Embedding generation complete!")
    print(f"Dataset saved to: {output_path}")
    print(f"Total pages: {len(ds.page_id)}")
    print(f"Amazon pages: {len(amazon_data)}")
    if calibre_data:
        print(f"CalibreComics pages: {len(calibre_data)}")

    # Compute mean page embedding (ignore empty/zero rows) and print summary.
    try:
        page_embs = np.array(ds['page_embeddings'].values)
        norms = np.linalg.norm(page_embs, axis=1)
        valid_mask = norms > 1e-6
        valid_count = int(np.sum(valid_mask))
        if valid_count > 0:
            mean_emb = page_embs[valid_mask].mean(axis=0).astype(np.float32)
        else:
            mean_emb = np.zeros((int(ds.attrs.get('embedding_dim', page_embs.shape[1]) if page_embs.size else 0),), dtype=np.float32)

        # Print a concise numeric summary to stdout (first 16 values and L2 norm)
        preview = np.array2string(mean_emb[:16], precision=6, separator=', ')
        l2 = float(np.linalg.norm(mean_emb)) if mean_emb.size else 0.0
        print(f"Mean page embedding (first 16 dims): {preview}")
        print(f"Mean embedding L2 norm: {l2:.6f}  (computed from {valid_count} pages)")

        # Save only if the user requested it
        if args.save_mean:
            mean_path = os.path.join(args.output_dir, 'mean_page_embedding.npy')
            np.save(mean_path, mean_emb)
            print(f"Saved mean page embedding to: {mean_path}")
    except Exception as e:
        print(f"Warning: failed to compute mean embedding: {e}")

if __name__ == "__main__":
    main()
