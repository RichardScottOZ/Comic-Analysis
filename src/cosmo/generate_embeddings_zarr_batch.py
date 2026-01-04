#!/usr/bin/env python3
"""
Generate PSS Embeddings to Zarr (Optimized for Local & Cloud)
Features:
- Xarray Region-Writes
- Multi-threaded Image Loading (DataLoader)
- Metadata extraction
- Local VLM root with S3 fallback + Manifest Lookup
- Runtime tracking
- Verbose logging for path debugging
"""

import os
import argparse
import json
import torch
import zarr
import s3fs
import numpy as np
import pandas as pd
import boto3
import xarray as xr
import re
import time
import csv
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SiglipVisionModel
from sentence_transformers import SentenceTransformer
from io import BytesIO

# --- Configuration ---
VISUAL_MODEL = "google/siglip-so400m-patch14-384"
TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VISUAL_DIM = 1152
TEXT_DIM = 1024

# Global Lookup
S3_LOOKUP = {}

def load_s3_lookup(manifest_path):
    print(f"Loading S3 manifest lookup: {manifest_path}")
    global S3_LOOKUP
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            # Store the full ID
            # Also store suffix parts to handle partial matches from local manifest
            parts = cid.split('/')
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in S3_LOOKUP:
                    S3_LOOKUP[suffix] = cid
            # Always ensure filename is keyed
            S3_LOOKUP[Path(cid).name] = cid

# --- Metadata Extraction ---
def clean_series_name(name: str) -> str:
    name_no_volume = re.sub(r'\s+(v\d+|\(\d{4}-\d{4}\)|\d{4})', '', name)
    cleaned = re.sub(r'[^\w\s]', '', name_no_volume)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned.lower()

def extract_metadata(path: str, cid: str):
    parts = Path(path).parts
    filename = parts[-1] if parts else ""
    meta = {'series': 'unknown', 'volume': 'unknown', 'issue': '000', 'page': 'p000', 'source': 'unknown'}
    if 'amazon' in path.lower(): meta['source'] = 'amazon'
    elif 'calibre' in path.lower(): meta['source'] = 'calibre'
    page_match = re.search(r'p(\d{3,4})', filename)
    if page_match: meta['page'] = f"p{page_match.group(1)}"
    issue_match = re.search(r'(\d{3})', filename)
    if issue_match: meta['issue'] = issue_match.group(1)
    if len(parts) >= 2:
        parent = parts[-2]
        meta['series'] = clean_series_name(parent)
        vol_match = re.search(r'(v\d+|\(\d{4}-\d{4}\)|\d{4})', parent)
        if vol_match: meta['volume'] = vol_match.group(1)
    return meta

def filter_boxes(data):
    if isinstance(data, dict):
        junk_keys = {'box_2d', 'box', 'polygon', 'detections', 'coordinates', 'bbox'}
        return {k: filter_boxes(v) for k, v in data.items() if k.lower() not in junk_keys}
    elif isinstance(data, list):
        return [filter_boxes(i) for i in data]
    else:
        return data

# --- Data Loading ---
class ComicEmbeddingDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.s3_client = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec['absolute_image_path']
        cid = rec['canonical_id']
        
        try:
            if path.startswith('s3://'):
                if self.s3_client is None: self.s3_client = boto3.client('s3')
                b, k = path[5:].split("/", 1)
                res = self.s3_client.get_object(Bucket=b, Key=k)
                image_bytes = res['Body'].read()
            else:
                with open(path, "rb") as f: image_bytes = f.read()
            
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            return {'image': img, 'id': cid, 'path': path, 'status': 'success'}
        except Exception as e:
            return {'id': cid, 'path': path, 'status': 'error', 'error': str(e)}

def collate_fn(batch):
    valid = [item for item in batch if item['status'] == 'success']
    errors = [item for item in batch if item['status'] == 'error']
    return valid, errors

# --- Models ---
def get_models(device):
    print("Loading models...")
    vis_processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
    vis_model = SiglipVisionModel.from_pretrained(VISUAL_MODEL).to(device).eval()
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    return vis_processor, vis_model, txt_model

def get_text_content(s3_client, bucket, prefix, canonical_id, local_vlm_root=None, verbose=False):
    """
    Fetches VLM analysis JSON from local cache or S3.
    """
    # Resolve ID
    target_id = canonical_id
    if S3_LOOKUP:
        if canonical_id in S3_LOOKUP:
            target_id = S3_LOOKUP[canonical_id]
        elif Path(canonical_id).name in S3_LOOKUP:
            target_id = S3_LOOKUP[Path(canonical_id).name]
        elif 'amazon' in canonical_id:
             idx = canonical_id.find('amazon')
             short = canonical_id[idx:]
             if short in S3_LOOKUP: target_id = S3_LOOKUP[short]
             
    if verbose:
        print(f"[LOOKUP] '{canonical_id}' -> '{target_id}'")

    # 1. Try Local First
    if local_vlm_root:
        # Standard candidates
        candidates = [
            os.path.join(local_vlm_root, target_id.replace('/', os.sep) + ".json"),
            os.path.join(local_vlm_root, f"{Path(target_id).name}.json"),
        ]
        
        # Try prepending CalibreComics_extracted if missing
        if "CalibreComics_extracted" not in target_id:
             candidates.append(os.path.join(local_vlm_root, "CalibreComics_extracted", target_id.replace('/', os.sep) + ".json"))

        for p in candidates:
            if verbose: print(f"  [CHECK] {p}")
            if os.path.exists(p):
                if verbose: print(f"  [FOUND] {p}")
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if 'OCRResult' in data: data = data['OCRResult']
                    cleaned = filter_boxes(data)
                    return json.dumps(cleaned, separators=(',', ':'))
                except Exception as e:
                    if verbose: print(f"  [ERROR] Read failed: {e}")
                    pass
        
        # If we reach here, local lookup FAILED.
        if verbose: print(f"  [MISSING] Could not find {target_id} locally.")
        return "" 

    # 2. Pure Cloud Mode (only if local root NOT provided)
    if s3_client:
        key = f"{prefix}/{target_id}.json"
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
            if 'OCRResult' in data: data = data['OCRResult']
            cleaned = filter_boxes(data)
            return json.dumps(cleaned, separators=(',', ':'))
        except Exception:
            pass
    
    return ""

# --- Main Worker ---
def process_chunk(chunk_data, s3_output, vlm_bucket='calibrecomics-extracted', vlm_prefix='vlm_analysis', batch_size=32, start_index=0, num_workers=4, local_vlm_root=None, verbose=False):
    import torch
    import xarray as xr
    import numpy as np
    import s3fs
    import boto3
    from PIL import Image
    from io import BytesIO
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_proc, vis_model, txt_model = get_models(device)
    s3_client = boto3.client('s3')
    
    # Initialize S3 Store
    if s3_output.startswith("s3://"):
        s3_fs = s3fs.S3FileSystem()
        store = s3fs.S3Map(root=s3_output, s3=s3_fs, check=False)
    else:
        store = s3_output

    dataset = ComicEmbeddingDataset(chunk_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    current_idx = start_index
    
    for valid_batch, error_batch in tqdm(loader, desc=f"Worker {start_index}"):
        if not valid_batch and not error_batch: continue
        
        actual_batch_size = len(valid_batch) + len(error_batch)
        
        # 1. Process Valid Items
        if valid_batch:
            images = [item['image'] for item in valid_batch]
            ids = [item['id'] for item in valid_batch]
            paths = [item['path'] for item in valid_batch]
            
            with torch.no_grad():
                # Visual
                inputs = vis_proc(images=images, return_tensors="pt").to(device)
                vis_embeds = vis_model(**inputs).pooler_output
                
                # Text
                texts = [get_text_content(s3_client, vlm_bucket, vlm_prefix, cid, local_vlm_root, verbose) for cid in ids]
                txt_embeds = txt_model.encode(texts, batch_size=len(texts), show_progress_bar=False)

            # Metadata
            meta_batch = [extract_metadata(p, cid) for p, cid in zip(paths, ids)]

            # Construct Dataset
            ds_batch = xr.Dataset(
                data_vars={
                    'visual': (['page_id', 'visual_dim'], vis_embeds.cpu().numpy().astype('float16')),
                    'text': (['page_id', 'text_dim'], txt_embeds.astype('float16')),
                    'ids': (['page_id'], np.array(ids, dtype='<U128')),
                    'series': (['page_id'], np.array([m['series'] for m in meta_batch], dtype='<U128')),
                    'volume': (['page_id'], np.array([m['volume'] for m in meta_batch], dtype='<U32')),
                    'issue': (['page_id'], np.array([m['issue'] for m in meta_batch], dtype='<U16')),
                    'page_num': (['page_id'], np.array([m['page'] for m in meta_batch], dtype='<U16')),
                    'source': (['page_id'], np.array([m['source'] for m in meta_batch], dtype='<U16'))
                },
                coords={'page_id': np.arange(current_idx, current_idx + len(ids))}
            )
            
            ds_batch.to_zarr(store, region={'page_id': slice(current_idx, current_idx + len(ids))})

        # Update global offset
        current_idx += actual_batch_size

    return f"Finished chunk starting at {start_index}"

def main():
    import time
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--s3-output', required=True)
    parser.add_argument('--s3-manifest', help='Optional S3 manifest for ID lookup')
    parser.add_argument('--vlm-bucket', default='calibrecomics-extracted')
    parser.add_argument('--vlm-prefix', default='vlm_analysis')
    parser.add_argument('--local-vlm-root', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4, help="Image loading threads")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging for file lookup')
    args = parser.parse_args()
    
    if args.s3_manifest:
        load_s3_lookup(args.s3_manifest)
        
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} rows.")
        
    all_data = df.to_dict('records')
    total_len = len(all_data)
    
    # Pre-allocate logic
    import shutil
    if args.limit and not args.s3_output.startswith("s3://"):
        if os.path.exists(args.s3_output):
            shutil.rmtree(args.s3_output)
            
    if not os.path.exists(args.s3_output) and not args.s3_output.startswith("s3://"):
        print(f"Creating skeleton: {args.s3_output}")
        coords = {'page_id': np.arange(total_len), 'visual_dim': np.arange(VISUAL_DIM), 'text_dim': np.arange(TEXT_DIM)}
        ds = xr.Dataset(
            data_vars={
                'visual': (['page_id', 'visual_dim'], np.zeros((total_len, VISUAL_DIM), dtype='float16')),
                'text': (['page_id', 'text_dim'], np.zeros((total_len, TEXT_DIM), dtype='float16')),
                'ids': (['page_id'], np.full(total_len, '', dtype='<U128')),
                'series': (['page_id'], np.full(total_len, '', dtype='<U128')),
                'volume': (['page_id'], np.full(total_len, '', dtype='<U32')),
                'issue': (['page_id'], np.full(total_len, '', dtype='<U16')),
                'page_num': (['page_id'], np.full(total_len, '', dtype='<U16')),
                'source': (['page_id'], np.full(total_len, '', dtype='<U16'))
            },
            coords=coords
        )
        ds.to_zarr(args.s3_output, compute=False, mode='w')

    process_chunk(all_data, args.s3_output, args.vlm_bucket, args.vlm_prefix, args.batch_size, start_index=0, num_workers=args.workers, local_vlm_root=args.local_vlm_root, verbose=args.verbose)
    
    elapsed = time.time() - start_time
    print(f"\nTotal Runtime: {elapsed:.2f} seconds")
    if args.limit:
        print(f"Estimated time for 1.22M pages: {(elapsed/args.limit * 1220000 / 3600):.2f} hours")

    print("\n--- Xarray Verification ---")
    try:
        ds_verify = xr.open_zarr(args.s3_output)
        print(ds_verify)
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
