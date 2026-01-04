#!/usr/bin/env python3
"""
Generate PSS Embeddings to Zarr (Xarray Region-Write Optimized)
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
BATCH_SIZE = 32

def clean_series_name(name: str) -> str:
    name_no_volume = re.sub(r'\s+(v\d+|\(\d{4}-\d{4}\)|\d{4})', '', name)
    cleaned = re.sub(r'[^\w\s]', '', name_no_volume)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned.lower()

def extract_metadata(path: str, cid: str):
    parts = Path(path).parts
    filename = parts[-1]
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

def get_models(device):
    vis_processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
    vis_model = SiglipVisionModel.from_pretrained(VISUAL_MODEL).to(device).eval()
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    return vis_processor, vis_model, txt_model

def get_text_content(s3_client, bucket, prefix, canonical_id):
    key = f"{prefix}/{canonical_id}.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        if 'OCRResult' in data: data = data['OCRResult']
        cleaned_data = filter_boxes(data)
        return json.dumps(cleaned_data, separators=(',', ':'))
    except Exception:
        return ""

def process_chunk(chunk_data, s3_output, vlm_bucket='calibrecomics-extracted', vlm_prefix='vlm_analysis', batch_size=32, start_index=0):
    """
    Worker function for Lithops. Writes to a pre-allocated Xarray Zarr using regions.
    """
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
    s3_fs = s3fs.S3FileSystem()
    store = s3fs.S3Map(root=s3_output, s3=s3_fs, check=False) if s3_output.startswith("s3://") else s3_output

    total_in_chunk = len(chunk_data)
    
    for i in range(0, len(chunk_data), batch_size):
        batch = chunk_data[i:i+batch_size]
        
        # Log progress
        if len(batch) > 0:
            print(f"[Worker {start_index}] Batch {i//batch_size}: {batch[0]['canonical_id']} ... ({len(batch)} items)")
        
        images = []
        valid_indices = []
        ids_batch = []
        for item in batch:
            path = item['absolute_image_path']
            cid = item['canonical_id']
            try:
                if path.startswith('s3://'):
                    b, k = path[5:].split("/", 1)
                    res = s3_client.get_object(Bucket=b, Key=k)
                    image_bytes = res['Body'].read()
                else:
                    with open(path, "rb") as f: image_bytes = f.read()
                img = Image.open(BytesIO(image_bytes)).convert('RGB')
                images.append(img)
                ids_batch.append(cid)
                meta_batch.append(extract_metadata(path, cid))
            except:
                images.append(Image.new('RGB', (384, 384))) # Placeholder
                ids_batch.append("FAILED")
                meta_batch.append(extract_metadata("", "FAILED"))

        # Inference
        with torch.no_grad():
            vis_embeds = vis_model(**vis_proc(images=images, return_tensors="pt").to(device)).pooler_output
            texts = [get_text_content(s3_client, vlm_bucket, vlm_prefix, cid) for cid in ids_batch]
            txt_embeds = txt_model.encode(texts, batch_size=len(texts), show_progress_bar=False)

        # Create Xarray Dataset for this Batch Region
        ds_batch = xr.Dataset(
            data_vars={
                'visual': (['page_id', 'visual_dim'], vis_embeds.cpu().numpy().astype('float16')),
                'text': (['page_id', 'text_dim'], txt_embeds.astype('float16')),
                'ids': (['page_id'], np.array(ids_batch, dtype='<U128')),
                'series': (['page_id'], np.array([m['series'] for m in meta_batch], dtype='<U128')),
                'volume': (['page_id'], np.array([m['volume'] for m in meta_batch], dtype='<U32')),
                'issue': (['page_id'], np.array([m['issue'] for m in meta_batch], dtype='<U16')),
                'page_num': (['page_id'], np.array([m['page'] for m in meta_batch], dtype='<U16')),
                'source': (['page_id'], np.array([m['source'] for m in meta_batch], dtype='<U16'))
            },
            coords={'page_id': np.arange(start_index + i, start_index + i + actual_batch_size)}
        )

        # Write to Region
        ds_batch.to_zarr(store, region={'page_id': slice(start_index + i, start_index + i + actual_batch_size)})

    return f"Finished chunk {start_index} to {start_index + total_in_chunk}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--s3-output', required=True)
    parser.add_argument('--vlm-bucket', default='calibrecomics-extracted')
    parser.add_argument('--vlm-prefix', default='vlm_analysis')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    
    # Load data
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} rows.")
        
    all_data = df.to_dict('records')
    
    # Pre-allocate logic (Local Test Mode)
    import shutil
    if args.limit and not args.s3_output.startswith("s3://"):
        if os.path.exists(args.s3_output):
            print(f"Cleaning up existing test output: {args.s3_output}")
            shutil.rmtree(args.s3_output)
            
    if not os.path.exists(args.s3_output):
        print(f"Creating skeleton: {args.s3_output}")
        # Call preallocate logic (simplified for local)
        coords = {'page_id': np.arange(args.limit), 'visual_dim': np.arange(VISUAL_DIM), 'text_dim': np.arange(TEXT_DIM)}
        ds = xr.Dataset(
            data_vars={
                'visual': (['page_id', 'visual_dim'], np.zeros((args.limit, VISUAL_DIM), dtype='float16')),
                'text': (['page_id', 'text_dim'], np.zeros((args.limit, TEXT_DIM), dtype='float16')),
                'ids': (['page_id'], np.full(args.limit, '', dtype='<U128')),
                'series': (['page_id'], np.full(args.limit, '', dtype='<U128')),
                'volume': (['page_id'], np.full(args.limit, '', dtype='<U32')),
                'issue': (['page_id'], np.full(args.limit, '', dtype='<U16')),
                'page_num': (['page_id'], np.full(args.limit, '', dtype='<U16')),
                'source': (['page_id'], np.full(args.limit, '', dtype='<U16'))
            },
            coords=coords
        )
        ds.to_zarr(args.s3_output, compute=False, mode='w')

    # Run locally (single chunk)
    process_chunk(all_data, args.s3_output, args.vlm_bucket, args.vlm_prefix, args.batch_size, start_index=0)
    
    print("\n--- Xarray Verification ---")
    ds = xr.open_zarr(args.s3_output)
    print(ds)

if __name__ == "__main__":
    main()
