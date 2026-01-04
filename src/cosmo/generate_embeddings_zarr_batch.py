#!/usr/bin/env python3
"""
Generate PSS Embeddings to Zarr (Optimized for Local & Cloud)
Features:
- Xarray Region-Writes
- Multi-threaded Image Loading (DataLoader)
- Metadata extraction (Regex improved)
- Local VLM root with S3 fallback + Manifest Lookup
- Runtime tracking
- VRAM Optimized (FP16 + Inference Mode)
- Correct Text serialization for PSS Classifier
- Float32 Output for Training Consistency
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
from numcodecs import Blosc

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
            parts = cid.split('/')
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in S3_LOOKUP:
                    S3_LOOKUP[suffix] = cid
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

    elif 'neonichiban' in path.lower(): meta['source'] = 'neonichiban'

    elif 'humble' in path.lower(): meta['source'] = 'humble_bundle'

    

    page_match = re.search(r'p(\d{3,4})', filename)

    if page_match: meta['page'] = f"p{page_match.group(1)}"

    

    issue_match = re.search(r'(\d{3})', filename)

    if issue_match: meta['issue'] = issue_match.group(1)

    

    if len(parts) >= 2:

        parent = parts[-2]

        

        # Heuristic: Check for junk parent folders

        if parent.lower() in ['jpg4cbz', 'working_files'] and len(parts) >= 3:

            # Use Grandparent

            series_folder = parts[-3]

        else:

            series_folder = parent

            

        meta['series'] = clean_series_name(series_folder)

        

                                # Improved volume regex: matches v03, vol3, _vol3, Vol. 1, Volume 1, Vol._1, (2000), etc.

        

                

        

                                vol_pattern = r'[ _\s\-\(](v\d+|vol\.?\s?_?\d+|volume\s?\d+|\d{4}-\d{4}|\d{4})[_\s\-\)]'

        

                

        

                                vol_match = re.search(vol_pattern, series_folder, re.IGNORECASE)

        

                

        

                        

        

                

        

                                

        

                

        

                        

        

                

        

                                if not vol_match and series_folder != parent:

        

                

        

                        

        

                

        

                                     vol_match = re.search(vol_pattern, parent, re.IGNORECASE)

        

                

        

                        

        

                

        

                                     

        

                

        

                        

        

                

        

                                # Fallback for end of string

        

                

        

                        

        

                

        

                                if not vol_match:

        

                

        

                        

        

                

        

                                     vol_match = re.search(r'[ _\s](v\d+|vol\.?\s?_?\d+|volume\s?\d+)$', parent, re.IGNORECASE):
             vol_match = re.search(r'[_\s](v\d+|vol\d+)$', parent, re.IGNORECASE)
             meta['volume'] = vol_match.group(1)
             
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
    print("Loading models in float16...")
    vis_processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
    vis_model = SiglipVisionModel.from_pretrained(VISUAL_MODEL, torch_dtype=torch.float16).to(device).eval()
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    txt_model.half()
    return vis_processor, vis_model, txt_model

def get_text_content(s3_client, bucket, prefix, canonical_id, local_vlm_root=None, verbose=False):
    target_id = canonical_id
    if S3_LOOKUP:
        if canonical_id in S3_LOOKUP: target_id = S3_LOOKUP[canonical_id]
        elif Path(canonical_id).name in S3_LOOKUP: target_id = S3_LOOKUP[Path(canonical_id).name]
        elif 'amazon' in canonical_id:
             idx = canonical_id.find('amazon')
             short = canonical_id[idx:]
             if short in S3_LOOKUP: target_id = S3_LOOKUP[short]
    
    if verbose: print(f"[LOOKUP] '{canonical_id}' -> '{target_id}'")
    
    if local_vlm_root:
        p_nested = os.path.join(local_vlm_root, target_id.replace('/', os.sep) + ".json")
        p_flat = os.path.join(local_vlm_root, f"{Path(target_id).name}.json")
        candidates = [p_nested, p_flat]
        if "CalibreComics_extracted" not in target_id:
             candidates.append(os.path.join(local_vlm_root, "CalibreComics_extracted", target_id.replace('/', os.sep) + ".json"))
        for p in candidates:
            if verbose: print(f"  [CHECK] {p}")
            if os.path.exists(p):
                if verbose: print(f"  [FOUND] {p}")
                try:
                    with open(p, 'r', encoding='utf-8') as f: data = json.load(f)
                    if 'OCRResult' in data: data = data['OCRResult']
                    cleaned = filter_boxes(data)
                    return json.dumps(cleaned, separators=(',', ':'))
                except: pass
        if verbose: print(f"  [MISSING] Could not find {target_id} locally.")
        return "" 
    
    if s3_client:
        key = f"{prefix}/{target_id}.json"
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
            if 'OCRResult' in data: data = data['OCRResult']
            cleaned = filter_boxes(data)
            return json.dumps(cleaned, separators=(',', ':'))
        except: pass
    return ""

def process_chunk(chunk_data, s3_output, vlm_bucket='calibrecomics-extracted', vlm_prefix='vlm_analysis', batch_size=32, start_index=0, num_workers=4, local_vlm_root=None, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_proc, vis_model, txt_model = get_models(device)
    s3_client = boto3.client('s3')
    
    if s3_output.startswith("s3://"):
        s3_fs = s3fs.S3FileSystem()
        store = s3fs.S3Map(root=s3_output, s3=s3_fs, check=False)
    else: store = s3_output
    dataset = ComicEmbeddingDataset(chunk_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    current_idx = start_index
    
    for valid_batch, error_batch in tqdm(loader, desc=f"Worker {start_index}"):
        if not valid_batch and not error_batch: continue
        actual_batch_size = len(valid_batch) + len(error_batch)
        
        if valid_batch:
            if verbose: print(f"[Worker {start_index}] Batch starting: {valid_batch[0]['id']}")
            images = [item['image'] for item in valid_batch]
            ids = [item['id'] for item in valid_batch]
            paths = [item['path'] for item in valid_batch]
            
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                inputs = vis_proc(images=images, return_tensors="pt").to(device)
                inputs = {k: v.to(torch.float16) if v.is_floating_point() else v for k, v in inputs.items()}
                out = vis_model(**inputs)
                vis_embeds = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:,0,:]
                
                texts = [get_text_content(s3_client, vlm_bucket, vlm_prefix, cid, local_vlm_root, verbose) for cid in ids]
                txt_embeds = txt_model.encode(texts, batch_size=len(texts), show_progress_bar=False)

            meta_batch = [extract_metadata(p, cid) for p, cid in zip(paths, ids)]

            ds_batch = xr.Dataset(
                data_vars={
                    'visual': (['page_id', 'visual_dim'], vis_embeds.cpu().numpy().astype('float32')),
                    'text': (['page_id', 'text_dim'], txt_embeds.astype('float32')),
                    'ids': (['page_id'], np.array(ids, dtype='<U512')),
                    'series': (['page_id'], np.array([m['series'] for m in meta_batch], dtype='<U256')),
                    'volume': (['page_id'], np.array([m['volume'] for m in meta_batch], dtype='<U64')),
                    'issue': (['page_id'], np.array([m['issue'] for m in meta_batch], dtype='<U32')),
                    'page_num': (['page_id'], np.array([m['page'] for m in meta_batch], dtype='<U6')),
                    'source': (['page_id'], np.array([m['source'] for m in meta_batch], dtype='<U16'))
                },
                coords={'page_id': np.arange(current_idx, current_idx + len(ids))}
            )
            ds_batch.to_zarr(store, region={'page_id': slice(current_idx, current_idx + len(ids))})

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
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    if args.s3_manifest: load_s3_lookup(args.s3_manifest)
    df = pd.read_csv(args.manifest)
    if args.limit: df = df.head(args.limit)
    all_data = df.to_dict('records')
    total_len = len(all_data)
    
    import shutil
    if args.limit and not args.s3_output.startswith("s3://"):
        if os.path.exists(args.s3_output): shutil.rmtree(args.s3_output)
            
    if not os.path.exists(args.s3_output) and not args.s3_output.startswith("s3://"):
        print(f"Creating skeleton: {args.s3_output}")
        from numcodecs import Blosc
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        coords = {'page_id': np.arange(total_len), 'visual_dim': np.arange(VISUAL_DIM), 'text_dim': np.arange(TEXT_DIM)}
        ds = xr.Dataset(
            data_vars={
                'visual': (['page_id', 'visual_dim'], np.zeros((total_len, VISUAL_DIM), dtype='float32')),
                'text': (['page_id', 'text_dim'], np.zeros((total_len, TEXT_DIM), dtype='float32')),
                'ids': (['page_id'], np.full(total_len, '', dtype='<U512')),
                'series': (['page_id'], np.full(total_len, '', dtype='<U256')),
                'volume': (['page_id'], np.full(total_len, '', dtype='<U64')),
                'issue': (['page_id'], np.full(total_len, '', dtype='<U32')),
                'page_num': (['page_id'], np.full(total_len, '', dtype='<U6')),
                'source': (['page_id'], np.full(total_len, '', dtype='<U16'))
            },
            coords=coords
        )
        encoding = {v: {'compressor': compressor} for v in ds.data_vars}
        ds.to_zarr(args.s3_output, compute=False, mode='w', encoding=encoding)

    process_chunk(all_data, args.s3_output, args.vlm_bucket, args.vlm_prefix, args.batch_size, 0, args.workers, args.local_vlm_root, args.verbose)
    
    elapsed = time.time() - start_time
    print(f"\nTotal Runtime: {elapsed:.2f} seconds")
    if args.limit: print(f"Estimated time for 1.22M pages: {(elapsed/args.limit * 1220000 / 3600):.2f} hours")
    try:
        ds_verify = xr.open_zarr(args.s3_output)
        print(ds_verify)
    except: pass

if __name__ == "__main__":
    main()
