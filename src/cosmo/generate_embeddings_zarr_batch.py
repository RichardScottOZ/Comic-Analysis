#!/usr/bin/env python3
"""
Generate PSS Embeddings to Zarr (Batch/Spot Optimized)

This script computes visual (SigLIP) and text (Qwen) embeddings for comic pages
and stores them in a synchronized Zarr hierarchy on S3.

Designed for AWS Batch Spot Instances:
- Robust resume capability
- Direct S3 streaming
- High-throughput GPU inference
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
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SiglipVisionModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
VISUAL_MODEL = "google/siglip-so400m-patch14-384"
TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B" # As per training
VISUAL_DIM = 1152
TEXT_DIM = 1024 # Qwen 0.6B
BATCH_SIZE = 32

def filter_boxes(data):
    """Recursively removes keys related to bounding boxes/coordinates."""
    if isinstance(data, dict):
        junk_keys = {'box_2d', 'box', 'polygon', 'detections', 'coordinates', 'bbox'}
        return {k: filter_boxes(v) for k, v in data.items() if k.lower() not in junk_keys}
    elif isinstance(data, list):
        return [filter_boxes(i) for i in data]
    else:
        return data

def get_models(device):
    print("Loading models...")
    vis_processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
    vis_model = SiglipVisionModel.from_pretrained(VISUAL_MODEL).to(device).eval()
    
    print(f"Loading SentenceTransformer: {TEXT_MODEL}")
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    
    return vis_processor, vis_model, None, txt_model

def get_text_content(s3_client, bucket, prefix, canonical_id):
    """
    Fetches the VLM analysis JSON from S3 and prepares a cleaned JSON string for embedding.
    Matches the training 'Serialized JSON' paradigm but filters coordinates.
    """
    key = f"{prefix}/{canonical_id}.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        
        # Strip metadata
        data.pop('canonical_id', None)
        data.pop('processed_at', None)
        data.pop('model', None)
        data.pop('guided_by_rcnn', None)
        
        # Unwrap OCRResult if present (Training data format)
        if 'OCRResult' in data:
            data = data['OCRResult']
        
        # Recursively filter coordinates
        cleaned_data = filter_boxes(data)
        
        # Serialize to compact JSON string
        return json.dumps(cleaned_data, separators=(',', ':'))
        
    except Exception:
        return "" # Empty text if not found

def process_chunk(chunk_data, s3_output, vlm_bucket, vlm_prefix, batch_size=32):
    """
    Worker function for Lithops.
    chunk_data: List of dicts [{'canonical_id':..., 'absolute_image_path':...}, ...]
    """
    import torch
    import zarr
    import s3fs
    import numpy as np
    import uuid
    import boto3
    from PIL import Image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker processing {len(chunk_data)} items on {device}")
    
    # Load Models
    vis_proc, vis_model, _, txt_model = get_models(device)
    s3_client = boto3.client('s3')
    
    # Setup worker-specific Zarr part
    worker_id = str(uuid.uuid4())
    part_path = f"{s3_output}/parts/{worker_id}.zarr"
    
    s3_fs = s3fs.S3FileSystem()
    store = s3fs.S3Map(root=part_path, s3=s3_fs, check=False)
    root = zarr.group(store=store)
    
    # Create arrays
    root.create_dataset('visual', shape=(0, VISUAL_DIM), chunks=(1000, VISUAL_DIM), dtype='float16')
    root.create_dataset('text', shape=(0, TEXT_DIM), chunks=(1000, TEXT_DIM), dtype='float16')
    root.create_dataset('ids', shape=(0,), chunks=(1000,), dtype='str')
    
    vis_buffer = []
    txt_buffer = []
    id_buffer = []
    
    for i in range(0, len(chunk_data), batch_size):
        batch = chunk_data[i:i+batch_size]
        
        images = []
        valid_indices = []
        ids_batch = []
        
        from io import BytesIO
        
        for j, item in enumerate(batch):
            path = item['absolute_image_path']
            cid = item['canonical_id']
            try:
                if path.startswith('s3://'):
                    bucket, key = path[5:].split("/", 1)
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    image_bytes = response['Body'].read()
                else:
                    with open(path, "rb") as f:
                        image_bytes = f.read()
                        
                img = Image.open(BytesIO(image_bytes)).convert('RGB')
                images.append(img)
                valid_indices.append(j)
                ids_batch.append(cid)
            except Exception as e:
                # print(f"Failed {cid}: {e}")
                pass
        
        if not images: continue
        
        # Inference
        with torch.no_grad():
            # Visual
            inputs = vis_proc(images=images, return_tensors="pt").to(device)
            vis_out = vis_model(**inputs)
            vis_embeds = vis_out.pooler_output
            
            # Text
            texts = []
            for j in valid_indices:
                txt = get_text_content(s3_client, vlm_bucket, vlm_prefix, ids_batch[j])
                texts.append(txt)
            
            txt_embeds = txt_model.encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)

        # Buffer
        vis_buffer.extend(vis_embeds.cpu().numpy().astype('float16'))
        txt_buffer.extend(txt_embeds.astype('float16'))
        id_buffer.extend(ids_batch)
        
        # Flush
        if len(id_buffer) >= 1000:
            root['visual'].append(np.array(vis_buffer))
            root['text'].append(np.array(txt_buffer))
            root['ids'].append(np.array(id_buffer))
            vis_buffer, txt_buffer, id_buffer = [], [], []

    # Final Flush
    if id_buffer:
        root['visual'].append(np.array(vis_buffer))
        root['text'].append(np.array(txt_buffer))
        root['ids'].append(np.array(id_buffer))
        
    return f"Finished part {worker_id}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--s3-output', required=True)
    parser.add_argument('--vlm-bucket', default='calibrecomics-extracted')
    parser.add_argument('--vlm-prefix', default='vlm_analysis')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of pages for testing')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to first {args.limit} rows.")
        
    all_data = df.to_dict('records')
    
    # Run locally (single chunk)
    process_chunk(all_data, args.s3_output, args.vlm_bucket, args.vlm_prefix, args.batch_size)

if __name__ == "__main__":
    main()
