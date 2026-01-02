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
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

# --- Configuration ---
VISUAL_MODEL = "google/siglip-so400m-patch14-384"
TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B" # As per training
VISUAL_DIM = 1152
TEXT_DIM = 1024 # Qwen 0.6B
BATCH_SIZE = 32

class ComicDataset(Dataset):
    def __init__(self, manifest_path, processed_ids):
        self.data = []
        print(f"Loading manifest: {manifest_path}")
        
        # Load manifest using pandas for speed
        df = pd.read_csv(manifest_path)
        
        # Filter processed
        if processed_ids:
            initial_len = len(df)
            df = df[~df['canonical_id'].isin(processed_ids)]
            print(f"Skipped {initial_len - len(df)} processed pages.")
            
        self.data = df.to_dict('records')
        print(f"Remaining pages to process: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            'id': row['canonical_id'],
            'image_path': row['absolute_image_path'],
        }

def get_models(device):
    print("Loading models...")
    # Visual
    vis_processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
    vis_model = AutoModel.from_pretrained(VISUAL_MODEL).to(device).eval()
    
    # Text (SentenceTransformer)
    print(f"Loading SentenceTransformer: {TEXT_MODEL}")
    txt_model = SentenceTransformer(TEXT_MODEL, device=str(device))
    
    return vis_processor, vis_model, None, txt_model

def get_text_content(s3_client, bucket, prefix, canonical_id):
    """
    Fetches the VLM analysis JSON from S3 to get the text for embedding.
    """
    key = f"{prefix}/{canonical_id}.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        
        # Strategy: Embed the "Overall Summary" + "Panel Descriptions"
        text = data.get('overall_summary', '')
        
        if 'panels' in data:
            for p in data['panels']:
                desc = p.get('description', '')
                if desc: text += " " + desc
                
        return text.strip()
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
    from PIL import Image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Worker processing {len(chunk_data)} items on {device}")
    
    # Load Models
    vis_proc, vis_model, _, txt_model = get_models(device)
    s3_client = boto3.client('s3')
    
    # Setup worker-specific Zarr part
    worker_id = str(uuid.uuid4())
    part_path = f"{s3_output}/parts/{worker_id}.zarr"
    
    s3 = s3fs.S3FileSystem()
    store = s3fs.S3Map(root=part_path, s3=s3, check=False)
    root = zarr.group(store=store)
    
    # Create arrays
    root.create_dataset('visual', shape=(0, VISUAL_DIM), chunks=(1000, VISUAL_DIM), dtype='float16')
    root.create_dataset('text', shape=(0, TEXT_DIM), chunks=(1000, TEXT_DIM), dtype='float16')
    root.create_dataset('ids', shape=(0,), chunks=(1000,), dtype='str')
    
    vis_buffer = []
    txt_buffer = []
    id_buffer = []
    
    # Process Loop
    for i in range(0, len(chunk_data), batch_size):
        batch = chunk_data[i:i+batch_size]
        
        images = []
        valid_indices = []
        ids_batch = []
        
        for j, item in enumerate(batch):
            path = item['absolute_image_path']
            cid = item['canonical_id']
            try:
                bucket, key = path.replace("s3://", "").split("/", 1)
                response = s3_client.get_object(Bucket=bucket, Key=key)
                img = Image.open(response['Body']).convert('RGB')
                images.append(img)
                valid_indices.append(j)
                ids_batch.append(cid)
            except Exception as e:
                print(f"Failed {cid}: {e}")
        
        if not images: continue
        
        # Inference
        with torch.no_grad():
            # Visual
            inputs = vis_proc(images=images, return_tensors="pt").to(device)
            vis_out = vis_model(**inputs)
            if hasattr(vis_out, 'pooler_output'):
                vis_embeds = vis_out.pooler_output
            else:
                vis_embeds = vis_out.last_hidden_state.mean(dim=1)
            
            # Text
            texts = []
            for j in valid_indices:
                txt = get_text_content(s3_client, vlm_bucket, vlm_prefix, ids_batch[j])
                texts.append(txt)
            
            txt_embeds = txt_model.encode(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
            txt_embeds = torch.from_numpy(txt_embeds)

        # Buffer
        vis_buffer.extend(vis_embeds.cpu().numpy().astype('float16'))
        txt_buffer.extend(txt_embeds.numpy().astype('float16'))
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
    args = parser.parse_args()
    
    # Load all data
    df = pd.read_csv(args.manifest)
    all_data = df.to_dict('records')
    
    # Run locally (single chunk)
    process_chunk(all_data, args.s3_output, args.vlm_bucket, args.vlm_prefix, args.batch_size)

if __name__ == "__main__":
    main()