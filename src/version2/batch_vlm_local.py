#!/usr/bin/env python3
"""
Manifest-driven VLM Captioning (Local GPU - Florence-2)

This script runs VLM captioning (default: Florence-2-large) locally on a GPU using a manifest file.
It outputs individual JSON files for each page containing the generated caption.

Usage:
    python src/version2/batch_vlm_local.py \
        --manifest manifests/calibrecomics-extracted_manifest.csv \
        --output-dir E:/CalibreComics_captions \
        --image-root E:/CalibreComics_extracted \
        --model microsoft/Florence-2-large \
        --batch-size 4
"""

import os
import argparse
import csv
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Global Settings ---
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Disable parallelism tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collate_fn(batch):
    """Custom collate function."""
    return tuple(zip(*batch))

class ManifestDataset(Dataset):
    """Dataset loading images defined in a manifest CSV, skipping existing outputs."""
    def __init__(self, manifest_path, image_root=None, output_dir=None):
        self.image_root = Path(image_root) if image_root else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.records = []

        print(f"Loading manifest from: {manifest_path}")
        initial_count = 0
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                initial_count += 1
                # Skip if output already exists
                if self.output_dir:
                    out_path = self.output_dir / f"{row['canonical_id']}_caption.json"
                    if out_path.exists():
                        continue
                self.records.append(row)
        
        print(f"Loaded {initial_count} total records. Skipped {initial_count - len(self.records)} existing outputs.")
        print(f"Will process {len(self.records)} images.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        canonical_id = row['canonical_id']
        image_path_raw = row['absolute_image_path']
        local_path = None

        # Resolve local path
        if self.image_root:
            if image_path_raw.startswith('s3://'):
                try:
                    relative_path = image_path_raw.split('/', 3)[3]
                    local_path = self.image_root / relative_path
                except IndexError:
                    return None, {'canonical_id': canonical_id, 'status': 'error', 'error': 'Invalid S3 URI'}
            else:
                local_path = self.image_root / image_path_raw
        else:
            local_path = Path(image_path_raw)

        # Load image
        try:
            if not local_path or not local_path.exists():
                return None, {'canonical_id': canonical_id, 'status': 'error', 'error': f"Image not found: {local_path}"}
            
            img = Image.open(local_path).convert('RGB')
            return img, {
                'canonical_id': canonical_id, 
                'status': 'success'
            }
        except Exception as e:
            return None, {
                'canonical_id': canonical_id, 
                'status': 'error', 
                'error': str(e)
            }

def save_output(out_data, output_dir):
    """Saves caption data to a structured JSON file."""
    canonical_id = out_data['canonical_id']
    out_path = Path(output_dir) / f"{canonical_id}_caption.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Manifest-driven VLM Captioning (Local GPU).')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--output-dir', required=True, help='Root directory to save caption JSON files')
    parser.add_argument('--image-root', required=True, help='Local root folder where images are stored')
    parser.add_argument('--model', type=str, default='microsoft/Florence-2-large', help='HuggingFace model ID')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size. Florence-2-large fits ~4-6 on 16GB VRAM.')
    parser.add_argument('--workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--prompt', type=str, default='<MORE_DETAILED_CAPTION>', help='Task prompt for Florence-2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Initialize Model ---
    print(f"Loading model: {args.model}")
    try:
        # Load processor and model
        # trust_remote_code=True is required for Florence-2
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
        
        # Optimize for inference
        if device.type == 'cuda':
            model = model.half() # Use FP16 for speed/memory
            
        model.eval()
    except Exception as e:
        print(f"Fatal: Could not load model. Error: {e}")
        return

    # --- Dataset & Loader ---
    dataset = ManifestDataset(
        manifest_path=args.manifest, 
        image_root=args.image_root, 
        output_dir=args.output_dir
    )
    
    if len(dataset) == 0:
        print("All images in the manifest have already been processed.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    # --- Inference Loop ---
    print(f"Starting captioning for {len(dataset)} images...")
    
    total_processed = 0
    total_errors = 0
    # Note: 'skipped' are filtered out during dataset initialization, 
    # but we can track failures/successes during this loop.
    
    pbar = tqdm(dataloader, desc="Captioning Pages")
    with torch.no_grad():
        for batch in pbar:
            # 1. Filter valid images
            valid_imgs = []
            valid_infos = []
            
            for img, info in batch:
                if info and info['status'] == 'success' and img is not None:
                    valid_imgs.append(img)
                    valid_infos.append(info)
                elif info and info['status'] == 'error':
                    save_output({'canonical_id': info['canonical_id'], 'error': info['error']}, args.output_dir)
                    total_errors += 1

            if not valid_imgs:
                pbar.set_description(f"Success: {total_processed} | Errors: {total_errors}")
                continue

            # 2. Prepare Inputs
            try:
                # Florence-2 batch processing
                # We repeat the text prompt for every image in the batch
                texts = [args.prompt] * len(valid_imgs)
                
                inputs = processor(text=texts, images=valid_imgs, return_tensors="pt", padding=True).to(device, torch.float16 if device.type == 'cuda' else torch.float32)

                # 3. Generate
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True
                )

                # 4. Decode
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
                # 5. Post-process & Save
                for i, text in enumerate(generated_texts):
                    info = valid_infos[i]
                    canonical_id = info['canonical_id']
                    
                    # Florence-2 post-processing to parse the task result
                    parsed_answer = processor.post_process_generation(
                        text, 
                        task=args.prompt, 
                        image_size=(valid_imgs[i].width, valid_imgs[i].height)
                    )
                    
                    # The result is usually a dictionary, e.g. {'<MORE_DETAILED_CAPTION>': 'The image shows...'}
                    # We extract the value to save cleaner JSON
                    caption_text = parsed_answer.get(args.prompt, text)

                    out_data = {
                        'canonical_id': canonical_id,
                        'model': args.model,
                        'task': args.prompt,
                        'caption': caption_text
                    }
                    save_output(out_data, args.output_dir)
                    total_processed += 1
                
                # Update progress bar description
                pbar.set_description(f"Success: {total_processed} | Errors: {total_errors}")

            except Exception as e:
                print(f"Batch inference failed: {e}")
                # Save error for all in batch
                for info in valid_infos:
                    save_output({'canonical_id': info['canonical_id'], 'error': f"Inference failed: {str(e)}"}, args.output_dir)
                    total_errors += 1
                pbar.set_description(f"Success: {total_processed} | Errors: {total_errors}")
                continue

    print(f"\n--- Captioning Complete ---")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
