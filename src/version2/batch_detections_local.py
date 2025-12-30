#!/usr/bin/env python3
"""
Manifest-driven Faster R-CNN detection (Local GPU)

This script runs Faster R-CNN detections locally on a GPU using a manifest file.
It outputs individual JSON files for each page, mirroring the folder structure of the canonical_id.

Usage:
    python src/version2/batch_detections_local.py \
        --manifest manifests/calibrecomics-extracted_manifest.csv \
        --output-dir E:/CalibreComics_detections \
        --weights path/to/your/weights.pth \
        --image-root E:/CalibreComics_extracted \
        --batch-size 8
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
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

# --- Global Settings ---
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_fn(batch):
    """Custom collate function for object detection."""
    return tuple(zip(*batch))

class ManifestDataset(Dataset):
    """Dataset loading images defined in a manifest CSV, skipping existing outputs."""
    def __init__(self, manifest_path, image_root=None, transform=None, output_dir=None):
        self.transform = transform
        self.image_root = Path(image_root) if image_root else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.records = []

        print(f"Loading manifest from: {manifest_path}")
        initial_count = 0
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                initial_count += 1
                # Skip if output already exists (canonical_id.json)
                if self.output_dir:
                    out_path = self.output_dir / f"{row['canonical_id']}.json"
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

        # Logic for resolving the local path
        if self.image_root:
            # If image_root provided, try to resolve S3 or relative paths against it
            if image_path_raw.startswith('s3://'):
                try:
                    relative_path = image_path_raw.split('/', 3)[3]
                    local_path = self.image_root / relative_path
                except IndexError:
                    return None, {'canonical_id': canonical_id, 'status': 'error', 'error': 'Invalid S3 URI'}
            else:
                local_path = self.image_root / image_path_raw
        else:
            # If no image_root, assume the path in manifest is already correct (local absolute or relative)
            if image_path_raw.startswith('s3://'):
                 return None, {'canonical_id': canonical_id, 'status': 'error', 'error': 'Cannot process S3 URI without --image-root'}
            local_path = Path(image_path_raw)

        try:
            if not local_path or not local_path.exists():
                # raise FileNotFoundError(f"Image not found at resolved path: {local_path}")
                return None, {'canonical_id': canonical_id, 'status': 'error', 'error': f"Image not found: {local_path}"}
            
            img = Image.open(local_path).convert('RGB')
            original_size = img.size  # (width, height)
            if self.transform:
                img = self.transform(img)
            return img, {
                'canonical_id': canonical_id, 
                'original_size': original_size,
                'status': 'success'
            }
        except Exception as e:
            # Return placeholder and error info if loading fails
            img = torch.zeros((3, 1024, 1024), dtype=torch.float32)
            return img, {
                'canonical_id': canonical_id, 
                'status': 'error', 
                'error': str(e)
            }

def get_transform():
    """Returns the required torchvision transforms for the model."""
    transforms = []
    transforms.append(T.Resize((1024, 1024), antialias=True))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)

def get_model(num_classes=5):
    """Loads a pre-trained Faster R-CNN model and replaces the classifier head."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def save_output(out_data, output_dir):
    """Saves detection data to a structured JSON file."""
    canonical_id = out_data['canonical_id']
    out_path = Path(output_dir) / f"{canonical_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Manifest-driven Faster R-CNN detection (Local GPU).')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV file')
    parser.add_argument('--output-dir', required=True, help='Root directory to save JSON files')
    parser.add_argument('--weights', default='C:\\Users\\Richard\\OneDrive\\GIT\\CoMix\\benchmarks\\weights\\fasterrcnn\\faster_rcnn-c100-best-10052024_092536.pth', help='Path to the .pth weights file')
    parser.add_argument('--image-root', required=False, help='Local root folder (optional).')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for GPU inference.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for filtering detections.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, processing will be very slow.")

    # --- Initialize Model ---
    print(f"Loading weights from {args.weights}")
    try:
        model = get_model()
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Fatal: Could not load model. Error: {e}")
        return

    # --- Dataset & Loader ---
    dataset = ManifestDataset(
        manifest_path=args.manifest, 
        image_root=args.image_root, 
        transform=get_transform(),
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
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- Class Mappings ---
    # COCO-style mapping (0 is background, not in this dict)
    CLS_MAPPING = {
        1: 'panel',
        2: 'character',
        3: 'text',
        4: 'face'
    }

    # --- Inference Loop ---
    print(f"Starting detection processing for {len(dataset)} images...")
    
    with torch.no_grad():
        for batch_imgs, batch_infos in tqdm(dataloader, desc="Detecting Panels"):
            # Filter out failed loads (where info['status'] == 'error')
            valid_batch_imgs = []
            valid_batch_infos = []
            
            # Since collate_fn zips them, batch_imgs is a tuple of images, batch_infos is a tuple of dicts
            for img, info in zip(batch_imgs, batch_infos):
                if info and info['status'] == 'success':
                    valid_batch_imgs.append(img)
                    valid_batch_infos.append(info)
                elif info and info['status'] == 'error':
                    # Log error immediately
                    save_output({'canonical_id': info['canonical_id'], 'error': info['error']}, args.output_dir)

            if not valid_batch_imgs:
                continue

            # --- Inference ---
            try:
                gpu_imgs = [img.to(device) for img in valid_batch_imgs]
                batch_results = model(gpu_imgs)
            except Exception as e:
                print(f"Batch inference failed: {e}")
                # Log batch failure for all images in this valid batch
                for info in valid_batch_infos:
                    save_output({'canonical_id': info['canonical_id'], 'error': f'Batch inference failed: {e}'}, args.output_dir)
                continue

            # --- Process & Save Results ---
            for i, result_cpu in enumerate(batch_results):
                info = valid_batch_infos[i]
                canonical_id = info['canonical_id']
                
                # Filter by confidence
                mask = result_cpu['scores'].cpu() > args.conf_threshold
                boxes = result_cpu['boxes'].cpu()[mask]
                scores = result_cpu['scores'].cpu()[mask]
                labels = result_cpu['labels'].cpu()[mask]
                
                orig_w, orig_h = info['original_size']
                
                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Scale box from resized (1024x1024) to original dimensions
                    scale_x = orig_w / 1024.0
                    scale_y = orig_h / 1024.0
                    
                    real_box = [
                        float(x1 * scale_x), float(y1 * scale_y),
                        float(x2 * scale_x), float(y2 * scale_y)
                    ]
                    
                    # Map COCO label index to class name
                    cls_name = CLS_MAPPING.get(label.item(), 'unknown')

                    detections.append({
                        'label': cls_name,
                        'score': float(score),
                        'box_xyxy': real_box
                    })

                out_data = {
                    'canonical_id': canonical_id,
                    'image_size_wh': [orig_w, orig_h],
                    'detections': detections
                }
                save_output(out_data, args.output_dir)

    print(f"\n--- Detection Complete ---")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
