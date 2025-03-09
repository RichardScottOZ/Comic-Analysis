import os
import json
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from comix.utils import add_image, add_annotation, COCO_OUTPUT, YOLO_TARGET_SIZE

# Disable gradient computation globally
torch.set_grad_enabled(False)

def collate_fn(batch):
    """Custom collate function to handle batches of PIL Images and their info."""
    imgs, infos = zip(*batch)
    return list(imgs), list(infos)

class SimpleDataset(Dataset):
    """Simple dataset for loading images from a directory."""
    def __init__(self, root_dir, target_size=YOLO_TARGET_SIZE):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.image_paths = []
        self.book_chapters = []
        
        # Walk through the directory structure
        for book_chapter_dir in sorted(self.root_dir.iterdir()):
            if book_chapter_dir.is_dir():
                images = sorted([f for f in book_chapter_dir.iterdir() if f.suffix.lower() == '.jpg'])
                self.image_paths.extend(images)
                self.book_chapters.extend([book_chapter_dir.name] * len(images))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        book_chapter = self.book_chapters[idx]
        page_no = self.image_paths[idx].stem  # Get the filename without extension
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        
        # Resize using LANCZOS for high-quality downsampling
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return img, (img_path, book_chapter, page_no)

def draw_detection(img, box, label, score, color):
    """Draw a single detection with OpenCV."""
    # Ensure coordinates are integers and within image bounds
    height, width = img.shape[:2]
    x1 = max(0, min(width-1, int(box[0])))
    y1 = max(0, min(height-1, int(box[1])))
    x2 = max(0, min(width-1, int(box[0] + box[2])))  # x + width
    y2 = max(0, min(height-1, int(box[1] + box[3])))  # y + height
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    text = f'{label}: {score:.2f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    
    # Draw text
    cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img

def draw_legend(img, classes, colors):
    """Draw legend with OpenCV."""
    padding = 10
    box_size = 20
    line_height = 25
    legend_width = 150
    legend_height = (len(classes) + 1) * line_height
    
    # Draw white background for legend
    cv2.rectangle(img, 
                 (img.shape[1] - legend_width - padding, padding),
                 (img.shape[1] - padding, padding + legend_height),
                 (255, 255, 255), -1)
    cv2.rectangle(img,
                 (img.shape[1] - legend_width - padding, padding),
                 (img.shape[1] - padding, padding + legend_height),
                 (0, 0, 0), 1)
    
    # Draw "Legend" text
    cv2.putText(img, "Legend:", 
                (img.shape[1] - legend_width + 5, padding + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Draw class entries
    for i, (cls, color) in enumerate(zip(classes, colors)):
        y = padding + (i + 1) * line_height
        # Draw color box
        cv2.rectangle(img,
                     (img.shape[1] - legend_width + 5, y),
                     (img.shape[1] - legend_width + box_size, y + box_size),
                     color, 2)
        # Draw class name
        cv2.putText(img, cls,
                    (img.shape[1] - legend_width + box_size + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 detection on 2000AD dataset.')
    parser.add_argument('--input-path', type=str, default='data/datasets.unify/2000ad/images',
                       help='Path to the unified 2000AD images')
    parser.add_argument('--output-path', type=str, default='data/predicts.coco/2000ad/yolo-mix/predictions.json',
                       help='Path to save prediction COCO JSON file')
    parser.add_argument('--weights-path', type=str, default='benchmarks/weights/yolov8',
                       help='Path to the YOLOv8 model weights')
    parser.add_argument('--weights-name', type=str, default='yolov8x-mix',
                       help='Name of the weights file (without .pt extension)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--save-vis', type=int, default=None,
                       help='Number of visualization images to save')
    args = parser.parse_args()

    print("\nInitializing YOLOv8 detection...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup dataset and dataloader
    dataset = SimpleDataset(args.input_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(data_loader)}")

    # Initialize model
    print("\nLoading YOLOv8 model...")
    model_path = os.path.join(args.weights_path, f"{args.weights_name}.pt")
    model = YOLO(model_path)
    print("Model loaded successfully")

    # Class mappings and colors for visualization
    CLS_MAPPING = {
        0: 1,  # panel
        1: 2,  # character
        2: 4,  # text
        3: 7   # face
    }

    CLS2COLOR = {
        'panel': (0, 255, 0),    # Green
        'character': (0, 0, 255), # Red
        'text': (255, 0, 0),     # Blue
        'face': (255, 0, 255)    # Magenta
    }

    # Setup visualization directory if needed
    if args.save_vis:
        save_path = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(save_path, exist_ok=True)

    # Initialize COCO format output
    coco_output = COCO_OUTPUT
    annotation_id = 1
    total_visualized = 0  # Keep track of total visualized images

    # Run inference
    print("\nStarting inference...")
    with torch.no_grad():
        for batch_idx, (batch_imgs, batch_info) in enumerate(tqdm(data_loader)):
            print(f"\nProcessing batch {batch_idx + 1}/{len(data_loader)}")
            
            # Get predictions
            results = model(batch_imgs, verbose=False)

            # Process each image in the batch
            for i, info in enumerate(batch_info):
                img_path, book_chapter, page_no = info  # Correctly unpack the info tuple
                image_id = f"{book_chapter}_{page_no}"
                print(f"    Processing image: {image_id}")
                
                # Add image to COCO output
                coco_output = add_image(coco_output, image_id, img_path)
                image_size = Image.open(img_path).size
                
                result = results[i]
                boxes = result.boxes
                
                # Convert boxes to COCO format and add annotations
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                
                for box, class_idx, score in zip(xyxy, cls, conf):
                    x1, y1, x2, y2 = box
                    bbox = [
                        float(x1),
                        float(y1),
                        float(x2 - x1),  # width
                        float(y2 - y1)   # height
                    ]
                    
                    coco_output = add_annotation(
                        coco_output, annotation_id, image_id,
                        CLS_MAPPING[int(class_idx)], bbox, float(score)
                    )
                    annotation_id += 1

                # Save visualization if requested
                if args.save_vis and total_visualized < args.save_vis:
                    try:
                        print(f"      Saving visualization for {image_id}")
                        # Read image with OpenCV
                        img = cv2.imread(img_path)
                        
                        # Track which classes have valid detections
                        valid_classes = []
                        valid_colors = []
                        
                        # Map class indices to names
                        cls_names = {
                            0: 'panel',
                            1: 'character',
                            2: 'text',
                            3: 'face'
                        }
                        
                        # Draw detections
                        for box, class_idx, score in zip(xyxy, cls, conf):
                            cls_name = cls_names[int(class_idx)]
                            if cls_name not in valid_classes:
                                valid_classes.append(cls_name)
                                valid_colors.append(CLS2COLOR[cls_name])
                            
                            bbox = [
                                int(box[0]), int(box[1]),
                                int(box[2] - box[0]),  # width
                                int(box[3] - box[1])   # height
                            ]
                            
                            img = draw_detection(
                                img,
                                bbox,
                                cls_name,
                                score,
                                CLS2COLOR[cls_name]
                            )
                        
                        # Draw legend if we have any detections
                        if valid_classes:
                            img = draw_legend(img, valid_classes, valid_colors)
                        
                        # Save the image
                        save_path_img = os.path.join(save_path, f"{image_id}.png")
                        cv2.imwrite(save_path_img, img)
                        
                        total_visualized += 1
                        print(f"      Saved visualization {total_visualized}/{args.save_vis}: {image_id}")
                        
                    except Exception as e:
                        print(f"      Warning: Failed to save visualization for {image_id}: {str(e)}")
                        continue

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(coco_output, f)

    print(f'Output saved to {args.output_path}')
    if args.save_vis:
        print(f'Saved {total_visualized} visualizations to {save_path}')

if __name__ == "__main__":
    main() 