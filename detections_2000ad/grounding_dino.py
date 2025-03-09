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
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
from comix.utils import add_image, add_annotation, COCO_OUTPUT

# Disable gradient computation globally
torch.set_grad_enabled(False)

def collate_fn(batch):
    """Custom collate function to handle PIL images and metadata."""
    images = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    return images, list(zip(*metadata))

class SimpleDataset(Dataset):
    """Simple dataset for loading images from a directory."""
    def __init__(self, root_dir, target_size=(800, 800)):
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
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        return img, (img_path, book_chapter, page_no)

def process_batch(model, processor, batch, text_prompt, device, box_threshold, text_threshold):
    """Process a batch of images with the given text prompt."""
    with torch.no_grad():
        # Convert PIL images to pixel values if needed
        inputs = processor(
            images=batch,
            text=[text_prompt] * len(batch),
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1] for img in batch]).to(device)  # Convert (W,H) to (H,W)
        results = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    return results

def unify_results(batch_results, class_name):
    """Unify the results format."""
    new_batch = []
    for res in batch_results:
        # Extract scores and boxes directly from the results
        scores = res["scores"].detach().cpu().numpy()
        boxes = res["boxes"].detach().cpu().numpy()
        labels = [class_name] * len(scores)
        
        results = {
            'scores': scores,
            'labels': labels,
            'boxes': boxes
        }
        new_batch.append(results)
    return new_batch

def draw_detection(img, box, label, score, color):
    """Draw a single detection with OpenCV."""
    # Ensure coordinates are integers and within image bounds
    height, width = img.shape[:2]
    x1 = max(0, min(width-1, int(box[0])))
    y1 = max(0, min(height-1, int(box[1])))
    x2 = max(0, min(width-1, int(box[2])))
    y2 = max(0, min(height-1, int(box[3])))
    
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
    parser = argparse.ArgumentParser(description='Run Grounding DINO detection on 2000AD dataset.')
    parser.add_argument('--input-path', type=str, default='data/datasets.unify/2000ad/images',
                       help='Path to the unified 2000AD images')
    parser.add_argument('--output-path', type=str, default='data/predicts.coco/2000ad/grounding-dino/predictions.json',
                       help='Path to save prediction COCO JSON file')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--save-vis', type=int, default=None,
                       help='Number of visualization images to save')
    parser.add_argument('--box-threshold', type=float, default=0.3,
                       help='Box confidence threshold')
    parser.add_argument('--text-threshold', type=float, default=0.1,
                       help='Text confidence threshold')
    args = parser.parse_args()

    print("\nInitializing Grounding DINO detection...")
    
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

    # Initialize model and processor
    print("\nLoading Grounding DINO model and processor...")
    processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    model = model.to(device).eval()
    print("Model loaded successfully")

    # Text prompts for each class
    texts = {
        'panel': "panel . comic panel . frame .",
        'character': "character . person . boy . girl . student . woman . man . animal . human . individual.",
        'text': "text . script . writing . printed text . handwritten text .",
        'face': "face . facial expression ."
    }

    # Class mappings for COCO format
    CLS_MAPPING = {
        'panel': 1,
        'character': 2,
        'text': 4,
        'face': 7
    }

    # For visualizations
    CLS2COLOR = {
        'panel': 'green',
        'character': 'red',
        'text': 'blue',
        'face': 'magenta'
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
            
            # Process each class
            cls2results = {}
            for cls_str, text in texts.items():
                print(f"  Processing class: {cls_str}")
                results = process_batch(model, processor, batch_imgs, text, device, args.box_threshold, args.text_threshold)
                cls2results[cls_str] = unify_results(results, cls_str)

            # Process each image in the batch
            for i, (img_path, book_chapter, page_no) in enumerate(zip(*batch_info)):
                image_id = f"{book_chapter}_{page_no}"
                print(f"    Processing image: {image_id}")
                coco_output = add_image(coco_output, image_id, img_path)
                image_size = Image.open(img_path).size

                # Process detections for each class
                for cls_str, results in cls2results.items():
                    res_imm = results[i]
                    for bbox, score in zip(res_imm['boxes'], res_imm['scores']):
                        # Convert to COCO format [x,y,width,height]
                        x1, y1, x2, y2 = bbox.tolist()  # Convert numpy array to list
                        bbox_coco = [
                            float(x1),
                            float(y1),
                            float(x2 - x1),  # width
                            float(y2 - y1)   # height
                        ]
                        
                        coco_output = add_annotation(
                            coco_output, annotation_id, image_id,
                            CLS_MAPPING[cls_str], bbox_coco, float(score)
                        )
                        annotation_id += 1

                # Save visualization if requested and haven't reached the limit
                if args.save_vis and total_visualized < args.save_vis:
                    try:
                        print(f"      Saving visualization for {image_id}")
                        # Read image with OpenCV
                        img = cv2.imread(img_path)
                        
                        # Track which classes have valid detections
                        valid_classes = set()
                        
                        # Convert color names to BGR
                        color_map = {
                            'green': (0, 255, 0),
                            'red': (255, 0, 0),
                            'blue': (0, 0, 255),
                            'magenta': (255, 0, 255)
                        }
                        
                        # Draw detections for each class
                        for cls_str, results in cls2results.items():
                            res_imm = results[i]
                            boxes = res_imm['boxes']
                            scores = res_imm['scores']
                            
                            for bbox, score in zip(boxes, scores):
                                if score > args.box_threshold:
                                    valid_classes.add(cls_str)
                                    x1, y1, x2, y2 = bbox.tolist()  # Convert numpy array to list
                                    box_coords = [int(x1), int(y1), int(x2), int(y2)]
                                    print(f"        {cls_str}: score={score:.2f}, box={box_coords}")
                                    img = draw_detection(
                                        img,
                                        box_coords,
                                        cls_str,
                                        score,
                                        color_map[CLS2COLOR[cls_str]]
                                    )
                        
                        # Draw legend if we have any detections
                        if valid_classes:
                            valid_classes = sorted(list(valid_classes))
                            valid_colors = [color_map[CLS2COLOR[cls]] for cls in valid_classes]
                            img = draw_legend(img, valid_classes, valid_colors)
                        
                        # Convert back to BGR for saving
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
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