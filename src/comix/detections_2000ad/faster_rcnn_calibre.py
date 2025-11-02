import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image, ImageFile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from comix.utils import get_image_id, add_image, add_annotation, COCO_OUTPUT

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_fn(batch):
    return tuple(zip(*batch))

class SimpleDataset(Dataset):
    """Simple dataset for loading images from a directory."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.book_chapters = []
        
        # Walk through the directory structure
        count = 0
        # Define supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # Only process files with image extensions
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    self.image_paths.append(os.path.join(root, file))
                    self.book_chapters.append(root)

        #for book_chapter_dir in sorted(self.root_dir.iterdir()):
            #if book_chapter_dir.is_dir():
                #images = sorted([f for f in book_chapter_dir.iterdir() if f.suffix.lower() == '.jpg'])
                #self.image_paths.extend(images)
                #self.book_chapters.extend([book_chapter_dir.name] * len(images))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        book_chapter = self.book_chapters[idx]
        #page_no = self.image_paths[idx].stem  # Get the filename without extension
        page_no = self.image_paths[idx].split('\\')[-1]  # Get the filename without extension
        
        # Load and transform image with error handling
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, (img_path, book_chapter, page_no)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image or skip this file
            # For now, we'll create a black image as placeholder
            img = Image.new('RGB', (1024, 1024), color='black')
            if self.transform:
                img = self.transform(img)
            return img, (img_path, book_chapter, page_no)

def get_transform():
    transforms = []
    transforms.append(T.Resize((1024, 1024)))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_model(num_classes=5):  # 4 classes + background
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser(description='Run Faster R-CNN detection on 2000AD dataset.')
    parser.add_argument('--input-path', type=str, default='data/datasets.unify/2000ad/images',
                       help='Path to the unified 2000AD images')
    parser.add_argument('--output-path', type=str, default='data/predicts.coco/2000ad/faster-rcnn',
                       help='Path to save prediction COCO JSON file')
    parser.add_argument('--weights-path', type=str, default='benchmarks/weights/fasterrcnn',
                       help='Path to the Faster R-CNN weights')
    parser.add_argument('--weights-name', type=str, default='faster_rcnn-c100-best-10052024_092536',
                       help='Name of the weights file (without .pth extension)')
    parser.add_argument('--save-vis', type=int, default=None,
                       help='Number of visualization images to save')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup dataset and dataloader
    dataset = SimpleDataset(args.input_path, transform=get_transform())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                           collate_fn=collate_fn)

    # Load model
    model = get_model()
    model.load_state_dict(torch.load(f'{args.weights_path}/{args.weights_name}.pth'))
    model.to(device)
    model.eval()

    # Class mappings
    CLS_MAPPING = {
        0: 1,  # panel
        1: 2,  # character
        2: 4,  # text
        3: 7   # face
    }

    # For visualizations
    CLS2COLOR = {
        1: 'green',    # panel
        2: 'red',      # character
        4: 'blue',     # text
        7: 'magenta'   # face
    }

    # Setup visualization directory if needed
    if args.save_vis:
        save_path = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(save_path, exist_ok=True)

    # Initialize COCO format output
    coco_output = COCO_OUTPUT
    annotation_id = 0

    # Run inference
    with torch.no_grad():
        for batch_idx, (batch_imgs, batch_info) in enumerate(tqdm(data_loader)):
            # Move images to device
            batch_imgs = list(img.to(device) for img in batch_imgs)
            
            # Get predictions
            batch_results = model(batch_imgs)
            batch_results = [{k: v.to('cpu') for k, v in t.items()} for t in batch_results]

            # Process each image in the batch
            for results, (img_path, book_chapter, page_no) in zip(batch_results, batch_info):
                image_id = f"{book_chapter}_{page_no}"
                coco_output = add_image(coco_output, image_id, img_path)
                img_size = Image.open(img_path).size

                # Filter by confidence
                mask = results['scores'] > args.conf_threshold
                boxes = results['boxes'][mask]
                scores = results['scores'][mask]
                labels = results['labels'][mask]

                # Add detections to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    cls = label.item() - 1  # Convert to 0-based indexing
                    
                    # Convert to COCO format [x,y,width,height]
                    bbox = [
                        int(x1 * img_size[0] / 1024),
                        int(y1 * img_size[1] / 1024),
                        int((x2 - x1) * img_size[0] / 1024),
                        int((y2 - y1) * img_size[1] / 1024)
                    ]
                    
                    coco_output = add_annotation(
                        coco_output, annotation_id, image_id,
                        CLS_MAPPING[cls], bbox, score.item()
                    )
                    annotation_id += 1

                # Save visualization if requested
                if args.save_vis and batch_idx < args.save_vis:
                    print("vis for image path", img_path)
                    print(f"Number of detections before filtering: {len(results['boxes'])}")
                    print(f"Number of detections after filtering (conf > {args.conf_threshold}): {len(boxes)}")
                    print(f"Max confidence score: {results['scores'].max().item():.3f}")
                    
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                    ax.imshow(Image.open(img_path))
                    
                    # Use the filtered boxes and labels (above confidence threshold)
                    for box, label in zip(boxes, labels):
                        x1, y1, x2, y2 = box.tolist()
                        cls = label.item() - 1
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Scale coordinates to original image size
                        x1 = x1 * img_size[0] / 1024
                        y1 = y1 * img_size[1] / 1024
                        w = w * img_size[0] / 1024
                        h = h * img_size[1] / 1024
                        
                        rect = Rectangle(
                            (x1, y1), w, h,
                            linewidth=2,
                            edgecolor=CLS2COLOR[CLS_MAPPING[cls]],
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                    
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, f"{image_id}.png"), 
                              bbox_inches='tight', pad_inches=0)
                    plt.close()

    # Save results
    print("output path:",args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(coco_output, f)

    print(f'Output saved to {args.output_path}')
    if args.save_vis:
        print(f'Visualizations saved to {save_path}')

if __name__ == "__main__":
    main() 