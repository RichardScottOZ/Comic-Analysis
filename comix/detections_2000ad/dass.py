import os
import copy
import cv2
import json
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from comix.utils import get_image_id, add_image, add_annotation, COCO_OUTPUT
from modules.DASS_Det_Inference.dass_det.models.yolox import YOLOX
from modules.DASS_Det_Inference.dass_det.models.yolo_head import YOLOXHead
from modules.DASS_Det_Inference.dass_det.models.yolo_head_stem import YOLOXHeadStem
from modules.DASS_Det_Inference.dass_det.models.yolo_pafpn import YOLOPAFPN
from modules.DASS_Det_Inference.dass_det.data.data_augment import ValTransform
from modules.DASS_Det_Inference.dass_det.utils import postprocess, vis
import torch
import torch.serialization
import numpy as np
from pathlib import Path

class SimpleDataset(Dataset):
    """Simple dataset for loading images from a directory."""
    def __init__(self, root_dir, transform=None, input_size=(1024, 1024)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.input_size = input_size
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
        
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # BGR to RGB
        
        if self.transform:
            img, _ = self.transform(img, None, self.input_size)
        
        return img, (book_chapter, page_no)

def main():
    parser = argparse.ArgumentParser(description='Run DASS detection on 2000AD dataset.')
    parser.add_argument('--input-path', type=str, default='data/datasets.unify/2000ad/images',
                       help='Path to the unified 2000AD images')
    parser.add_argument('--output-path', type=str, default='data/predicts.coco/2000ad/dass-m109',
                       help='Path to save prediction COCO JSON file')
    parser.add_argument('--model-size', type=str, default='xl', choices=['xs', 'xl'],
                       help='Model size')
    parser.add_argument('--weights-path', type=str, default='benchmarks/weights/dass',
                       help='Path to the DASS model weights')
    parser.add_argument('--save-vis', type=int, default=None,
                       help='Number of visualization images to save')
    args = parser.parse_args()

    # Dataset setup
    val_transform = ValTransform()
    resize_size = (1024, 1024)
    dataset = SimpleDataset(args.input_path, transform=val_transform, input_size=resize_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model setup
    model_path = f"{args.weights_path}/{args.model_size}_m109_finetuned_stage3.pth"
    nms_thold = 0.4
    conf_thold = 0.65

    if args.model_size == "xs":
        depth, width = 0.33, 0.375
    elif args.model_size == "xl":
        depth, width = 1.33, 1.25

    # Initialize model
    model = YOLOX(backbone=YOLOPAFPN(depth=depth, width=width),
                  head_stem=YOLOXHeadStem(width=width),
                  face_head=YOLOXHead(1, width=width),
                  body_head=YOLOXHead(1, width=width))

    # Add safe globals for numpy scalar
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

    # Load model weights
    d = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if "teacher_model" in d.keys():
        model.load_state_dict(d["teacher_model"])
    else:
        model.load_state_dict(d["model"])
    model = model.eval().cuda()
    del d

    # Constants for detection
    MODE = 0  # 0 for both face and body detection
    FACE_INDEX = 7
    BODY_INDEX = 2
    annotation_id = 1

    # Setup visualization directory if needed
    if args.save_vis:
        save_path = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(save_path, exist_ok=True)

    # Initialize COCO format output
    coco_output = COCO_OUTPUT

    # Run inference
    model = model.eval()
    with torch.no_grad():
        for b, (batch_imgs, (book_chapter, page_no)) in enumerate(tqdm(data_loader)):
            # Move batch to GPU
            batch_imgs = batch_imgs.cuda().float()
            face_preds, body_preds = model(batch_imgs, mode=MODE)
            face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)
            body_preds = postprocess(body_preds, 1, conf_thold, nms_thold)

            # Process predictions
            for i in range(len(batch_imgs)):
                image_id = f"{book_chapter[i]}_{page_no[i]}"
                img_path = str(dataset.image_paths[b * data_loader.batch_size + i])
                coco_output = add_image(coco_output, image_id, img_path)

                # Handle face predictions
                if face_preds[i] is not None:
                    face_preds_i = face_preds[i].cpu()
                else:
                    face_preds_i = torch.empty(0, 5)

                # Handle body predictions
                if body_preds[i] is not None:
                    body_preds_i = body_preds[i].cpu()
                else:
                    body_preds_i = torch.empty(0, 5)

                # Combine predictions
                preds = torch.cat([face_preds_i, body_preds_i], dim=0)
                classes = torch.cat([FACE_INDEX*torch.ones(face_preds_i.shape[0]), 
                                  BODY_INDEX*torch.ones(body_preds_i.shape[0])])

                # Add annotations
                for pred, cls in zip(preds, classes):
                    x1, y1, x2, y2, score = pred.numpy()
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # Convert to [x,y,w,h]
                    coco_output = add_annotation(coco_output, annotation_id, image_id, 
                                              int(cls.item()), bbox, score.item())
                    annotation_id += 1

                # Save visualization if requested
                if args.save_vis and b < args.save_vis:
                    img = cv2.imread(img_path)[:,:,::-1]
                    fake_classes = classes.clone()
                    fake_classes[classes == BODY_INDEX] = 0
                    fake_classes[classes == FACE_INDEX] = 1
                    vis_img = Image.fromarray(vis(copy.deepcopy(img), preds[:,:4], 
                                                preds[:,4], fake_classes, conf=0.0, 
                                                class_names=["Body", "Face"]))
                    vis_img.save(os.path.join(save_path, f"{image_id}.png"))

    # Create output directory and save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(coco_output, f)

    print(f'Output saved to {args.output_path}')

if __name__ == "__main__":
    main() 