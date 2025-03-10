#!/usr/bin/env python3
"""
Generate captions for 2000AD comic panels using various models.

This script processes comic panels from 2000AD comics using panel annotations
and generates captions using the specified model.

Usage:
    python generate_captions.py --model florence2 --input-dir data/datasets.unify/2000ad/images --output-dir data/predicts.captions/2000ad --batch-size 64 [--save-txt] [--save-csv]
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import tempfile
import multiprocessing
import numpy as np

# Import the model-specific code from the original generate_captions.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioning.generate_captions import (
    Florence2Model, MiniCPMModel, Qwen2VLModel, Qwen2VLModelQuant, Idefics2Model, Idefics3Model,
    extract_caption, extract_list, base_prompt
)

class Panel2000ADDataset(Dataset):
    """Dataset for 2000AD Comic Panels based on compiled CSV annotations."""
    def __init__(self, root_dir, annotations_df, transform=None, config=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.annotations = annotations_df
        
        # Load hash mapping
        self.hash_mapping = self._load_hash_mapping()
        
        # Print dataset initialization info
        print(f"\nDataset initialization:")
        print(f"Root directory: {self.root_dir}")
        print(f"Number of annotations: {len(self.annotations)}")
        print(f"Number of unique hashes: {len(set(self.hash_mapping.values()))}")
        print(f"Sample paths from annotations:")
        for _, row in self.annotations.head().iterrows():
            # Extract comic number (e.g., PRG1795)
            comic_no = str(row['comic_no'])
            if '/' in comic_no:
                comic_no = comic_no.split('/')[-1]
            
            # Find matching hash from mapping
            hash_id = None
            for book_name, hash_val in self.hash_mapping.items():
                if book_name in comic_no or comic_no in book_name:
                    hash_id = hash_val
                    break
            
            if hash_id:
                print(f"  Original: {row['subdb']}/{row['comic_no']}/{int(row['page_no']):03d}.jpg")
                print(f"  Mapped: {hash_id}/{int(row['page_no']):03d}.jpg")
            else:
                print(f"  Warning: No hash mapping found for comic {comic_no}")
        
        if config and not config.get('override', False):
            self.annotations = self.remove_done_panels(self.annotations, config)
    
    def _load_hash_mapping(self):
        """Load the hash mapping from the CSV file."""
        mapping_file = Path('data/datasets.unify/2000ad/book_chapter_hash_mapping.csv')
        if not mapping_file.exists():
            print(f"Warning: Hash mapping file not found at {mapping_file}")
            return {}
        
        mapping = {}
        try:
            df = pd.read_csv(mapping_file)
            # Group by book_name to get unique hash for each book
            for book_name, group in df.groupby('book_name'):
                mapping[book_name] = group['book_chapter_hash'].iloc[0]
            print(f"Loaded {len(mapping)} book-to-hash mappings")
            return mapping
        except Exception as e:
            print(f"Error loading hash mapping: {e}")
            return {}
    
    def remove_done_panels(self, panels, config):
        """Remove panels that have already been processed."""
        caption_csv = config.get('caption_csv')
        list_csv = config.get('list_csv')
        
        processed = set()
        for file_name in [caption_csv, list_csv]:
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)
                key_tuples = zip(df['subdb'], df['comic_no'], df['page_no'], df['panel_no'])
                processed.update(key_tuples)
        
        # Create a tuple for each row in the panels DataFrame
        panels['key'] = list(zip(panels['subdb'], panels['comic_no'], panels['page_no'], panels['panel_no']))
        original_count = len(panels)
        panels = panels[~panels['key'].isin(processed)].copy()
        panels.drop(columns=['key'], inplace=True)
        
        print(f'Removed {original_count - len(panels)} already processed panels.')
        return panels
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.annotations.iloc[idx]

        #print(row)
        
        # Extract comic number (e.g., PRG1795)
        comic_no = str(row['comic_no'])
        if '/' in comic_no:
            comic_no = comic_no.split('/')[-1]
        
        # Find matching hash from mapping
        hash_id = None
        for book_name, hash_val in self.hash_mapping.items():
            if book_name in comic_no or comic_no in book_name:
                hash_id = hash_val
                break
        
        if not hash_id and 1 ==2:
            print(f"Warning: No hash mapping found for comic {comic_no}")
            blank_img = Image.new('RGB', (224, 224))
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, {
                'subdb': row['subdb'],
                'comic_no': row['comic_no'],
                'page_no': str(int(row['page_no'])),
                'panel_no': row['panel_no']
            }
        
        hash_id = row['comic_no']
        # Construct image path using hash mapping
        usepage = str(row['page_no'])
        page_no = usepage
        if len(usepage) == 1:
            page_no = '00' + usepage
        if len(usepage) == 2:
            page_no = '0' + usepage

        #page_path = self.root_dir / hash_id / f"{int(row['page_no']):03d}.jpg"
        page_path = self.root_dir / hash_id / f"{page_no}.jpg"
        #print("PAGEPATH",page_path)

        
        if not page_path.exists():
            print(f"Warning: Image not found at {page_path}")
            blank_img = Image.new('RGB', (224, 224))
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, {
                'subdb': row['subdb'],
                'comic_no': row['comic_no'],
                'page_no': str(int(row['page_no'])),
                'panel_no': row['panel_no']
            }
        
        # Open the page image
        try:
            page_img = Image.open(page_path).convert('RGB')
            
            # Extract the panel using bounding box
            x1, y1, x2, y2 = map(float, [row['x1'], row['y1'], row['x2'], row['y2']])
            panel = page_img.crop((x1, y1, x2, y2))

            if 1 == 2:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle

                # Debug visualization
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                # Plot original image with bounding box
                ax1.imshow(page_img)
                ax1.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
                ax1.set_title('Original Image with Panel Box')
                
                # Plot cropped panel
                ax2.imshow(panel)
                ax2.set_title('Cropped Panel')
                
                # Save the debug visualization
                #debug_dir = Path('debug_panels')
                #debug_dir.mkdir(exist_ok=True)
                #plt.savefig(debug_dir / f"{hash_id}_{int(row['page_no']):03d}_panel{row['panel_no']}.png")
                plt.show()
                plt.close()
                        
            
            if self.transform:
                panel = self.transform(panel)
            
            return panel, {
                'subdb': row['subdb'],
                'comic_no': row['comic_no'],
                'page_no': str(row['page_no']),
                'panel_no': row['panel_no']
            }
        except Exception as e:
            print(f"Error processing image {page_path}: {e}")
            blank_img = Image.new('RGB', (224, 224))
            if self.transform:
                blank_img = self.transform(blank_img)
            return blank_img, {
                'subdb': row['subdb'],
                'comic_no': row['comic_no'],
                'page_no': str(row['page_no']),
                'panel_no': row['panel_no']
            }

def collate_panels(batch):
    """Collate function for the dataloader that handles panels and their info."""
    return [item[0] for item in batch], [item[1] for item in batch]

def pil_collate(batch):
    """Custom collate function to handle batches of PIL Images and their info."""
    imgs, infos = zip(*batch)
    return list(imgs), list(infos)


def main():
    parser = argparse.ArgumentParser(description='Generate captions for 2000AD comic panels.')
    parser.add_argument('--model', type=str, required=True,
                       choices=['minicpm2.6', 'qwen2','qwen2quant', 'florence2', 'idefics2', 'idefics3'],
                       help='Model to use for captioning')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Root directory containing the comic images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the caption outputs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save raw results as txt files')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save results to CSV files')
    parser.add_argument('--override', action='store_true',
                       help='Override existing processed files')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    args = parser.parse_args()
    
    # Load panel annotations
    annotations_file = Path(r'C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\compiled_panels_annotations.csv')
    if not annotations_file.exists():
        raise ValueError(f"Panel annotations file not found: {annotations_file}")
    
    annotations_df = pd.read_csv(annotations_file)
    print(f"Loaded {len(annotations_df)} panel annotations")
    
    # Print sample of annotations to verify structure
    print("\nSample of annotations:")
    print(annotations_df.head())
    print("\nAnnotations columns:", annotations_df.columns.tolist())
    
    # Setup output directories
    output_dir = Path(args.output_dir) / f'{args.model}-cap'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    if args.model == 'florence2':
        model = Florence2Model()
    elif args.model == 'minicpm2.6':
        model = MiniCPMModel()
    elif args.model == 'qwen2':
        model = Qwen2VLModel()
    elif args.model == 'qwen2quant':
        model = Qwen2VLModelQuant()
    elif args.model == 'idefics2':
        model = Idefics2Model()
    elif args.model == 'idefics3':
        model = Idefics3Model()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    model.initialize()
    
    # Move model to GPU and set data type
    if torch.cuda.is_available():
        print("\nMoving model to GPU...")
        # First convert model to half precision
        if not 'qwen2quant' in args.model:
            model.model = model.model.half()
        # Then move to GPU
        model.model = model.model.to('cuda')
        
        # Configure processor settings
        if hasattr(model, 'processor'):
            if hasattr(model.processor, 'image_processor'):
                model.processor.image_processor.do_rescale = False  # Don't rescale since we'll pass PIL images
                model.processor.image_processor.do_convert_rgb = True  # Keep RGB conversion
                # Ensure processor uses same dtype as model
                if hasattr(model.processor, 'image_mean') and hasattr(model.processor, 'image_std'):
                    model.processor.image_mean = torch.tensor(model.processor.image_mean, dtype=torch.float16, device='cuda')
                    model.processor.image_std = torch.tensor(model.processor.image_std, dtype=torch.float16, device='cuda')
        print("Model moved to GPU and converted to float16")
    
    # Setup dataset and dataloader
    dataset = Panel2000ADDataset(
        root_dir=args.input_dir,
        annotations_df=annotations_df,
        transform=None,  # Don't use transform, let model processor handle it
        config={
            'override': args.override,
            'caption_csv': output_dir / 'captions.csv',
            'list_csv': output_dir / 'lists.csv'
        }
    )
    
    # Simple collate function that passes PIL images directly
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pil_collate
    )
    
    # Setup output files
    if args.save_csv:
        caption_file = open(output_dir / 'captions.csv', 'a', newline='', encoding='utf-8')
        list_file = open(output_dir / 'lists.csv', 'a', newline='', encoding='utf-8')
        caption_writer = csv.writer(caption_file)
        list_writer = csv.writer(list_file)
        
        # Write headers if files are empty
        if caption_file.tell() == 0:
            caption_writer.writerow(['subdb', 'comic_no', 'page_no', 'panel_no', 'caption'])
        if list_file.tell() == 0:
            list_writer.writerow(['subdb', 'comic_no', 'page_no', 'panel_no', 'items'])
    
    # Process panels
    with torch.no_grad():
        for batch_imgs, batch_info in tqdm(dataloader, desc=f"Processing panels with {args.model}"):
            # Generate captions
            results = model.infer(batch_imgs, prompt=base_prompt)
            
            # Save results
            for idx, (result, info) in enumerate(zip(results, batch_info)):
                if args.save_txt:
                    # Save raw results
                    txt_path = output_dir / 'results' / f"{info['subdb']}_{info['comic_no']}_{info['page_no']}_{info['panel_no']}.txt"
                    txt_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(result)
                
                if args.save_csv:
                    # Extract and save structured results
                    caption = extract_caption(result)
                    items = extract_list(result)
                    
                    caption_writer.writerow([
                        info['subdb'], info['comic_no'], info['page_no'], info['panel_no'],
                        caption if caption else ""
                    ])
                    list_writer.writerow([
                        info['subdb'], info['comic_no'], info['page_no'], info['panel_no'],
                        ",".join(items) if items else ""
                    ])
    
    # Cleanup
    if args.save_csv:
        caption_file.close()
        list_file.close()
    
    print("\nProcessing complete!")
    if args.save_csv:
        print(f"Results saved to:")
        print(f"  Captions: {output_dir}/captions.csv")
        print(f"  Lists: {output_dir}/lists.csv")
    if args.save_txt:
        print(f"  Raw results: {output_dir}/results/")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main() 