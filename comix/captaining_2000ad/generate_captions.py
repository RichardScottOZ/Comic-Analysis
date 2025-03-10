#!/usr/bin/env python3
"""
Create Panel Annotations CSV from Detection Results

This script converts panel detection results from various models into a unified
panel annotations CSV file that can be used by the captioning script.

Usage:
    python create_panel_annotations.py [--detector yolo-mix] [--confidence 0.5]
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class HashMapper:
    """Class to handle hash mapping and caching."""
    def __init__(self):
        self.hash_mapping = None
        self.debug_count = 0
    
    def load_mapping(self):
        """Load hash mapping from CSV file."""
        if self.hash_mapping is None:
            mapping_file = Path('data/datasets.unify/2000ad/book_chapter_hash_mapping.csv')
            if not mapping_file.exists():
                raise ValueError(f"Hash mapping file not found: {mapping_file}")
            
            print("\nLoading hash mapping file...")
            mapping_df = pd.read_csv(mapping_file)
            # Create mapping using book_chapter_hash as key and (book_name, chapter_name) as value
            self.hash_mapping = dict(zip(mapping_df['book_chapter_hash'], 
                                       zip(mapping_df['book_name'], mapping_df['chapter_name'])))
            print(f"Loaded {len(self.hash_mapping)} hash mappings")
            
            # Print some example mappings
            print("\nExample mappings:")
            for i, (hash_val, (book, chapter)) in enumerate(list(self.hash_mapping.items())[:5]):
                print(f"{hash_val} -> {book}, {chapter}")
    
    def get_info(self, image_path):
        """Get subdb and comic_no from hash."""
        # Print the first few paths to understand the format
        if self.debug_count < 5:
            print(f"Processing path: {image_path}")
            self.debug_count += 1
        
        # Ensure mapping is loaded
        self.load_mapping()
        
        try:
            # Handle different path formats
            path = Path(image_path)
            
            # Extract hash and page number
            if '\\' in image_path:
                # Windows path format: hash\number.jpg
                hash_part = path.parent.name
                page_no = path.stem
            elif '_' in image_path:
                # Hash_number format
                hash_part = image_path.split('_')[0]
                page_no = image_path.split('_')[1]
            else:
                # Try original path format as fallback
                parts = path.parts
                page_no = path.stem
                comic_no = parts[-2]
                subdb = parts[-3]
                return subdb, comic_no, page_no
            
            # Look up the hash
            if hash_part in self.hash_mapping:
                book_name, chapter_name = self.hash_mapping[hash_part]
                # Use book_name as comic_no and a fixed value for subdb
                return "2000ad", book_name, page_no
            else:
                print(f"\nHash {hash_part} not found in mapping. Available hashes (first 5):")
                for h in list(self.hash_mapping.keys())[:5]:
                    print(f"  {h}")
                raise ValueError(f"Hash {hash_part} not found in mapping")
                
        except Exception as e:
            print(f"Error processing path '{image_path}': {str(e)}")
            print(f"Path components: {list(Path(image_path).parts)}")
            raise

def load_predictions(predictions_file):
    """Load predictions from JSON file."""
    print(f"Loading predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        data = json.load(f)
        print("\nPredictions data structure:")
        if isinstance(data, list):
            print(f"Data is a list with {len(data)} items")
            if len(data) > 0:
                print("First item structure:")
                print(json.dumps(data[0], indent=2)[:500])
        elif isinstance(data, dict):
            print("Data is a dictionary with keys:")
            print(list(data.keys()))
            for key in data:
                if isinstance(data[key], list) and len(data[key]) > 0:
                    print(f"\nExample of {key}:")
                    print(json.dumps(data[key][0], indent=2)[:500])
        return data

def convert_predictions_to_df(predictions, confidence_threshold=0.5):
    """Convert predictions to DataFrame with panel annotations."""
    records = []
    page_dimensions = []  # List to store page dimensions
    
    # Handle COCO format
    if isinstance(predictions, dict):
        if 'images' in predictions and 'annotations' in predictions:
            print("\nProcessing COCO format predictions...")
            # Create image lookup and store dimensions
            images = {}
            for img in predictions['images']:
                images[img['id']] = img['file_name']
                if 'width' in img and 'height' in img:
                    page_dimensions.append({
                        'width': img['width'],
                        'height': img['height']
                    })
            print(f"Found {len(images)} images")
            
            # Print some example image paths
            print("\nExample image paths:")
            for i, path in enumerate(list(images.values())[:5]):
                print(f"  {i+1}: {path}")
            
            # Group annotations by image_id
            image_annotations = {}
            for ann in predictions['annotations']:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                if ann.get('score', 1.0) >= confidence_threshold:
                    image_annotations[image_id].append(ann)
            
            print(f"\nFound {len(image_annotations)} images with annotations")
            
            # Initialize hash mapper
            hash_mapper = HashMapper()
            
            # Process each image
            success_count = 0
            for image_id, annotations in tqdm(image_annotations.items(), desc="Processing images"):
                try:
                    image_path = images[image_id]
                    subdb, comic_no, page_no = hash_mapper.get_info(image_path)
                    
                    # Convert annotations to boxes
                    boxes = []
                    max_x = 0
                    max_y = 0
                    for ann in annotations:
                        x, y, w, h = ann['bbox']
                        boxes.append({
                            'box': [x, y, x + w, y + h],  # Convert from COCO [x,y,w,h] to [x1,y1,x2,y2]
                            'score': ann.get('score', 1.0),
                            'center_y': y + h/2,
                            'center_x': x + w/2
                        })
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)
                    
                    # If dimensions not in image metadata, estimate from boxes
                    if not page_dimensions:
                        page_dimensions.append({
                            'width': max_x,
                            'height': max_y
                        })
                    
                    # Sort boxes top-to-bottom, left-to-right
                    boxes.sort(key=lambda x: (x['center_y'], x['center_x']))
                    
                    # Create records
                    for panel_no, box_info in enumerate(boxes, 1):
                        x1, y1, x2, y2 = box_info['box']
                        records.append({
                            'subdb': subdb,
                            'comic_no': comic_no,
                            'page_no': float(page_no),
                            'panel_no': panel_no,
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2),
                            'confidence': float(box_info['score'])
                        })
                    success_count += 1
                
                except Exception as e:
                    print(f"Error processing image {image_id} ({images[image_id]}): {e}")
                    continue
            
            print(f"\nSuccessfully processed {success_count} out of {len(image_annotations)} images")
            
            # Calculate and display page resolution statistics
            if page_dimensions:
                widths = [d['width'] for d in page_dimensions]
                heights = [d['height'] for d in page_dimensions]
                
                print("\nPage Resolution Statistics:")
                print(f"Average dimensions: {sum(widths)/len(widths):.1f} x {sum(heights)/len(heights):.1f}")
                print(f"Min dimensions: {min(widths):.1f} x {min(heights):.1f}")
                print(f"Max dimensions: {max(widths):.1f} x {max(heights):.1f}")
                
                # Calculate most common resolutions
                from collections import Counter
                resolutions = Counter([f"{int(w)}x{int(h)}" for w, h in zip(widths, heights)])
                print("\nMost common resolutions:")
                for res, count in resolutions.most_common(5):
                    print(f"  {res}: {count} pages ({count/len(page_dimensions)*100:.1f}%)")
        else:
            print("Unknown dictionary format. Keys found:", list(predictions.keys()))
            return pd.DataFrame()
    else:
        print(f"Unexpected predictions type: {type(predictions)}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    if len(df) > 0:
        print("\nDataFrame summary:")
        print(f"Total panels found: {len(df)}")
        print("\nSample of records:")
        print(df.head())
        print("\nValue ranges:")
        for col in ['x1', 'y1', 'x2', 'y2']:
            print(f"{col}: {df[col].min():.1f} to {df[col].max():.1f}")
        
        # Print some statistics
        print("\nPanels per page statistics:")
        panels_per_page = df.groupby(['subdb', 'comic_no', 'page_no']).size()
        print(f"Mean: {panels_per_page.mean():.1f}")
        print(f"Median: {panels_per_page.median():.1f}")
        print(f"Min: {panels_per_page.min()}")
        print(f"Max: {panels_per_page.max()}")
    else:
        print("No valid panels found!")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create panel annotations CSV from detection results.')
    parser.add_argument('--detector', type=str, default='yolo-mix',
                       choices=['yolo-mix', 'magi', 'grounding-dino', 'faster-rcnn', 'dass-m109'],
                       help='Detector to use for panel annotations')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for panel detections')
    parser.add_argument('--output-dir', type=str, default='benchmarks/cap_val/data',
                       help='Directory to save the panel annotations CSV')
    args = parser.parse_args()
    
    # Setup paths
    predictions_file = Path(f'data/predicts.coco/2000ad/{args.detector}/predictions.json')
    if not predictions_file.exists():
        raise ValueError(f"Predictions file not found: {predictions_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process predictions
    predictions = load_predictions(predictions_file)
    df = convert_predictions_to_df(predictions, args.confidence)
    
    # Save to CSV
    output_file = output_dir / 'compiled_panels_annotations.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved panel annotations to {output_file}")
    print(f"Total panels found: {len(df)}")
    print("\nSample of the data:")
    print(df.head())

if __name__ == '__main__':
    main() 