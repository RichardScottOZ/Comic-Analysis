#!/usr/bin/env python3
"""
Batch comic analysis script that walks through a directory structure
and processes each image with the comic analysis model.
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import time

def find_image_files(root_dir):
    """Find all image files in the directory structure."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory not found: {root_dir}")
        return []
    
    print(f"Scanning directory: {root_dir}")
    
    # Walk through all subdirectories
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            print(f"Processing subdirectory: {subdir.name}")
            for image_file in subdir.glob("*.jpg"):  # Focus on jpg files based on the structure
                if image_file.is_file():
                    image_files.append(image_file)
    
    print(f"Found {len(image_files)} image files")
    return image_files

def process_image(image_path, output_dir, model="qwen/qwen2.5-vl-32b-instruct:free", 
                 temperature=0.1, top_p=1.0, skip_existing=True, input_dir=None):
    """Process a single image with the comic analysis script."""
    
    # Create output filename based on the image path
    if input_dir:
        # Use the provided input directory for relative path calculation
        input_path = Path(input_dir)
        try:
            relative_path = image_path.relative_to(input_path)
        except ValueError:
            # Fallback: use just the filename if relative path fails
            relative_path = Path(image_path.name)
    else:
        # Fallback: use just the filename
        relative_path = Path(image_path.name)
    
    output_filename = f"{relative_path.parent}_{relative_path.stem}.json"
    output_path = Path(output_dir) / output_filename
    
    # Skip if output already exists and skip_existing is True
    if skip_existing and output_path.exists():
        return "skipped"
    
    # Create the command
    cmd = [
        "python", "benchmarks/detections/openrouter/comictest3.py",
        "--image-path", str(image_path),
        "--output-path", str(output_path),
        "--model", model,
        "--temperature", str(temperature),
        "--top-p", str(top_p)
    ]
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            return "success"
        else:
            print(f"Error processing {image_path}: {result.stderr}")
            return "error"
            
    except subprocess.TimeoutExpired:
        print(f"Timeout processing {image_path}")
        return "timeout"
    except Exception as e:
        print(f"Exception processing {image_path}: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description='Batch process comic images with analysis')
    parser.add_argument('--input-dir', type=str, 
                       default=r'C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\2000ad\images',
                       help='Root directory containing comic images')
    parser.add_argument('--output-dir', type=str, 
                       default='benchmarks/detections/openrouter/analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-vl-32b-instruct:free',
                       help='Model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for generation (0.0-1.0)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip images that already have analysis results')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--start-from', type=str, default=None,
                       help='Start processing from a specific image path')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = find_image_files(args.input_dir)
    
    if not image_files:
        
        print("No image files found!")
        return
    
    # Filter by start-from if specified
    if args.start_from:
        start_path = Path(args.start_from)
        try:
            start_index = next(i for i, img in enumerate(image_files) if img == start_path)
            image_files = image_files[start_index:]
            print(f"Starting from image: {start_path}")
        except StopIteration:
            print(f"Start image not found: {args.start_from}")
            return
    
    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Limited to {args.max_images} images for testing")
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Process images with progress bar
    results = {"success": 0, "error": 0, "timeout": 0, "skipped": 0}
    
    for image_file in tqdm(image_files, desc="Processing images"):
        result = process_image(
            image_file, 
            output_dir, 
            args.model, 
            args.temperature, 
            args.top_p, 
            args.skip_existing,
            args.input_dir # Pass input_dir to process_image
        )
        results[result] += 1
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(1)
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Success: {results['success']}")
    print(f"Errors: {results['error']}")
    print(f"Timeouts: {results['timeout']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total: {sum(results.values())}")

if __name__ == "__main__":
    main() 