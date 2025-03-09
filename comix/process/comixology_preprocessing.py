import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def process_image(input_path: Path, output_path: Path, target_size=(1024, 1536)):
    """Process a single image with high-quality downsampling.
    
    Args:
        input_path (Path): Path to input image
        output_path (Path): Path to save processed image
        target_size (tuple): Target size for downsampling (width, height)
    """
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scaling factor to maintain aspect ratio
        width, height = img.size
        scale = min(target_size[0] / width, target_size[1] / height)
        new_size = (int(width * scale), int(height * scale))
        
        # Use LANCZOS resampling for high-quality downsampling
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with high quality
        img.save(output_path, 'JPEG', quality=95)

def main():
    parser = argparse.ArgumentParser(description='Preprocess Comixology images for the CoMix pipeline.')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing Comixology images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save processed images')
    parser.add_argument('--target-width', type=int, default=1024,
                       help='Target width for downsampling')
    parser.add_argument('--target-height', type=int, default=1536,
                       help='Target height for downsampling')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = (args.target_width, args.target_height)
    
    # Get all image files recursively
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(input_dir.rglob(ext))
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Target size: {target_size}")
    print(f"Output directory: {output_dir}\n")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Maintain same directory structure in output
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix('.jpg')
        
        try:
            process_image(img_path, out_path, target_size)
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

if __name__ == '__main__':
    main() 