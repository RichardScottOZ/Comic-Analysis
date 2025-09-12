"""
Check what pages the CLOSURE-Lite dataset will load
"""

import os
import json
from pathlib import Path
import random

def check_dataset(json_dir, image_root, num_samples=10):
    """Check what pages are in the dataset"""
    
    # Convert Windows paths to WSL paths if needed
    if json_dir.startswith('E:/') or json_dir.startswith('E:\\'):
        json_dir = "/mnt/e" + json_dir[2:].replace('\\', '/')
    if image_root.startswith('E:/') or image_root.startswith('E:\\'):
        image_root = "/mnt/e" + image_root[2:].replace('\\', '/')
    
    print(f"🔍 Checking dataset in: {json_dir}")
    print(f"🖼️  Image root: {image_root}")
    
    # Get all JSON files
    json_paths = list(Path(json_dir).glob("*.json"))
    print(f"📄 Found {len(json_paths)} JSON files")
    
    if len(json_paths) == 0:
        print("❌ No JSON files found!")
        return
    
    # Sample some files to check
    sample_files = random.sample(json_paths, min(num_samples, len(json_paths)))
    
    print(f"\n📋 Sample of {len(sample_files)} files:")
    for i, json_path in enumerate(sample_files):
        print(f"  {i+1}. {json_path.name}")
    
    # Check a few JSON files in detail
    print(f"\n🔍 Detailed analysis of first 3 files:")
    
    total_pages = 0
    total_panels = 0
    
    for i, json_path in enumerate(sample_files[:3]):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                
                print(f"\n📖 File {i+1}: {json_path.name}")
                print(f"   Pages in file: {len(data)}")
                
                for j, page in enumerate(data[:2]):  # Check first 2 pages
                    print(f"   Page {j+1}:")
                    print(f"     - Image: {page.get('page_image_path', 'N/A')}")
                    print(f"     - Panels: {len(page.get('panels', []))}")
                    
                    # Check if image exists
                    img_path = page.get('page_image_path', '')
                    if img_path.startswith('E:/') or img_path.startswith('E:\\'):
                        img_path = "/mnt/e" + img_path[2:].replace('\\', '/')
                    
                    if os.path.exists(img_path):
                        print(f"     - ✅ Image exists")
                    else:
                        print(f"     - ❌ Image missing: {img_path}")
                    
                    # Show panel info
                    panels = page.get('panels', [])
                    for k, panel in enumerate(panels[:3]):  # First 3 panels
                        coords = panel.get('panel_coords', [])
                        text = panel.get('text', [])
                        print(f"       Panel {k+1}: coords={coords}, text_len={len(text)}")
                
                total_pages += len(data)
                total_panels += sum(len(page.get('panels', [])) for page in data)
                
        except Exception as e:
            print(f"❌ Error reading {json_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Total JSON files: {len(json_paths):,}")
    print(f"   Sample pages checked: {total_pages}")
    print(f"   Sample panels checked: {total_panels}")
    print(f"   Avg panels per page: {total_panels/total_pages:.1f}" if total_pages > 0 else "   No pages found")
    
    # Check image availability
    print(f"\n🖼️  Image availability check:")
    missing_images = 0
    checked_images = 0
    
    for json_path in sample_files[:10]:  # Check 10 files
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                
                for page in data[:1]:  # Check first page of each file
                    img_path = page.get('page_image_path', '')
                    if img_path:
                        if img_path.startswith('E:/') or img_path.startswith('E:\\'):
                            img_path = "/mnt/e" + img_path[2:].replace('\\', '/')
                        
                        checked_images += 1
                        if not os.path.exists(img_path):
                            missing_images += 1
                            
        except Exception as e:
            print(f"Error checking {json_path}: {e}")
    
    if checked_images > 0:
        print(f"   Images checked: {checked_images}")
        print(f"   Missing images: {missing_images}")
        print(f"   Availability: {(checked_images-missing_images)/checked_images*100:.1f}%")
    else:
        print("   No images checked")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check CLOSURE-Lite dataset')
    parser.add_argument('--json_dir', type=str, required=True,
                       help='Directory containing DataSpec JSON files')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Root directory for comic images')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample files to check')
    
    args = parser.parse_args()
    
    check_dataset(args.json_dir, args.image_root, args.num_samples)

if __name__ == "__main__":
    main()
