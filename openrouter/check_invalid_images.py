#!/usr/bin/env python3
"""
Check for invalid image files in extracted comic directories.
Identifies corrupted or non-image files that may need re-extraction.
"""

import argparse
from pathlib import Path
import os
from PIL import Image
import json
from collections import defaultdict
from tqdm import tqdm

def is_valid_image_file(file_path):
    """Check if a file is a valid image by trying to open it with PIL."""
    try:
        with Image.open(file_path) as img:
            # Try to access image data to ensure it's actually valid
            img.verify()
        return True
    except Exception as e:
        return False

def check_image_files(extracted_dir):
    """Scan extracted directory for invalid image files."""
    print(f"Scanning extracted directory: {extracted_dir}")
    
    extracted_path = Path(extracted_dir)
    if not extracted_path.exists():
        print(f"Error: Extracted directory not found: {extracted_dir}")
        return {}
    
    # Track invalid files by their source comic folder
    invalid_files_by_folder = defaultdict(list)
    total_files_checked = 0
    total_invalid_files = 0
    
    # Scan all image files in the extracted directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    
    print("Scanning for image files...")
    
    # First, collect all image files to get total count for progress bar
    image_files = []
    for image_file in extracted_path.rglob("*"):
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            image_files.append(image_file)
    
    print(f"Found {len(image_files)} image files to check")
    
    # Now check each image file with progress bar
    for image_file in tqdm(image_files, desc="Checking image files", unit="files"):
        total_files_checked += 1
        
        if not is_valid_image_file(image_file):
            total_invalid_files += 1
            
            # Determine the source comic folder (the folder containing this image)
            comic_folder = image_file.parent.name
            
            invalid_files_by_folder[comic_folder].append({
                'file_path': str(image_file),
                'file_name': image_file.name,
                'file_size': image_file.stat().st_size,
                'relative_path': str(image_file.relative_to(extracted_path))
            })
            
            if total_invalid_files <= 10:  # Show first 10 invalid files
                print(f"  Invalid: {image_file.relative_to(extracted_path)}")
    
    print(f"\nScan completed:")
    print(f"  Total image files checked: {total_files_checked}")
    print(f"  Invalid image files found: {total_invalid_files}")
    print(f"  Comic folders with invalid files: {len(invalid_files_by_folder)}")
    
    return invalid_files_by_folder

def generate_report(invalid_files_by_folder, output_file=None):
    """Generate a detailed report of invalid image files."""
    print("\n" + "="*80)
    print("INVALID IMAGE FILES REPORT")
    print("="*80)
    
    if not invalid_files_by_folder:
        print("\n✅ No invalid image files found!")
        return
    
    print(f"\nFound invalid image files in {len(invalid_files_by_folder)} comic folders:")
    
    # Sort folders by number of invalid files (most problematic first)
    sorted_folders = sorted(invalid_files_by_folder.items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    for folder_name, invalid_files in sorted_folders:
        print(f"\n📁 {folder_name} ({len(invalid_files)} invalid files):")
        
        # Sort files by size to identify potential issues
        sorted_files = sorted(invalid_files, key=lambda x: x['file_size'])
        
        for file_info in sorted_files:
            size_mb = file_info['file_size'] / (1024 * 1024)
            print(f"  - {file_info['file_name']} ({size_mb:.2f} MB)")
            print(f"    Path: {file_info['relative_path']}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY:")
    print(f"  Total invalid files: {sum(len(files) for files in invalid_files_by_folder.values())}")
    print(f"  Folders with issues: {len(invalid_files_by_folder)}")
    
    # Identify folders with many invalid files (likely extraction problems)
    problematic_folders = [(name, files) for name, files in invalid_files_by_folder.items() 
                          if len(files) > 0]  # Changed from > 5 to > 0
    
    if problematic_folders:
        print(f"\n⚠️  PROBLEMATIC FOLDERS (any invalid files):")
        for folder_name, files in problematic_folders:
            print(f"  - {folder_name}: {len(files)} invalid files")
    
    # Save detailed report to JSON if requested
    if output_file:
        report_data = {
            'summary': {
                'total_invalid_files': sum(len(files) for files in invalid_files_by_folder.values()),
                'folders_with_issues': len(invalid_files_by_folder),
                'problematic_folders': len(problematic_folders)
            },
            'invalid_files_by_folder': dict(invalid_files_by_folder)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Check for invalid image files in extracted comic directories')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Extracted directory to scan for invalid images')
    parser.add_argument('--output-report', type=str, default=None,
                       help='Save detailed report to JSON file (optional)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all invalid files (not just first 10)')
    
    args = parser.parse_args()
    
    # Check for invalid image files
    invalid_files_by_folder = check_image_files(args.extracted_dir)
    
    # Generate report
    generate_report(invalid_files_by_folder, args.output_report)
    
    # Show all files if requested
    if args.show_all and invalid_files_by_folder:
        print(f"\n📋 ALL INVALID FILES:")
        for folder_name, invalid_files in invalid_files_by_folder.items():
            print(f"\n{folder_name}:")
            for file_info in invalid_files:
                print(f"  {file_info['relative_path']}")
    
    # Provide recommendations
    if invalid_files_by_folder:
        print(f"\n🔧 RECOMMENDATIONS:")
        print(f"  1. Check if invalid files are 0-byte or corrupted")
        print(f"  2. Re-extract problematic comic folders")
        print(f"  3. Verify source CBZ/CBR files are not corrupted")
        print(f"  4. Use the JSON report to identify patterns in invalid files")

if __name__ == "__main__":
    main() 