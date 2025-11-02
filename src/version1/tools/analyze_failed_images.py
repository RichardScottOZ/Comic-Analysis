#!/usr/bin/env python3
"""
Analyze failed images from batch comic analysis to identify problematic files.
Checks file sizes, resolutions, and other properties to help diagnose issues.
"""

import argparse
from pathlib import Path
import json
from PIL import Image
import os
from collections import defaultdict
import re
from tqdm import tqdm

def parse_error_log(error_text):
    """Parse the error output from batch_comic_analysis_multi to extract failed file paths."""
    failed_files = []
    
    # Extract file paths from error messages
    # Pattern: E:\path\to\file.jpg: error message
    pattern = r'(E:\\[^:]+\.(?:jpg|jpeg|png|bmp|tiff)):'
    
    matches = re.findall(pattern, error_text)
    for match in matches:
        failed_files.append(match)
    
    return failed_files

def get_image_properties(image_path):
    """Get detailed properties of an image file."""
    try:
        file_path = Path(image_path)
        
        if not file_path.exists():
            return {
                'exists': False,
                'error': 'File does not exist'
            }
        
        # Get file size
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Try to open with PIL
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_type = img.format
                mode = img.mode
                
                # Check if image is corrupted
                try:
                    img.verify()
                    is_corrupted = False
                except Exception:
                    is_corrupted = True
                
                # Get additional properties
                properties = {
                    'exists': True,
                    'file_size_bytes': file_size,
                    'file_size_mb': round(file_size_mb, 2),
                    'width': width,
                    'height': height,
                    'resolution': f"{width}x{height}",
                    'total_pixels': width * height,
                    'megapixels': round((width * height) / 1000000, 2),
                    'format': format_type,
                    'mode': mode,
                    'is_corrupted': is_corrupted,
                    'aspect_ratio': round(width / height, 2) if height > 0 else 0
                }
                
                return properties
                
        except Exception as e:
            return {
                'exists': True,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size_mb, 2),
                'error': f'PIL error: {str(e)}',
                'is_corrupted': True
            }
    
    except Exception as e:
        return {
            'exists': False,
            'error': f'General error: {str(e)}'
        }

def analyze_failed_files(failed_files, output_file=None):
    """Analyze a list of failed files and generate a report."""
    print(f"Analyzing {len(failed_files)} failed files...")
    
    results = []
    size_distribution = defaultdict(int)
    resolution_distribution = defaultdict(int)
    format_distribution = defaultdict(int)
    corrupted_files = []
    very_large_files = []
    very_high_res_files = []
    
    for file_path in tqdm(failed_files, desc="Analyzing files"):
        properties = get_image_properties(file_path)
        properties['file_path'] = file_path
        
        results.append(properties)
        
        if properties.get('exists', False):
            # Size distribution
            size_mb = properties.get('file_size_mb', 0)
            if size_mb > 50:
                very_large_files.append(file_path)
            
            # Resolution distribution
            megapixels = properties.get('megapixels', 0)
            if megapixels > 50:  # Very high resolution
                very_high_res_files.append(file_path)
            
            # Format distribution
            format_type = properties.get('format', 'unknown')
            format_distribution[format_type] += 1
            
            # Corrupted files
            if properties.get('is_corrupted', False):
                corrupted_files.append(file_path)
    
    # Generate report
    print(f"\n=== FAILED IMAGES ANALYSIS REPORT ===")
    print(f"Total failed files: {len(failed_files)}")
    print(f"Files that exist: {sum(1 for r in results if r.get('exists', False))}")
    print(f"Files that don't exist: {sum(1 for r in results if not r.get('exists', False))}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Very large files (>50MB): {len(very_large_files)}")
    print(f"Very high resolution files (>50MP): {len(very_high_res_files)}")
    
    # Size analysis
    if results:
        sizes = [r.get('file_size_mb', 0) for r in results if r.get('exists', False)]
        if sizes:
            print(f"\n=== FILE SIZE ANALYSIS ===")
            print(f"Average size: {sum(sizes)/len(sizes):.2f} MB")
            print(f"Largest file: {max(sizes):.2f} MB")
            print(f"Smallest file: {min(sizes):.2f} MB")
            
            # Size distribution
            size_ranges = {
                '0-1MB': 0, '1-5MB': 0, '5-10MB': 0, 
                '10-20MB': 0, '20-50MB': 0, '50MB+': 0
            }
            
            for size in sizes:
                if size < 1:
                    size_ranges['0-1MB'] += 1
                elif size < 5:
                    size_ranges['1-5MB'] += 1
                elif size < 10:
                    size_ranges['5-10MB'] += 1
                elif size < 20:
                    size_ranges['10-20MB'] += 1
                elif size < 50:
                    size_ranges['20-50MB'] += 1
                else:
                    size_ranges['50MB+'] += 1
            
            print(f"\nSize distribution:")
            for range_name, count in size_ranges.items():
                print(f"  {range_name}: {count} files")
    
    # Resolution analysis
    if results:
        resolutions = [r.get('megapixels', 0) for r in results if r.get('exists', False)]
        if resolutions:
            print(f"\n=== RESOLUTION ANALYSIS ===")
            print(f"Average resolution: {sum(resolutions)/len(resolutions):.2f} MP")
            print(f"Highest resolution: {max(resolutions):.2f} MP")
            print(f"Lowest resolution: {min(resolutions):.2f} MP")
            
            # Resolution distribution
            res_ranges = {
                '0-1MP': 0, '1-5MP': 0, '5-10MP': 0, 
                '10-20MP': 0, '20-50MP': 0, '50MP+': 0
            }
            
            for res in resolutions:
                if res < 1:
                    res_ranges['0-1MP'] += 1
                elif res < 5:
                    res_ranges['1-5MP'] += 1
                elif res < 10:
                    res_ranges['5-10MP'] += 1
                elif res < 20:
                    res_ranges['10-20MP'] += 1
                elif res < 50:
                    res_ranges['20-50MP'] += 1
                else:
                    res_ranges['50MP+'] += 1
            
            print(f"\nResolution distribution:")
            for range_name, count in res_ranges.items():
                print(f"  {range_name}: {count} files")
    
    # Format analysis
    if format_distribution:
        print(f"\n=== FORMAT ANALYSIS ===")
        for format_type, count in format_distribution.items():
            print(f"  {format_type}: {count} files")
    
    # Show problematic files
    if very_large_files:
        print(f"\n=== VERY LARGE FILES (>50MB) ===")
        for file_path in very_large_files[:10]:  # Show first 10
            props = next(r for r in results if r['file_path'] == file_path)
            print(f"  {file_path}: {props.get('file_size_mb', 0):.2f} MB, {props.get('resolution', 'unknown')}")
        if len(very_large_files) > 10:
            print(f"  ... and {len(very_large_files) - 10} more")
    
    if very_high_res_files:
        print(f"\n=== VERY HIGH RESOLUTION FILES (>50MP) ===")
        for file_path in very_high_res_files[:10]:  # Show first 10
            props = next(r for r in results if r['file_path'] == file_path)
            print(f"  {file_path}: {props.get('megapixels', 0):.2f} MP, {props.get('resolution', 'unknown')}")
        if len(very_high_res_files) > 10:
            print(f"  ... and {len(very_high_res_files) - 10} more")
    
    if corrupted_files:
        print(f"\n=== CORRUPTED FILES ===")
        for file_path in corrupted_files[:10]:  # Show first 10
            props = next(r for r in results if r['file_path'] == file_path)
            print(f"  {file_path}: {props.get('error', 'Unknown error')}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more")
    
    # Save detailed results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_failed': len(failed_files),
                    'corrupted_files': len(corrupted_files),
                    'very_large_files': len(very_large_files),
                    'very_high_res_files': len(very_high_res_files)
                },
                'problematic_files': {
                    'corrupted': corrupted_files,
                    'very_large': very_large_files,
                    'very_high_resolution': very_high_res_files
                },
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze failed images from batch comic analysis')
    parser.add_argument('--error-log', type=str, help='File containing error output from batch analysis')
    parser.add_argument('--failed-files', type=str, nargs='+', help='List of failed file paths')
    parser.add_argument('--output-file', type=str, help='Output JSON file for detailed results')
    parser.add_argument('--extract-from-text', type=str, help='Extract file paths from error text')
    
    args = parser.parse_args()
    
    failed_files = []
    
    if args.error_log:
        with open(args.error_log, 'r', encoding='utf-8') as f:
            error_text = f.read()
        failed_files = parse_error_log(error_text)
    
    elif args.failed_files:
        failed_files = args.failed_files
    
    elif args.extract_from_text:
        failed_files = parse_error_log(args.extract_from_text)
    
    else:
        print("Please provide either --error-log, --failed-files, or --extract-from-text")
        return
    
    if not failed_files:
        print("No failed files found!")
        return
    
    analyze_failed_files(failed_files, args.output_file)

if __name__ == "__main__":
    main() 