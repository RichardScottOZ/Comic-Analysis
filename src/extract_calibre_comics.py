#!/usr/bin/env python3
"""
Extract CalibreComics CBZ/CBR files with multiprocessing support.
Extracts comic archives into individual folders with images.
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import zipfile
import rarfile
import shutil
import tempfile
import os
from tqdm import tqdm
import time
from datetime import datetime

# Configure rarfile for RAR extraction
rarfile.UNRAR_TOOL = "unrar"  # You may need to install unrar

def is_comic_archive(file_path):
    """Check if file is a comic archive (CBZ/CBR)."""
    return file_path.suffix.lower() in ['.cbz', '.cbr']

def extract_cbz_file(cbz_path, output_dir):
    """Extract CBZ file (ZIP format) to output directory."""
    try:
        with zipfile.ZipFile(cbz_path, 'r') as zip_ref:
            # Get list of image files
            image_files = [f for f in zip_ref.namelist() 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
            
            if not image_files:
                return False, "No image files found in CBZ"
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract images
            for image_file in image_files:
                zip_ref.extract(image_file, output_dir)
            
            return True, f"Extracted {len(image_files)} images"
            
    except Exception as e:
        return False, f"Error extracting CBZ: {str(e)}"

def extract_cbr_file(cbr_path, output_dir):
    """Extract CBR file (RAR format) to output directory."""
    try:
        with rarfile.RarFile(cbr_path, 'r') as rar_ref:
            # Get list of image files
            image_files = [f for f in rar_ref.namelist() 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
            
            if not image_files:
                return False, "No image files found in CBR"
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract images
            for image_file in image_files:
                rar_ref.extract(image_file, output_dir)
            
            return True, f"Extracted {len(image_files)} images"
            
    except Exception as e:
        return False, f"Error extracting CBR: {str(e)}"

def extract_single_comic(args):
    """Extract a single comic file - designed for multiprocessing."""
    comic_path, output_base_dir, preserve_structure, input_base_dir = args
    
    try:
        # Determine output path
        if preserve_structure:
            # Preserve the full directory structure
            relative_path = comic_path.relative_to(input_base_dir)
            output_dir = output_base_dir / relative_path.parent / comic_path.stem
        else:
            # Flatten structure - just use comic name
            output_dir = output_base_dir / comic_path.stem
        
        # Extract based on file type
        if comic_path.suffix.lower() == '.cbz':
            success, message = extract_cbz_file(comic_path, output_dir)
        elif comic_path.suffix.lower() == '.cbr':
            success, message = extract_cbr_file(comic_path, output_dir)
        else:
            return {
                'status': 'error',
                'file_path': str(comic_path),
                'error': f'Unsupported file type: {comic_path.suffix}'
            }
        
        if success:
            return {
                'status': 'success',
                'file_path': str(comic_path),
                'output_dir': str(output_dir),
                'message': message
            }
        else:
            return {
                'status': 'extraction_error',
                'file_path': str(comic_path),
                'error': message
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'file_path': str(comic_path),
            'error': str(e)
        }

def find_comic_files(input_dir):
    """Find all CBZ/CBR files in the directory structure."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return []
    
    print(f"Scanning directory: {input_dir}")
    comic_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if is_comic_archive(file_path):
                comic_files.append(file_path)
    
    print(f"Found {len(comic_files)} comic archive files")
    return sorted(comic_files)

def main():
    parser = argparse.ArgumentParser(description='Extract CalibreComics CBZ/CBR files with multiprocessing')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing CalibreComics structure')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for extracted comics')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve the original directory structure in output')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip comics that already have extracted folders')
    parser.add_argument('--max-comics', type=int, default=None,
                       help='Maximum number of comics to process (for testing)')
    parser.add_argument('--report-file', type=str, default='extraction_report.json',
                       help='File to save extraction report')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all comic files
    comic_files = find_comic_files(input_dir)
    
    if not comic_files:
        print("No comic archive files found!")
        return
    
    # Limit number of comics if specified
    if args.max_comics:
        comic_files = comic_files[:args.max_comics]
        print(f"Limited to {len(comic_files)} comics for testing")
    
    print(f"\nProcessing {len(comic_files)} comic files...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Preserve structure: {args.preserve_structure}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Prepare arguments for multiprocessing
    process_args = []
    skipped_count = 0
    
    for comic_file in comic_files:
        # Determine output path for checking if exists
        if args.preserve_structure:
            relative_path = comic_file.relative_to(input_dir)
            output_path = output_dir / relative_path.parent / comic_file.stem
        else:
            output_path = output_dir / comic_file.stem
        
        # Skip if output already exists and skip-existing is set
        if args.skip_existing and output_path.exists():
            skipped_count += 1
            continue
        
        process_args.append((
            comic_file,
            output_dir,
            args.preserve_structure,
            input_dir # Pass input_dir to extract_single_comic
        ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing folders")
    
    if not process_args:
        print("No comics to process")
        return
    
    print(f"Processing {len(process_args)} comics")
    
    # Set number of workers
    max_workers = args.max_workers or min(mp.cpu_count(), len(process_args))
    print(f"Using {max_workers} worker processes")
    
    # Process comics in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(extract_single_comic, args): args for args in process_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(process_args), desc="Extracting comics") as pbar:
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                # Update progress bar description with current status
                success_count = sum(1 for r in results if r['status'] == 'success')
                error_count = sum(1 for r in results if r['status'] == 'error')
                extraction_error_count = sum(1 for r in results if r['status'] == 'extraction_error')
                pbar.set_description(f"Success: {success_count}, Errors: {error_count}, Extraction Errors: {extraction_error_count}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    extraction_error_count = sum(1 for r in results if r['status'] == 'extraction_error')
    
    print(f"\n=== Extraction Summary ===")
    print(f"Total comics: {len(process_args)}")
    print(f"Successful: {success_count}")
    print(f"Extraction errors: {extraction_error_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per comic: {total_time/len(process_args):.2f} seconds")
    
    # Create detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'total_comics': len(process_args),
        'successful': success_count,
        'extraction_errors': extraction_error_count,
        'errors': error_count,
        'skipped': skipped_count,
        'preserve_structure': args.preserve_structure,
        'max_workers': max_workers,
        'results': results
    }
    
    # Save report
    import json
    with open(args.report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {args.report_file}")
    
    # Print errors if any
    if error_count > 0:
        print(f"\n=== Errors ===")
        for result in results:
            if result['status'] == 'error':
                print(f"  {result['file_path']}: {result['error']}")
    
    if extraction_error_count > 0:
        print(f"\n=== Extraction Errors ===")
        for result in results:
            if result['status'] == 'extraction_error':
                print(f"  {result['file_path']}: {result['error']}")

if __name__ == "__main__":
    main() 