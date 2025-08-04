#!/usr/bin/env python3
"""
Convert PDF/EPUB files to CBZ format with multiprocessing support.
Checks for existing CBZ files and only converts when needed.
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import zipfile
import tempfile
import shutil
import os
from tqdm import tqdm
import time
from datetime import datetime
import fitz  # PyMuPDF for PDF processing
import ebooklib
from ebooklib import epub
from PIL import Image
import io

def is_comic_archive(file_path):
    """Check if file is a comic archive (CBZ/CBR)."""
    return file_path.suffix.lower() in ['.cbz', '.cbr']

def is_convertible_file(file_path):
    """Check if file can be converted to CBZ (PDF/EPUB)."""
    return file_path.suffix.lower() in ['.pdf', '.epub']

def has_existing_cbz(folder_path):
    """Check if folder already has a CBZ file."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return False
    
    # Look for CBZ files in the folder
    cbz_files = list(folder.glob("*.cbz"))
    return len(cbz_files) > 0

def convert_pdf_to_cbz(pdf_path, output_dir):
    """Convert PDF file to CBZ format."""
    try:
        # Open PDF
        pdf_doc = fitz.open(pdf_path)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract pages as images
            image_files = []
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                
                # Render page as image (higher DPI for better quality)
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image for processing
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Save image
                image_filename = f"page_{page_num+1:03d}.png"
                image_path = temp_path / image_filename
                img.save(image_path, "PNG")
                image_files.append(image_filename)
            
            # Create CBZ file
            cbz_path = output_dir / f"{pdf_path.stem}.cbz"
            with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                for image_file in image_files:
                    cbz.write(temp_path / image_file, image_file)
            
            return True, f"Converted {len(image_files)} pages to CBZ"
            
    except Exception as e:
        return False, f"Error converting PDF: {str(e)}"

def convert_epub_to_cbz(epub_path, output_dir):
    """Convert EPUB file to CBZ format."""
    try:
        # Open EPUB
        book = epub.read_epub(epub_path)
        
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract images from EPUB
            image_files = []
            image_counter = 1
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    # Get image data
                    img_data = item.get_content()
                    
                    # Determine image format
                    if img_data.startswith(b'\xff\xd8'):  # JPEG
                        ext = '.jpg'
                    elif img_data.startswith(b'\x89PNG'):  # PNG
                        ext = '.png'
                    else:
                        ext = '.jpg'  # Default to JPEG
                    
                    # Save image
                    image_filename = f"page_{image_counter:03d}{ext}"
                    image_path = temp_path / image_filename
                    
                    with open(image_path, 'wb') as f:
                        f.write(img_data)
                    
                    image_files.append(image_filename)
                    image_counter += 1
            
            if not image_files:
                return False, "No images found in EPUB"
            
            # Create CBZ file
            cbz_path = output_dir / f"{epub_path.stem}.cbz"
            with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                for image_file in image_files:
                    cbz.write(temp_path / image_file, image_file)
            
            return True, f"Converted {len(image_files)} images to CBZ"
            
    except Exception as e:
        return False, f"Error converting EPUB: {str(e)}"

def convert_single_file(args):
    """Convert a single file to CBZ - designed for multiprocessing."""
    file_path, output_dir, preserve_structure, input_base_dir = args
    
    try:
        # Determine output path
        if preserve_structure:
            # Preserve the full directory structure
            relative_path = file_path.relative_to(input_base_dir)
            output_path = output_dir / relative_path.parent
        else:
            # Flatten structure - just use file name
            output_path = output_dir
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if CBZ already exists
        cbz_path = output_path / f"{file_path.stem}.cbz"
        if cbz_path.exists():
            return {
                'status': 'skipped',
                'file_path': str(file_path),
                'message': 'CBZ file already exists'
            }
        
        # Convert based on file type
        if file_path.suffix.lower() == '.pdf':
            success, message = convert_pdf_to_cbz(file_path, output_path)
        elif file_path.suffix.lower() == '.epub':
            success, message = convert_epub_to_cbz(file_path, output_path)
        else:
            return {
                'status': 'error',
                'file_path': str(file_path),
                'error': f'Unsupported file type: {file_path.suffix}'
            }
        
        if success:
            return {
                'status': 'success',
                'file_path': str(file_path),
                'output_path': str(cbz_path),
                'message': message
            }
        else:
            return {
                'status': 'conversion_error',
                'file_path': str(file_path),
                'error': message
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'file_path': str(file_path),
            'error': str(e)
        }

def find_convertible_files(input_dir):
    """Find all PDF/EPUB files that need conversion."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return []
    
    print(f"Scanning directory: {input_dir}")
    convertible_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_path):
        root_path = Path(root)
        
        # Check if this folder already has a CBZ file
        if has_existing_cbz(root_path):
            continue  # Skip folders that already have CBZ files
        
        # Look for PDF/EPUB files in this folder
        for file in files:
            file_path = root_path / file
            if is_convertible_file(file_path):
                convertible_files.append(file_path)
    
    print(f"Found {len(convertible_files)} files to convert")
    return sorted(convertible_files)

def main():
    parser = argparse.ArgumentParser(description='Convert PDF/EPUB files to CBZ format with multiprocessing')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing comic folders')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve the original directory structure in output')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that already have CBZ versions')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--report-file', type=str, default='conversion_report.json',
                       help='File to save conversion report')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir  # Use same directory
    
    # Find all convertible files
    convertible_files = find_convertible_files(input_dir)
    
    if not convertible_files:
        print("No files to convert!")
        return
    
    # Limit number of files if specified
    if args.max_files:
        convertible_files = convertible_files[:args.max_files]
        print(f"Limited to {len(convertible_files)} files for testing")
    
    print(f"\nProcessing {len(convertible_files)} files...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Preserve structure: {args.preserve_structure}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Prepare arguments for multiprocessing
    process_args = []
    skipped_count = 0
    
    for file_path in convertible_files:
        # Determine output path for checking if exists
        if args.preserve_structure:
            relative_path = file_path.relative_to(input_dir)
            output_path = output_dir / relative_path.parent
        else:
            output_path = output_dir
        
        cbz_path = output_path / f"{file_path.stem}.cbz"
        
        # Skip if CBZ already exists and skip-existing is set
        if args.skip_existing and cbz_path.exists():
            skipped_count += 1
            continue
        
        process_args.append((
            file_path,
            output_dir,
            args.preserve_structure,
            input_dir
        ))
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing CBZ files")
    
    if not process_args:
        print("No files to process")
        return
    
    print(f"Processing {len(process_args)} files")
    
    # Set number of workers
    max_workers = args.max_workers or min(mp.cpu_count(), len(process_args))
    print(f"Using {max_workers} worker processes")
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(convert_single_file, args): args for args in process_args}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(process_args), desc="Converting files") as pbar:
            for future in as_completed(future_to_args):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                # Update progress bar description with current status
                success_count = sum(1 for r in results if r['status'] == 'success')
                error_count = sum(1 for r in results if r['status'] == 'error')
                conversion_error_count = sum(1 for r in results if r['status'] == 'conversion_error')
                skipped_count = sum(1 for r in results if r['status'] == 'skipped')
                pbar.set_description(f"Success: {success_count}, Errors: {error_count}, Conversion Errors: {conversion_error_count}, Skipped: {skipped_count}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    conversion_error_count = sum(1 for r in results if r['status'] == 'conversion_error')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"\n=== Conversion Summary ===")
    print(f"Total files: {len(process_args)}")
    print(f"Successful: {success_count}")
    print(f"Conversion errors: {conversion_error_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per file: {total_time/len(process_args):.2f} seconds")
    
    # Create detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'total_files': len(process_args),
        'successful': success_count,
        'conversion_errors': conversion_error_count,
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
    
    if conversion_error_count > 0:
        print(f"\n=== Conversion Errors ===")
        for result in results:
            if result['status'] == 'conversion_error':
                print(f"  {result['file_path']}: {result['error']}")

if __name__ == "__main__":
    main() 