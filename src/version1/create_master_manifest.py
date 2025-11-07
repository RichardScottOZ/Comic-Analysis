# tools/create_master_manifest.py

import argparse
import csv
import os
import zipfile
import rarfile
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import io

# --- Configuration & Setup ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# --- Extraction Functions (Unchanged) ---
def extract_cbz(container_path: Path, output_dir: Path):
    """Extracts images from a .cbz (zip) file. Returns count of expected images."""
    with zipfile.ZipFile(container_path, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if not f.endswith('/') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
        if not image_files:
            raise RuntimeError("No image files found in CBZ.")
        zip_ref.extractall(output_dir, members=image_files)
        return len(image_files)

def extract_cbr(container_path: Path, output_dir: Path):
    """Extracts images from a .cbr (rar) file. Returns count of expected images."""
    try:
        with rarfile.RarFile(container_path, 'r') as rar_ref:
            image_files = [f for f in rar_ref.namelist() if not f.endswith('/') and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
            if not image_files:
                raise RuntimeError("No image files found in CBR.")
            rar_ref.extractall(output_dir)
            return len(image_files)
    except rarfile.UNRARToolError:
        print("\nERROR: 'unrar' command not found. Please install the 'unrar' utility on your system to extract .cbr files.")
        raise

def extract_pdf(container_path: Path, output_dir: Path):
    """Extracts images from a .pdf file using high-quality settings. Returns count of pages."""
    doc = fitz.open(container_path)
    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        raise RuntimeError("PDF contains no pages.")
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        output_image_path = output_dir / f"page_{i:04d}.png"
        pix.save(output_image_path)
    doc.close()
    return page_count

def extract_epub(container_path: Path, output_dir: Path):
    """Extracts images from an .epub file. Returns count of images."""
    book = epub.read_epub(container_path)
    image_counter = 0
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            img_data = item.get_content()
            ext = Path(item.get_name()).suffix or '.jpg'
            image_filename = f"page_{image_counter:04d}{ext}"
            image_path = output_dir / image_filename
            with open(image_path, 'wb') as f:
                f.write(img_data)
            image_counter += 1
    if image_counter == 0:
        raise RuntimeError("No images found in EPUB.")
    return image_counter

# --- Main Manifest Logic ---

def create_master_manifest(input_dirs: list[str], extraction_dir: str, output_csv: str):
    print("--- Starting Master Manifest Creation ---")
    
    # --- Phase 1: Process Container Files ---
    print("\n--- Phase 1: Processing Container Files ---")
    all_container_files = []
    container_extensions = [".cbz", ".cbr", ".pdf", ".epub"]
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.is_dir():
            print(f"Warning: Input directory not found at {input_path}. Skipping.")
            continue
        print(f"Scanning for containers in {input_path}...")
        found_files = [p for p in input_path.rglob("*") if p.suffix.lower() in container_extensions]
        all_container_files.extend(found_files)
    
    if not all_container_files:
        print("No container files found.")
    else:
        print(f"Found {len(all_container_files)} container files to process.")

    extraction_path = Path(extraction_dir)
    extraction_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(output_csv).parent / "manifest_creation.log"
    
    processed_image_paths = set() # Keep track of images from containers
    
    # Track stats
    total_expected_pages = 0
    total_extracted_pages = 0
    total_valid_pages = 0
    containers_with_mismatches = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file, \
         open(log_path, 'w', encoding='utf-8') as log_file:
        
        writer = csv.writer(csv_file)
        writer.writerow(["canonical_id", "absolute_image_path"]) # Header

        if all_container_files:
            for container_path in tqdm(all_container_files, desc="Processing containers"):
                try:
                    rel_container_path = container_path.relative_to(next(p for p in [Path(d) for d in input_dirs] if container_path.is_relative_to(p)))
                    output_dir_for_container = extraction_path / rel_container_path.with_suffix('')

                    expected_count = 0
                    if not (output_dir_for_container.exists() and any(output_dir_for_container.iterdir())):
                        output_dir_for_container.mkdir(parents=True, exist_ok=True)
                        ext = container_path.suffix.lower()
                        if ext == '.cbz': expected_count = extract_cbz(container_path, output_dir_for_container)
                        elif ext == '.cbr': expected_count = extract_cbr(container_path, output_dir_for_container)
                        elif ext == '.pdf': expected_count = extract_pdf(container_path, output_dir_for_container)
                        elif ext == '.epub': expected_count = extract_epub(container_path, output_dir_for_container)
                        total_expected_pages += expected_count

                    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
                    extracted_images = sorted([p for p in output_dir_for_container.rglob("*") if p.suffix.lower() in image_extensions])
                    extracted_count = len(extracted_images)
                    total_extracted_pages += extracted_count

                    valid_count = 0
                    for image_file in extracted_images:
                        try:
                            with Image.open(image_file) as img:
                                img.verify()
                            
                            canonical_id = str(image_file.relative_to(extraction_path).with_suffix('').as_posix())
                            absolute_path = str(image_file.resolve())
                            writer.writerow([canonical_id, absolute_path])
                            processed_image_paths.add(absolute_path)
                            valid_count += 1
                        except Exception as img_e:
                            log_file.write(f"CORRUPT_IMAGE\t{str(image_file.resolve())}\t{img_e}\n")
                    
                    total_valid_pages += valid_count
                    
                    # Check for mismatches (only if we actually extracted in this run)
                    if expected_count > 0 and extracted_count != expected_count:
                        mismatch_msg = f"{container_path.name}: expected {expected_count}, extracted {extracted_count}, valid {valid_count}"
                        containers_with_mismatches.append(mismatch_msg)
                        log_file.write(f"PAGE_COUNT_MISMATCH\t{str(container_path.resolve())}\texpected={expected_count}\textracted={extracted_count}\tvalid={valid_count}\n")

                except Exception as cont_e:
                    log_file.write(f"CONTAINER_ERROR\t{str(container_path.resolve())}\t{cont_e}\n")

        # --- Phase 2: Process Loose Image Files ---
        print("\n--- Phase 2: Processing Loose Image Files ---")
        all_loose_images = []
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
        for input_dir in input_dirs:
            input_path = Path(input_dir)
            if not input_path.is_dir(): continue
            print(f"Scanning for loose images in {input_path}...")
            found_files = [p for p in input_path.rglob("*") if p.suffix.lower() in image_extensions]
            all_loose_images.extend(found_files)

        # Filter out images that were already processed from a container extraction
        # This handles the case where an input_dir is the same as the extraction_dir
        new_loose_images = [img for img in all_loose_images if str(img.resolve()) not in processed_image_paths]

        if not new_loose_images:
            print("No new loose image files found to add.")
        else:
            print(f"Found {len(new_loose_images)} new loose image files to validate and add.")
            for image_file in tqdm(new_loose_images, desc="Processing loose images"):
                try:
                    with Image.open(image_file) as img:
                        img.verify()

                    # Find which input_dir this file belongs to to create a relative path
                    input_base_dir = next(p for p in [Path(d) for d in input_dirs] if image_file.is_relative_to(p))
                    canonical_id = str(image_file.relative_to(input_base_dir).with_suffix('').as_posix())
                    absolute_path = str(image_file.resolve())
                    writer.writerow([canonical_id, absolute_path])
                except Exception as img_e:
                    log_file.write(f"CORRUPT_IMAGE\t{str(image_file.resolve())}\t{img_e}\n")

    print(f"\n--- Manifest Creation Complete ---")
    print(f"✅ Manifest saved to: {output_csv}")
    print(f"ℹ️ All extracted images are located in: {extraction_dir}")
    print(f"\n📊 Extraction Statistics:")
    print(f"   Total containers processed: {len(all_container_files)}")
    
    # Always show extracted and valid counts (even if we didn't extract anything new)
    if total_extracted_pages > 0 or total_valid_pages > 0:
        if total_expected_pages > 0:
            print(f"   Expected pages (from new extractions): {total_expected_pages:,}")
            print(f"   Extracted pages (all containers): {total_extracted_pages:,}")
            print(f"   Valid pages written to manifest: {total_valid_pages:,}")
            if total_extracted_pages != total_expected_pages:
                print(f"   ⚠️ Extraction mismatch: {total_expected_pages - total_extracted_pages:,} pages")
            if total_valid_pages != total_extracted_pages:
                print(f"   ⚠️ Validation failures: {total_extracted_pages - total_valid_pages:,} corrupt images")
        else:
            # No new extractions, but we still validated existing containers
            print(f"   (All containers were already extracted)")
            print(f"   Total pages found: {total_extracted_pages:,}")
            print(f"   Valid pages written to manifest: {total_valid_pages:,}")
            if total_valid_pages != total_extracted_pages:
                print(f"   ⚠️ Validation failures: {total_extracted_pages - total_valid_pages:,} corrupt images")
    
    if containers_with_mismatches:
        print(f"\n⚠️ {len(containers_with_mismatches)} containers had page count mismatches:")
        for msg in containers_with_mismatches[:10]:  # Show first 10
            print(f"     • {msg}")
        if len(containers_with_mismatches) > 10:
            print(f"     ... and {len(containers_with_mismatches) - 10} more (see log)")
    print(f"\n⚠️ Check {log_path} for any errors during extraction or validation.")

def main():
    parser = argparse.ArgumentParser(description="Extracts comic containers and creates a master manifest CSV.")
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True, help="One or more root directories for comic containers and/or loose images.")
    parser.add_argument('--extraction_dir', type=str, required=True, help="A single, central directory to store all extracted images.")
    parser.add_argument('--output_csv', type=str, default="master_manifest.csv", help="Path for the output master manifest CSV file.")
    args = parser.parse_args()

    create_master_manifest(args.input_dirs, args.extraction_dir, args.output_csv)

if __name__ == "__main__":
    main()