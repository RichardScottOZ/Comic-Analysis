import os
import csv
import shutil
import zipfile
import argparse
from PIL import Image
from pathlib import Path
from comix.utils import generate_hash
import tempfile

def extract_cbz(cbz_path, temp_dir):
    """Extract a CBZ file to a temporary directory."""
    with zipfile.ZipFile(cbz_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    # Get list of image files (supporting jpg, jpeg, png)
    image_files = []
    for root, _, files in os.walk(temp_dir):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def main(args):
    input_path = Path(args.input_path)
    output_images_path = Path(args.output_path) / 'images'
    output_images_path.mkdir(parents=True, exist_ok=True)

    mapping_csv_path = Path(args.output_path) / 'book_chapter_hash_mapping.csv'

    book_hash_map = {}
    with open(mapping_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['book_chapter_hash', 'new_image_name', 'book_name', 'chapter_name', 'original_image_name'])

        # Process each CBZ file
        cbz_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() == '.cbz'])
        if args.limit:
            cbz_files = cbz_files[:args.limit]

        for cbz_path in cbz_files:
            book_name = cbz_path.stem
            print(f"Processing {book_name}...")

            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract CBZ and get list of image files
                image_files = extract_cbz(cbz_path, temp_dir)
                
                # Treat each CBZ as a single chapter
                chapter_name = "chapter1"  # You can modify this if your CBZ files have chapter information
                book_chapter_hash = generate_hash(f'{book_name}_{chapter_name}')
                
                output_chapter_images_path = output_images_path / book_chapter_hash
                if output_chapter_images_path.exists() and not args.override:
                    print(f"Skipping {book_name} as output directory already exists")
                    continue
                
                output_chapter_images_path.mkdir(exist_ok=True)

                # Process each image in the CBZ
                for image_no, image_path in enumerate(image_files):
                    new_image_name = f'{image_no:03d}'
                    original_image_name = os.path.basename(image_path)
                    
                    try:
                        # Convert any image format to JPG
                        img = Image.open(image_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        shape = img.size
                        if shape[0] == 0 or shape[1] == 0:
                            raise ValueError(f"Invalid image shape: {shape}")
                        
                        # Save as JPG
                        output_path = output_chapter_images_path / f'{new_image_name}.jpg'
                        img.save(output_path, 'JPEG')
                        img.close()
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
                        continue

                    csv_writer.writerow([book_chapter_hash, new_image_name, book_name, chapter_name, original_image_name])
                    book_hash_map[f'{book_name}/{chapter_name}/{original_image_name}'] = (book_chapter_hash, new_image_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Process 2000AD CBZ files.')
    parser.add_argument('-i', '--input-path', type=str, help='Path to the 2000AD folder containing CBZ files', default='2000AD')
    parser.add_argument('-o', '--output-path', type=str, help='Path to the output folder', default='data/datasets.unify/2000ad')
    parser.add_argument('-l', '--limit', type=int, help='Limit the number of books processed')
    parser.add_argument('--override', action='store_true', help='Override existing image folders if they exist')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args) 