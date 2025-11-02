#!/usr/bin/env python3
"""
Generate canonical mapping between image files, VLM JSONs, and COCO IDs.

This untangles the mess created by different naming algorithms by replicating
the EXACT logic from each source file.

Output: CSV with exact paths and existence checks for all ~337K records
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import csv

def extract_coco_mappings(coco_file):
    """Extract all image IDs and file_names from COCO JSON."""
    print(f"Loading COCO file: {coco_file}")
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Build mapping: image_id -> image info
    id_to_info = {}
    
    for img in coco_data.get('images', []):
        img_id = img.get('id')
        if img_id:
            id_to_info[img_id] = {
                'id': img_id,
                'file_name': img.get('file_name', ''),
                'width': img.get('width'),
                'height': img.get('height')
            }
    
    print(f"Loaded {len(id_to_info)} image records from COCO")
    
    # Show samples
    print("\nSample COCO records:")
    for i, (img_id, info) in enumerate(list(id_to_info.items())[:3]):
        print(f"  ID: {img_id[:100]}...")
        print(f"  file_name: {info['file_name']}")
        print()
    
    return id_to_info, coco_data

def build_image_index(image_dir):
    """Build index of all actual image files on disk."""
    print(f"Indexing images in: {image_dir}")
    image_index = {}  # normalized_path -> full_path
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                full_path = os.path.join(root, file)
                # Create normalized key (relative path from image_dir)
                try:
                    rel_path = os.path.relpath(full_path, image_dir)
                    norm_key = rel_path.replace('\\', '/').lower()
                    image_index[norm_key] = full_path
                except Exception as e:
                    print(f"Warning: Could not create relative path for {full_path}: {e}")
    
    print(f"Indexed {len(image_index)} images")
    return image_index

def build_vlm_index(vlm_dir):
    """Build index of all VLM JSON files."""
    print(f"Indexing VLM JSONs in: {vlm_dir}")
    vlm_index = {}  # normalized_stem -> full_path
    
    for root, dirs, files in os.walk(vlm_dir):
        for file in files:
            if file.lower().endswith('.json'):
                full_path = os.path.join(root, file)
                # Create normalized key (relative path without .json extension)
                try:
                    rel_path = os.path.relpath(full_path, vlm_dir)
                    # Remove .json extension
                    stem = os.path.splitext(rel_path)[0]
                    norm_key = stem.replace('\\', '/').lower()
                    vlm_index[norm_key] = full_path
                except Exception as e:
                    print(f"Warning: Could not create relative path for {full_path}: {e}")
    
    print(f"Indexed {len(vlm_index)} VLM JSONs")
    return vlm_index

def reconstruct_faster_rcnn_id(image_path):
    """
    Reconstruct the COCO image_id using EXACT logic from faster_rcnn_calibre.py.
    
    From faster_rcnn_calibre.py:
    - Line 40: book_chapters.append(root)  # Full directory path
    - Line 55: page_no = self.image_paths[idx].split('\\')[-1]  # Full filename with extension
    - Line 153: image_id = f"{book_chapter}_{page_no}"
    
    So: image_id = "{full_directory_path}_{filename_with_extension}"
    
    Example:
    Input:  E:\CalibreComics_extracted\13thfloor vol1 - Unknown\JPG4CBZ_0001.jpg
    Output: E:\CalibreComics_extracted\13thfloor vol1 - Unknown_JPG4CBZ_0001.jpg
    """
    try:
        # Get the directory part (book_chapter) - full path
        directory = os.path.dirname(image_path)
        
        # Get the filename part (page_no) - just the filename 
        filename = os.path.basename(image_path)
        
        # Concatenate with underscore exactly as faster_rcnn does
        image_id = f"{directory}_{filename}"
        
        return image_id
    except Exception as e:
        print(f"Warning: Could not reconstruct COCO ID for {image_path}: {e}")
        return None

def reconstruct_vlm_filename(image_path, image_dir):
    """
    Reconstruct the VLM JSON filename using EXACT logic from batch_comic_analysis_multi.py.
    
    From batch_comic_analysis_multi.py line 508-511:
        relative_path = image_file.relative_to(input_dir)
        unique_id = str(relative_path.with_suffix('')).replace('\\', '_').replace('/', '_')
        output_path = output_dir / f"{unique_id}.json"
    
    Example:
    Input:  E:\CalibreComics_extracted\13thfloor vol1 - Unknown\JPG4CBZ_0001.jpg
    Output: 13thfloor vol1 - Unknown_JPG4CBZ_0001.json
    """
    try:
        # Get relative path from image_dir
        rel_path = os.path.relpath(image_path, image_dir)
        
        # Remove extension
        stem = os.path.splitext(rel_path)[0]
        
        # Replace path separators with underscores
        unique_id = stem.replace('\\', '_').replace('/', '_')
        
        return f"{unique_id}.json"
    except Exception as e:
        print(f"Warning: Could not reconstruct VLM filename for {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate canonical mapping between images, VLM JSONs, and COCO IDs')
    parser.add_argument('--image_dir', required=True, help='Directory containing extracted images (e.g., E:\\CalibreComics_extracted)')
    parser.add_argument('--vlm_dir', required=True, help='Directory containing VLM analysis JSONs (e.g., E:\\CalibreComics_analysis)')
    parser.add_argument('--coco_file', required=True, help='COCO predictions JSON file (e.g., E:\\CalibreComics\\test_dections\\predictions.json)')
    parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records for testing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CANONICAL IMAGE-VLM-COCO MAPPING GENERATOR")
    print("=" * 80)
    
    # Step 1: Load COCO data
    coco_id_to_info, coco_data = extract_coco_mappings(args.coco_file)
    
    # Step 2: Index actual images on disk
    image_index = build_image_index(args.image_dir)
    
    # Step 3: Index VLM JSONs
    vlm_index = build_vlm_index(args.vlm_dir)
    
    # Step 4: Generate canonical mapping
    print("\nGenerating canonical mapping...")
    
    records = []
    
    # Iterate through all images on disk
    image_files = sorted(image_index.values())
    if args.limit:
        image_files = image_files[:args.limit]
        print(f"Limited to {args.limit} images for testing")
    
    for image_path in tqdm(image_files, desc="Mapping images"):
        record = {}
        
        # 1. Image path (ground truth)
        record['image_path'] = image_path
        record['image_exists'] = os.path.exists(image_path)
        
        # 2. Reconstruct COCO image_id using EXACT logic from faster_rcnn_calibre.py
        coco_id_reconstructed = reconstruct_faster_rcnn_id(image_path)
        record['coco_id_reconstructed'] = coco_id_reconstructed if coco_id_reconstructed else ''
        
        # 3. Try to find actual COCO ID by searching for matches
        # The COCO ID might have extra extensions (.png.png) so we need to search
        actual_coco_id = None
        actual_coco_info = None
        
        # First try exact match with reconstructed ID
        if coco_id_reconstructed and coco_id_reconstructed in coco_id_to_info:
            actual_coco_id = coco_id_reconstructed
            actual_coco_info = coco_id_to_info[coco_id_reconstructed]
            # DEBUG
            if '25thanniversary' in image_path.lower():
                print(f"\n=== EXACT MATCH FOUND ===")
                print(f"Image: {image_path}")
                print(f"Reconstructed ID: {coco_id_reconstructed}")
                print(f"Found in COCO: {actual_coco_id}")
                print(f"File name: {actual_coco_info.get('file_name')}")
                print("=" * 50)
        else:
            # Try fuzzy match - look for COCO IDs that contain parts of our path
            try:
                rel_path = os.path.relpath(image_path, args.image_dir)
                # Remove extension and normalize
                stem = os.path.splitext(rel_path)[0]
                stem_normalized = stem.replace('\\', '_').replace('/', '_').lower()
                
                # Search for COCO IDs containing this stem
                for coco_id, info in coco_id_to_info.items():
                    coco_id_normalized = coco_id.replace('\\', '_').replace('/', '_').lower()
                    if stem_normalized in coco_id_normalized:
                        actual_coco_id = coco_id
                        actual_coco_info = info
                        # DEBUG
                        if '25thanniversary' in image_path.lower():
                            print(f"\n=== FUZZY MATCH FOUND ===")
                            print(f"Image: {image_path}")
                            print(f"Stem normalized: {stem_normalized}")
                            print(f"Matched COCO ID: {coco_id}")
                            print(f"COCO ID normalized: {coco_id_normalized}")
                            print(f"File name: {info.get('file_name')}")
                            print("=" * 50)
                        break
            except Exception as e:
                # Don't silently pass - this might be hiding bugs
                print(f"Warning: Fuzzy match failed for {image_path}: {e}")
                pass
        
        # Record what we found - NEVER write reconstructed value here!
        if actual_coco_id and actual_coco_info:
            # Write the ACTUAL values from the COCO dict, not our reconstructed key
            record['predictions_json_id'] = actual_coco_info.get('id', '')  # From dict VALUE
            record['predictions_json_filename'] = actual_coco_info.get('file_name', '')  # From dict VALUE
            record['coco_id_exists'] = True
        else:
            # If not found, leave empty - do NOT write reconstructed value!
            record['predictions_json_id'] = ''
            record['predictions_json_filename'] = ''
            record['coco_id_exists'] = False
        
        # 4. Reconstruct VLM JSON filename using EXACT logic from batch_comic_analysis_multi.py
        vlm_filename = reconstruct_vlm_filename(image_path, args.image_dir)
        record['vlm_json_filename'] = vlm_filename if vlm_filename else ''
        
        # Look up the actual VLM JSON path
        vlm_json_path = None
        if vlm_filename:
            # Try exact match first (flattened in vlm_dir root)
            candidate = os.path.join(args.vlm_dir, vlm_filename)
            if os.path.exists(candidate):
                vlm_json_path = candidate
            else:
                # Try looking in the index (normalized)
                vlm_stem = os.path.splitext(vlm_filename)[0].lower()
                if vlm_stem in vlm_index:
                    vlm_json_path = vlm_index[vlm_stem]
        
        record['vlm_json_path'] = vlm_json_path if vlm_json_path else ''
        record['vlm_json_exists'] = os.path.exists(vlm_json_path) if vlm_json_path else False
        
        # 5. Add relative paths for readability
        try:
            record['image_rel_path'] = os.path.relpath(image_path, args.image_dir)
        except:
            record['image_rel_path'] = image_path
        
        if vlm_json_path:
            try:
                record['vlm_json_rel_path'] = os.path.relpath(vlm_json_path, args.vlm_dir)
            except:
                record['vlm_json_rel_path'] = vlm_json_path
        else:
            record['vlm_json_rel_path'] = ''
        
        records.append(record)
    
    # Step 5: Write to CSV
    print(f"\nWriting {len(records)} records to {args.output_csv}")
    
    fieldnames = [
        'image_path',
        'image_exists',
        'image_rel_path',
        'predictions_json_id',
        'predictions_json_filename',
        'coco_id_exists',
        'coco_id_reconstructed',
        'vlm_json_filename',
        'vlm_json_path',
        'vlm_json_exists',
        'vlm_json_rel_path'
    ]
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    # Step 6: Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total = len(records)
    images_exist = sum(1 for r in records if r['image_exists'])
    coco_exists = sum(1 for r in records if r['coco_id_exists'])
    vlm_exists = sum(1 for r in records if r['vlm_json_exists'])
    all_three = sum(1 for r in records if r['image_exists'] and r['coco_id_exists'] and r['vlm_json_exists'])
    
    print(f"Total records: {total:,}")
    print(f"Images exist: {images_exist:,} ({images_exist/total*100:.1f}%)")
    print(f"COCO IDs exist: {coco_exists:,} ({coco_exists/total*100:.1f}%)")
    print(f"VLM JSONs exist: {vlm_exists:,} ({vlm_exists/total*100:.1f}%)")
    print(f"All three exist: {all_three:,} ({all_three/total*100:.1f}%)")
    
    # Missing analysis
    missing_coco = total - coco_exists
    missing_vlm = total - vlm_exists
    
    if missing_coco > 0:
        print(f"\n⚠️  {missing_coco:,} images missing from COCO predictions.json")
        print("Sample missing COCO IDs (first 5):")
        count = 0
        for r in records:
            if not r['coco_id_exists'] and count < 5:
                print(f"  - Image: {r['image_rel_path']}")
                print(f"    Reconstructed ID: {r['coco_id_reconstructed'][:100] if r['coco_id_reconstructed'] else 'N/A'}...")
                count += 1
        
        # Also show some that DO match to compare
        print("\nSample MATCHED records for comparison (first 3):")
        count = 0
        for r in records:
            if r['coco_id_exists'] and count < 3:
                print(f"  - Image: {r['image_rel_path']}")
                print(f"    Reconstructed: {r['coco_id_reconstructed'][:100] if r['coco_id_reconstructed'] else 'N/A'}...")
                print(f"    predictions.json 'id':        {r['predictions_json_id'][:100] if r['predictions_json_id'] else 'N/A'}...")
                print(f"    predictions.json 'file_name': {r['predictions_json_filename']}")
                print()
                count += 1
    
    if missing_vlm > 0:
        print(f"\n⚠️  {missing_vlm:,} images missing VLM analysis JSONs")
        print("Sample missing VLM files (first 5):")
        count = 0
        for r in records:
            if not r['vlm_json_exists'] and count < 5:
                print(f"  - Image: {r['image_rel_path']}")
                print(f"    Expected VLM: {r['vlm_json_filename']}")
                count += 1
    
    print(f"\n✓ Mapping complete! Saved to: {args.output_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
