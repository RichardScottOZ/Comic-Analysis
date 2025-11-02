#!/usr/bin/env python3
"""
Fix duplicate folder names by creating unique CBZ filenames.

This script identifies folders with duplicate names (ignoring Calibre index numbers)
and creates a new directory structure with unique CBZ filenames by adding
numbered suffixes before the .cbz extension.
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm

def extract_base_name(folder_name):
    """Extract the base name without Calibre index number."""
    # Remove Calibre index pattern: (number) at the end
    base_name = re.sub(r'\s*\(\d+\)\s*$', '', folder_name)
    return base_name.strip()

def find_duplicate_folders(source_dir):
    """Find folders with duplicate base names recursively through all subdirectories."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Group folders by base name
    folder_groups = defaultdict(list)
    
    # Recursively find all directories
    for folder in source_path.rglob("*"):
        if folder.is_dir():
            base_name = extract_base_name(folder.name)
            folder_groups[base_name].append(folder)
    
    # Find groups with multiple folders
    duplicates = {base_name: folders for base_name, folders in folder_groups.items() 
                 if len(folders) > 1}
    
    return duplicates

def find_cbz_files(folder_path):
    """Find all CBZ files in a folder."""
    cbz_files = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() == '.cbz':
            cbz_files.append(file)
    return cbz_files

def create_unique_filename(original_path, counter):
    """Create a unique filename by adding counter before .cbz extension."""
    stem = original_path.stem
    suffix = original_path.suffix
    return f"{stem}_{counter}{suffix}"

def process_duplicate_folders(duplicates, source_dir, output_dir, dry_run=False):
    """Process duplicate folders and create unique CBZ filenames while preserving directory structure."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    processed_files = 0
    
    # Count total files for progress bar
    for base_name, folders in duplicates.items():
        for folder in folders:
            cbz_files = find_cbz_files(folder)
            total_files += len(cbz_files)
    
    print(f"Found {len(duplicates)} duplicate folder groups")
    print(f"Total CBZ files to process: {total_files}")
    
    if dry_run:
        print("\n=== DRY RUN - No files will be copied ===")
    
    with tqdm(total=total_files, desc="Processing CBZ files") as pbar:
        for base_name, folders in duplicates.items():
            print(f"\nProcessing group: '{base_name}' ({len(folders)} folders)")
            
            # Create a counter for this group
            group_counter = 1
            
            for folder in folders:
                cbz_files = find_cbz_files(folder)
                
                if not cbz_files:
                    print(f"  ⚠️  No CBZ files found in: {folder.name}")
                    continue
                
                print(f"  📁 {folder.name} ({len(cbz_files)} CBZ files)")
                
                # Create the output folder (preserving directory structure)
                relative_path = folder.relative_to(source_path)
                output_folder = output_path / relative_path
                
                if not dry_run:
                    output_folder.mkdir(parents=True, exist_ok=True)
                
                for cbz_file in cbz_files:
                    # Create unique filename
                    unique_filename = create_unique_filename(cbz_file, group_counter)
                    output_file = output_folder / unique_filename
                    
                    if dry_run:
                        print(f"    Would copy: {folder.name}/{cbz_file.name} → {relative_path}/{unique_filename}")
                    else:
                        try:
                            shutil.copy2(cbz_file, output_file)
                            print(f"    ✅ Copied: {cbz_file.name} → {unique_filename}")
                        except Exception as e:
                            print(f"    ❌ Failed to copy {cbz_file.name}: {e}")
                    
                    group_counter += 1
                    processed_files += 1
                    pbar.update(1)
    
    return processed_files

def main():
    parser = argparse.ArgumentParser(description='Fix duplicate folder names by creating unique CBZ filenames')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing folders with duplicate names')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for unique CBZ files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    
    args = parser.parse_args()
    
    print(f"Scanning for duplicate folders in: {args.source_dir}")
    
    try:
        # Find duplicate folders
        duplicates = find_duplicate_folders(args.source_dir)
        
        if not duplicates:
            print("No duplicate folders found!")
            return
        
        print(f"\nFound {len(duplicates)} groups of duplicate folders:")
        for base_name, folders in duplicates.items():
            print(f"  '{base_name}': {len(folders)} folders")
            for folder in folders:
                cbz_count = len(find_cbz_files(folder))
                print(f"    - {folder.name} ({cbz_count} CBZ files)")
        
        # Process the duplicates
        processed_count = process_duplicate_folders(duplicates, args.source_dir, args.output_dir, args.dry_run)
        
        print(f"\n=== Summary ===")
        print(f"Duplicate groups processed: {len(duplicates)}")
        print(f"CBZ files processed: {processed_count}")
        
        if not args.dry_run:
            print(f"Files copied to: {args.output_dir}")
        else:
            print("DRY RUN COMPLETED - No files were copied")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 