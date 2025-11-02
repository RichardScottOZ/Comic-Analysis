#!/usr/bin/env python3
"""
Clean up extracted duplicates by identifying and removing non-unique folders.

This script uses the same duplicate detection method as fix_duplicate_names.py
to identify folders that would cause overwrites during extraction, and removes
them from the extracted directory.
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

def get_folders_to_remove_from_extracted(duplicates, extracted_dir):
    """Based on duplicates found in source, determine which folders to remove from extracted."""
    extracted_path = Path(extracted_dir)
    if not extracted_path.exists():
        raise ValueError(f"Extracted directory does not exist: {extracted_dir}")
    
    folders_to_remove = []
    
    for base_name, source_folders in duplicates.items():
        print(f"\nProcessing duplicate group: '{base_name}'")
        
        # Sort source folders to ensure consistent ordering
        sorted_source_folders = sorted(source_folders, key=lambda x: x.name)
        
        # Keep the first source folder, remove the rest
        keep_source_folder = sorted_source_folders[0]
        remove_source_folders = sorted_source_folders[1:]
        
        print(f"  Source: Keeping '{keep_source_folder.name}', removing {len(remove_source_folders)} others")
        
        # Check if the base folder exists in extracted directory
        extracted_folder = extracted_path / base_name
        
        if extracted_folder.exists():
            # We found the extracted folder - this is the one that would cause overwrites
            # We should remove it since it corresponds to the duplicate source folders
            folders_to_remove.append(extracted_folder)
            print(f"    Will remove from extracted: {base_name} (duplicate base folder)")
            print(f"      Full path: {extracted_folder}")
        else:
            print(f"    ⚠️  Not found in extracted: {base_name}")
    
    return folders_to_remove

def count_files_in_folder(folder_path):
    """Count all files in a folder recursively."""
    count = 0
    for item in folder_path.rglob("*"):
        if item.is_file():
            count += 1
    return count

def cleanup_extracted_folders(folders_to_remove, dry_run=False):
    """Remove the specified folders from the extracted directory."""
    total_folders = len(folders_to_remove)
    removed_folders = 0
    total_files_removed = 0
    
    print(f"\nTotal folders to remove from extracted: {total_folders}")
    
    if dry_run:
        print("\n=== DRY RUN - No folders will be removed ===")
    
    with tqdm(total=total_folders, desc="Removing folders from extracted") as pbar:
        for folder in folders_to_remove:
            file_count = count_files_in_folder(folder)
            
            if dry_run:
                print(f"Would remove: {folder.name} ({file_count} files)")
            else:
                try:
                    shutil.rmtree(folder)
                    print(f"❌ Removed: {folder.name} ({file_count} files)")
                    total_files_removed += file_count
                except Exception as e:
                    print(f"⚠️  Failed to remove {folder.name}: {e}")
            
            removed_folders += 1
            pbar.update(1)
    
    return removed_folders, total_files_removed

def main():
    parser = argparse.ArgumentParser(description='Clean up extracted duplicates by removing non-unique folders')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing folders with duplicate names (e.g., _rename)')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Extracted directory to clean up (e.g., _rename_extracted)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually removing folders')
    
    args = parser.parse_args()
    
    print(f"Scanning for duplicate folders in: {args.source_dir}")
    
    try:
        # Find duplicate folders in source directory
        duplicates = find_duplicate_folders(args.source_dir)
        
        if not duplicates:
            print("No duplicate folders found in source directory!")
            return
        
        print(f"\nFound {len(duplicates)} groups of duplicate folders in source:")
        for base_name, folders in duplicates.items():
            print(f"  '{base_name}': {len(folders)} folders")
            for folder in folders:
                print(f"    - {folder.name}")
        
        # Get folders to remove from extracted
        folders_to_remove = get_folders_to_remove_from_extracted(duplicates, args.extracted_dir)
        
        if not folders_to_remove:
            print("\nNo folders to remove from extracted directory!")
            return
        
        # Clean up the extracted folders
        removed_count, files_removed = cleanup_extracted_folders(folders_to_remove, args.dry_run)
        
        print(f"\n=== Summary ===")
        print(f"Duplicate groups found in source: {len(duplicates)}")
        print(f"Folders removed from extracted: {removed_count}")
        print(f"Files removed from extracted: {files_removed}")
        
        if not args.dry_run:
            print(f"Cleanup completed for: {args.extracted_dir}")
        else:
            print("DRY RUN COMPLETED - No folders were removed")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 