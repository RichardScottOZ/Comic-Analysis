#!/usr/bin/env python3
"""
Check if all folders from source directory have corresponding entries in extracted directory.
Identifies missing extractions and potential issues with the extraction process.
"""

import argparse
from pathlib import Path
import os
import shutil
import json
from collections import defaultdict
import re # Added for analyze_naming_mismatches

def find_comic_folders(source_dir):
    """Find all folders in source directory that contain comic files (recursively)."""
    source_path = Path(source_dir)
    comic_folders = []
    
    print(f"Scanning source directory: {source_dir}")
    
    # Find all folders that contain comic files
    for item in source_path.rglob("*"):
        if item.is_dir():
            # Check if this folder contains comic files
            comic_files = list(item.glob("*.cbz")) + list(item.glob("*.cbr"))
            if comic_files:
                comic_folders.append({
                    'name': item.name,  # Use the actual folder name
                    'path': item,
                    'comic_files': comic_files,
                    'comic_count': len(comic_files),
                    'relative_path': str(item.relative_to(source_path))
                })
    
    print(f"Found {len(comic_folders)} folders with comic files")
    
    # Show some sample folders for verification
    if len(comic_folders) > 0:
        print("Sample folders found:")
        for i, folder in enumerate(comic_folders[:10]):
            print(f"  {i+1}: {folder['name']} ({folder['comic_count']} comics)")
            print(f"      Path: {folder['relative_path']}")
        if len(comic_folders) > 10:
            print(f"  ... and {len(comic_folders) - 10} more folders")
    
    return comic_folders

def find_extracted_folders(extracted_dir):
    """Find all folders in extracted directory."""
    extracted_path = Path(extracted_dir)
    extracted_folders = []
    
    print(f"Scanning extracted directory: {extracted_dir}")
    
    for item in extracted_path.iterdir():
        if item.is_dir():
            # Count image files in this folder
            image_files = list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png"))
            extracted_folders.append({
                'name': item.name,
                'path': item,
                'image_count': len(image_files)
            })
    
    print(f"Found {len(extracted_folders)} extracted folders")
    
    # Show some sample folders for verification
    if len(extracted_folders) > 0:
        print("Sample extracted folders:")
        for i, folder in enumerate(extracted_folders[:10]):
            print(f"  {i+1}: {folder['name']} ({folder['image_count']} images)")
        if len(extracted_folders) > 10:
            print(f"  ... and {len(extracted_folders) - 10} more folders")
    
    return extracted_folders

def compare_folders(source_folders, extracted_folders):
    """Compare source and extracted folders to find mismatches."""
    print("\nComparing folders...")
    
    # Create lookup dictionaries using exact names (no normalization)
    source_lookup = {f['name']: f for f in source_folders}
    extracted_lookup = {f['name']: f for f in extracted_folders}
    
    # Find missing extractions
    missing_extractions = []
    for source_name, source_info in source_lookup.items():
        if source_name not in extracted_lookup:
            missing_extractions.append(source_info)
    
    # Find extra extracted folders (not in source)
    extra_extracted = []
    for extracted_name, extracted_info in extracted_lookup.items():
        if extracted_name not in source_lookup:
            extra_extracted.append(extracted_info)
    
    # Find potential name mismatches (similar names)
    potential_mismatches = []
    for source_name, source_info in source_lookup.items():
        if source_name not in extracted_lookup:
            # Look for similar names
            for extracted_name, extracted_info in extracted_lookup.items():
                if source_name in extracted_name or extracted_name in source_name:
                    potential_mismatches.append({
                        'source': source_info,
                        'extracted': extracted_info,
                        'source_name': source_name,
                        'extracted_name': extracted_name
                    })
    
    return {
        'missing_extractions': missing_extractions,
        'extra_extracted': extra_extracted,
        'potential_mismatches': potential_mismatches,
        'source_count': len(source_folders),
        'extracted_count': len(extracted_folders),
        'source_lookup': source_lookup,
        'extracted_lookup': extracted_lookup
    }

def analyze_naming_mismatches(source_folders):
    """Analyze cases where CBZ filenames would cause overwrites during extraction."""
    print("\nAnalyzing naming mismatches...")
    
    # Group folders by what their CBZ files would create as folder names
    cbz_folder_groups = {}
    
    for folder in source_folders:
        folder_name = folder['name']
        comic_files = folder['comic_files']
        
        for comic_file in comic_files:
            cbz_folder_name = comic_file.stem  # what extract_calibre_comics.py would create
            
            if cbz_folder_name not in cbz_folder_groups:
                cbz_folder_groups[cbz_folder_name] = []
            
            cbz_folder_groups[cbz_folder_name].append({
                'folder_name': folder_name,
                'comic_file': comic_file,
                'relative_path': folder['relative_path']
            })
    
    # Find cases where multiple folders would create the same CBZ folder name
    naming_issues = []
    
    for cbz_folder_name, folders in cbz_folder_groups.items():
        if len(folders) > 1:
            # This CBZ folder name would be created by multiple source folders - overwrite problem!
            for folder_info in folders:
                # Remove database index like (696) from folder name for suggested rename
                clean_folder_name = re.sub(r'\s*\(\d+\)\s*$', '', folder_info['folder_name'])
                
                naming_issues.append({
                    'folder_name': folder_info['folder_name'],
                    'clean_folder_name': clean_folder_name,
                    'comic_name': cbz_folder_name,
                    'comic_file': folder_info['comic_file'],
                    'relative_path': folder_info['relative_path'],
                    'overwrite_count': len(folders)
                })
    
    return naming_issues

def generate_report(comparison_results, naming_issues):
    """Generate a detailed report of the comparison."""
    print("\n" + "="*80)
    print("EXTRACTION CHECK REPORT")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  Source folders: {comparison_results['source_count']}")
    print(f"  Extracted folders: {comparison_results['extracted_count']}")
    print(f"  Missing extractions: {len(comparison_results['missing_extractions'])}")
    print(f"  Extra extracted folders: {len(comparison_results['extra_extracted'])}")
    print(f"  Potential name mismatches: {len(comparison_results['potential_mismatches'])}")
    print(f"  Naming issues found: {len(naming_issues)}")
    
    # Debug: Show sample names
    source_lookup = comparison_results['source_lookup']
    extracted_lookup = comparison_results['extracted_lookup']
    
    print(f"\nDEBUG - Sample source folder names:")
    for i, (name, folder) in enumerate(list(source_lookup.items())[:5]):
        print(f"  '{name}'")
    
    print(f"DEBUG - Sample extracted folder names:")
    for i, (name, folder) in enumerate(list(extracted_lookup.items())[:5]):
        print(f"  '{name}'")
    
    # Debug: Show Hellboy names specifically
    hellboy_source = [name for name in source_lookup.keys() if "hellboy" in name.lower()]
    hellboy_extracted = [name for name in extracted_lookup.keys() if "hellboy" in name.lower()]
    
    print(f"\nDEBUG - Hellboy source folders ({len(hellboy_source)}):")
    for name in hellboy_source[:10]:
        print(f"  '{name}'")
    
    print(f"DEBUG - Hellboy extracted folders ({len(hellboy_extracted)}):")
    for name in hellboy_extracted[:10]:
        print(f"  '{name}'")
    
    if naming_issues:
        print(f"\n🔧 OVERWRITE PROBLEMS FOUND ({len(naming_issues)}):")
        print("-" * 50)
        print("These files would overwrite each other during extraction and need renaming:")
        for issue in naming_issues:
            print(f"  Folder: {issue['folder_name']}")
            print(f"  Current CBZ: {issue['comic_file'].name}")
            print(f"  Would create folder: {issue['comic_name']} (shared by {issue['overwrite_count']} files)")
            print(f"  Should be renamed to: {issue['clean_folder_name']}.cbz")
            print(f"  Path: {issue['relative_path']}")
            print()
    
    if comparison_results['missing_extractions']:
        print(f"\nMISSING EXTRACTIONS ({len(comparison_results['missing_extractions'])}):")
        print("-" * 50)
        for folder in comparison_results['missing_extractions']:
            print(f"  {folder['name']} ({folder['comic_count']} comic files)")
            print(f"    Source path: {folder['relative_path']}")
            print(f"    Comic files: {[f.name for f in folder['comic_files']]}")
            print()
    
    if comparison_results['potential_mismatches']:
        print(f"\nPOTENTIAL NAME MISMATCHES ({len(comparison_results['potential_mismatches'])}):")
        print("-" * 50)
        for mismatch in comparison_results['potential_mismatches']:
            print(f"  Source: {mismatch['source']['name']}")
            print(f"  Extracted: {mismatch['extracted']['name']}")
            print()
    
    if comparison_results['extra_extracted']:
        print(f"\nEXTRA EXTRACTED FOLDERS ({len(comparison_results['extra_extracted'])}):")
        print("-" * 50)
        for folder in comparison_results['extra_extracted']:
            print(f"  {folder['name']} ({folder['image_count']} images)")
            print(f"    Path: {folder['path']}")
            print()

def create_renamed_copy(source_dir, output_dir, naming_issues):
    """Create a copy of only the problematic folders with renamed files to avoid overwrites."""
    print(f"\nCreating renamed copy at: {output_dir}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get unique folders that have naming issues
    problematic_folders = set()
    files_to_rename = {}
    
    for issue in naming_issues:
        # Get the folder path
        folder_path = issue['comic_file'].parent
        problematic_folders.add(folder_path)
        
        # Create a mapping from original file path to new filename
        original_file = issue['comic_file']
        new_filename = f"{issue['clean_folder_name']}.{original_file.suffix}"
        files_to_rename[original_file] = new_filename
    
    print(f"Found {len(problematic_folders)} folders with naming issues")
    print(f"Found {len(files_to_rename)} files that need renaming")
    
    # Copy only the problematic folders
    copied_count = 0
    renamed_count = 0
    
    for folder_path in problematic_folders:
        # Calculate relative path from source
        relative_folder_path = folder_path.relative_to(source_path)
        target_folder = output_path / relative_folder_path
        
        print(f"Copying folder: {relative_folder_path}")
        
        # Create target directory
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy all files in this folder
        for source_file in folder_path.iterdir():
            if source_file.is_file():
                target_file = target_folder / source_file.name
                
                # Check if this file needs renaming
                if source_file in files_to_rename:
                    # Rename the file
                    new_filename = files_to_rename[source_file]
                    target_file = target_folder / new_filename
                    print(f"  Renaming: {source_file.name} -> {new_filename}")
                    renamed_count += 1
                else:
                    print(f"  Copying: {source_file.name}")
                
                # Copy the file
                shutil.copy2(source_file, target_file)
                copied_count += 1
    
    print(f"\nCopy completed:")
    print(f"  Folders copied: {len(problematic_folders)}")
    print(f"  Total files copied: {copied_count}")
    print(f"  Files renamed: {renamed_count}")
    print(f"  Output directory: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Check if all source folders have corresponding extracted folders')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing original comic folders')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Extracted directory containing processed folders')
    parser.add_argument('--output-report', type=str, default=None,
                       help='Save report to file (optional)')
    parser.add_argument('--create-renamed-copy', type=str, default=None,
                       help='Create a renamed copy of source directory to avoid overwrites')
    parser.add_argument('--save-naming-issues', type=str, default=None,
                       help='Save naming issues to JSON file for fix_extraction.py')
    
    args = parser.parse_args()
    
    # Validate directories
    source_path = Path(args.source_dir)
    extracted_path = Path(args.extracted_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory not found: {args.source_dir}")
        return
    
    if not extracted_path.exists():
        print(f"Error: Extracted directory not found: {args.extracted_dir}")
        return
    
    # Find folders
    source_folders = find_comic_folders(args.source_dir)
    extracted_folders = find_extracted_folders(args.extracted_dir)
    
    # Compare
    comparison_results = compare_folders(source_folders, extracted_folders)
    
    # Analyze name lengths
    long_names = []
    
    # Analyze naming mismatches
    naming_issues = analyze_naming_mismatches(source_folders)
    
    # Generate report
    generate_report(comparison_results, naming_issues)
    
    # Create renamed copy if requested
    if args.create_renamed_copy and naming_issues:
        create_renamed_copy(args.source_dir, args.create_renamed_copy, naming_issues)
        print(f"\n✅ Renamed copy created! You can now run extract_calibre_comics.py on:")
        print(f"   {args.create_renamed_copy}")
        print(f"   This will avoid the overwrite problems.")
    
    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            generate_report(comparison_results, naming_issues)
            
            sys.stdout = original_stdout
        print(f"\nReport saved to: {args.output_report}")
    
    # Save naming issues to JSON if requested
    if args.save_naming_issues:
        try:
            # Convert WindowsPath objects to strings for JSON serialization
            json_safe_issues = []
            for issue in naming_issues:
                json_safe_issue = {
                    'folder_name': issue['folder_name'],
                    'clean_folder_name': issue['clean_folder_name'],
                    'comic_name': issue['comic_name'],
                    'comic_file': str(issue['comic_file']),  # Convert Path to string
                    'relative_path': issue['relative_path'],
                    'overwrite_count': issue['overwrite_count']
                }
                json_safe_issues.append(json_safe_issue)
            
            with open(args.save_naming_issues, 'w', encoding='utf-8') as f:
                json.dump(json_safe_issues, f, indent=4)
            print(f"\nNaming issues saved to: {args.save_naming_issues}")
        except Exception as e:
            print(f"Error saving naming issues to JSON: {e}")
    
    # Summary
    if comparison_results['missing_extractions']:
        print(f"\n⚠️  WARNING: {len(comparison_results['missing_extractions'])} folders are missing from extraction!")
        print("You may need to re-run extract_calibre_comics.py for these folders.")
    else:
        print(f"\n✅ All source folders have corresponding extracted folders!")

if __name__ == "__main__":
    main() 