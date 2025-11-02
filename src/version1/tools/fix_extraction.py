#!/usr/bin/env python3
"""
Fix extraction by removing problematic folders and re-extracting renamed versions.
This script handles the workflow after check_extract.py identifies overwrite problems.
"""

import argparse
import shutil
from pathlib import Path
import subprocess
import sys
import os

def identify_problematic_extracted_folders(extracted_dir, naming_issues):
    """Identify which folders in extracted_dir need to be removed."""
    print(f"Identifying problematic folders in: {extracted_dir}")
    
    # Get the CBZ folder names that would cause overwrites
    problematic_cbz_names = set()
    for issue in naming_issues:
        problematic_cbz_names.add(issue['comic_name'])
    
    print(f"Looking for {len(problematic_cbz_names)} problematic folder names:")
    for name in problematic_cbz_names:
        print(f"  - {name}")
    
    # Find matching folders in extracted directory
    extracted_path = Path(extracted_dir)
    folders_to_remove = []
    
    if extracted_path.exists():
        for item in extracted_path.iterdir():
            if item.is_dir() and item.name in problematic_cbz_names:
                folders_to_remove.append(item)
                print(f"  Found problematic folder: {item.name}")
    
    return folders_to_remove

def remove_problematic_folders(extracted_dir, folders_to_remove, dry_run=False):
    """Remove the problematic folders from extracted directory."""
    if dry_run:
        print(f"\nDRY RUN - Would remove {len(folders_to_remove)} problematic folders from: {extracted_dir}")
        print("Folders that would be removed:")
        for folder in folders_to_remove:
            print(f"  - {folder.name}")
        print(f"Total: {len(folders_to_remove)} folders would be removed")
        return len(folders_to_remove)
    else:
        print(f"\nRemoving {len(folders_to_remove)} problematic folders from: {extracted_dir}")
        
        removed_count = 0
        for folder in folders_to_remove:
            try:
                print(f"  Removing: {folder.name}")
                shutil.rmtree(folder)
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {folder.name}: {e}")
        
        print(f"Successfully removed {removed_count} folders")
        return removed_count

def extract_renamed_folders(rename_dir, output_dir, max_workers=None, dry_run=False):
    """Extract the renamed folders using extract_calibre_comics.py."""
    if dry_run:
        print(f"\nDRY RUN - Would extract renamed folders from: {rename_dir}")
        print(f"Output directory would be: {output_dir}")
        
        # Count files that would be extracted
        rename_path = Path(rename_dir)
        if rename_path.exists():
            cbz_files = list(rename_path.rglob("*.cbz")) + list(rename_path.rglob("*.cbr"))
            print(f"Would extract {len(cbz_files)} comic files")
            print("Sample files that would be extracted:")
            for file in cbz_files[:5]:
                print(f"  - {file.name}")
            if len(cbz_files) > 5:
                print(f"  ... and {len(cbz_files) - 5} more files")
        else:
            print(f"Warning: Rename directory does not exist: {rename_dir}")
        
        return True
    else:
        print(f"\nExtracting renamed folders from: {rename_dir}")
        print(f"Output directory: {output_dir}")
        
        # Build the command
        cmd = [
            sys.executable,
            "benchmarks/detections/openrouter/extract_calibre_comics.py",
            "--input-dir", str(rename_dir),
            "--output-dir", str(output_dir),
            "--preserve-structure",
            "--skip-existing"
        ]
        
        if max_workers:
            cmd.extend(["--max-workers", str(max_workers)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Extraction completed successfully!")
            print("Output:")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed with error code {e.returncode}")
            print("Error output:")
            print(e.stderr)
            return False

def create_merge_script(original_extracted_dir, renamed_extracted_dir, output_script_path):
    """Create a script to merge the two extracted directories."""
    print(f"\nCreating merge script: {output_script_path}")
    
    script_content = f'''#!/usr/bin/env python3
"""
Merge script to combine original and renamed extracted directories.
Run this after both extractions are complete.
"""

import shutil
from pathlib import Path
import argparse

def merge_extracted_directories(original_dir, renamed_dir, output_dir):
    """Merge the two extracted directories."""
    print(f"Merging extracted directories...")
    print(f"  Original: {{original_dir}}")
    print(f"  Renamed: {{renamed_dir}}")
    print(f"  Output: {{output_dir}}")
    
    original_path = Path(original_dir)
    renamed_path = Path(renamed_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from original directory
    print("Copying files from original directory...")
    copied_count = 0
    for source_file in original_path.rglob("*"):
        if source_file.is_file():
            relative_path = source_file.relative_to(original_path)
            target_file = output_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied_count += 1
    
    print(f"Copied {{copied_count}} files from original directory")
    
    # Copy all files from renamed directory (will overwrite any duplicates)
    print("Copying files from renamed directory...")
    renamed_count = 0
    for source_file in renamed_path.rglob("*"):
        if source_file.is_file():
            relative_path = source_file.relative_to(renamed_path)
            target_file = output_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            renamed_count += 1
    
    print(f"Copied {{renamed_count}} files from renamed directory")
    print(f"Merge completed! Total files in output: {{copied_count + renamed_count}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge extracted directories')
    parser.add_argument('--original-dir', type=str, required=True,
                       help='Original extracted directory')
    parser.add_argument('--renamed-dir', type=str, required=True,
                       help='Renamed extracted directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for merged results')
    
    args = parser.parse_args()
    merge_extracted_directories(args.original_dir, args.renamed_dir, args.output_dir)
'''
    
    with open(output_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_script_path, 0o755)
    
    print(f"Merge script created: {output_script_path}")
    print("To use it later, run:")
    print(f"  python {output_script_path} --original-dir \"{original_extracted_dir}\" --renamed-dir \"{renamed_extracted_dir}\" --output-dir \"[your_merged_output_dir]\"")

def main():
    parser = argparse.ArgumentParser(description='Fix extraction by removing problematic folders and re-extracting')
    parser.add_argument('--original-extracted-dir', type=str, required=True,
                       help='Original extracted directory with overwrite problems')
    parser.add_argument('--rename-dir', type=str, required=True,
                       help='Directory with renamed problematic folders')
    parser.add_argument('--renamed-extracted-dir', type=str, required=True,
                       help='Output directory for extracted renamed folders')
    parser.add_argument('--naming-issues-file', type=str, required=True,
                       help='JSON file with naming issues from check_extract.py')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum workers for extraction')
    parser.add_argument('--skip-removal', action='store_true',
                       help='Skip removing problematic folders (for testing)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip extracting renamed folders (use when already extracted by check_extract.py)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it (for verification)')
    
    args = parser.parse_args()
    
    # Load naming issues
    import json
    try:
        with open(args.naming_issues_file, 'r') as f:
            json_data = json.load(f)
        
        # Convert string paths back to Path objects
        naming_issues = []
        for issue in json_data:
            naming_issue = {
                'folder_name': issue['folder_name'],
                'clean_folder_name': issue['clean_folder_name'],
                'comic_name': issue['comic_name'],
                'comic_file': Path(issue['comic_file']),  # Convert string back to Path
                'relative_path': issue['relative_path'],
                'overwrite_count': issue['overwrite_count']
            }
            naming_issues.append(naming_issue)
        
        print(f"Loaded {len(naming_issues)} naming issues from: {args.naming_issues_file}")
    except Exception as e:
        print(f"Error loading naming issues: {e}")
        return
    
    # Step 1: Identify and remove problematic folders
    if not args.skip_removal:
        folders_to_remove = identify_problematic_extracted_folders(args.original_extracted_dir, naming_issues)
        if folders_to_remove:
            remove_problematic_folders(args.original_extracted_dir, folders_to_remove, args.dry_run)
        else:
            print("No problematic folders found to remove")
    else:
        print("Skipping folder removal (--skip-removal)")
    
    # Step 2: Extract renamed folders
    if not args.skip_extraction:
        success = extract_renamed_folders(args.rename_dir, args.renamed_extracted_dir, args.max_workers, args.dry_run)
        if not success:
            print("Extraction failed!")
            return
    else:
        print("Skipping extraction (--skip-extraction)")
    
    # Step 3: Create merge script
    merge_script_path = Path(args.renamed_extracted_dir).parent / "merge_extracted.py"
    create_merge_script(args.original_extracted_dir, args.renamed_extracted_dir, merge_script_path)
    
    print(f"\n✅ Fix extraction completed!")
    print(f"  Original extracted directory (cleaned): {args.original_extracted_dir}")
    print(f"  Renamed extracted directory: {args.renamed_extracted_dir}")
    print(f"  Merge script: {merge_script_path}")
    print(f"\nNext steps:")
    print(f"  1. Verify both extractions are complete")
    print(f"  2. Run the merge script to combine them")

if __name__ == "__main__":
    main() 