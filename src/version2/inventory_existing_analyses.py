#!/usr/bin/env python3
"""
Inventory Existing Local Analyses
Scans a local directory for JSON analysis files and attempts to map them 
to canonical_ids in the master manifest.
"""

import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

def load_manifest_lookup(manifest_path):
    """
    Loads manifest and returns a dictionary for fast lookup.
    Key: Image Filename (stem or name), Value: Canonical ID
    """
    print(f"Loading manifest: {manifest_path}")
    lookup = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['canonical_id']
            # We map the final component (filename) to the CID
            # e.g. .../3 Guns - p031 -> 3 Guns - p031
            name = Path(cid).name 
            
            if name not in lookup:
                lookup[name] = []
            lookup[name].append(cid)
            
    print(f"Loaded {len(lookup)} unique filenames from manifest.")
    return lookup

def scan_and_match(analysis_root, manifest_lookup, output_csv, debug=False):
    """
    Scans local directory and matches against lookup.
    """
    print(f"Scanning directory: {analysis_root}")
    root_path = Path(analysis_root)
    
    matches = []
    ambiguous = 0
    no_match = 0
    
    # Create CSV writer
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical_id', 'local_analysis_path', 'match_type'])
        
        # Walk directory
        files = list(root_path.rglob('*.json'))
        print(f"Found {len(files)} JSON files. Matching...")
        
        for json_file in tqdm(files):
            stem = json_file.stem
            # Cleanup common suffixes
            clean_stem = stem
            if clean_stem.endswith('_analysis'):
                clean_stem = clean_stem[:-9]
            elif clean_stem.endswith('_caption'):
                clean_stem = clean_stem[:-8]
            
            # Strategy 1: Exact Match (clean_stem == manifest filename)
            candidates = manifest_lookup.get(clean_stem)
            match_type = 'exact'

            # Strategy 2: Prefix Match (Local is 'Folder_Filename', Manifest is 'Filename')
            # '3 Guns_3 Guns - p031' -> match '3 Guns - p031'
            if not candidates and '_' in clean_stem:
                # Try splitting by first underscore (Folder_File)
                parts = clean_stem.split('_', 1)
                if len(parts) > 1:
                    suffix = parts[1]
                    candidates = manifest_lookup.get(suffix)
                    if candidates: match_type = 'suffix_split'
            
            # Strategy 3: Path Reconstruction / Suffix Reassembly
            # The local filename is a flattened path like 'Folder_Sub_File'.
            # The manifest filename might be 'File' or 'Sub_File'.
            # We don't know where the 'path' ends and 'filename' begins because both use '_'.
            # So we split by '_' and try to match suffixes of increasing length against the manifest lookup.
            
            if not candidates and '_' in clean_stem:
                parts = clean_stem.split('_')
                # Try suffixes of length 1 to N
                # e.g. "A_B_C" -> check "C", then "B_C", then "A_B_C"
                
                for i in range(1, len(parts) + 1):
                    # Reassemble suffix: e.g. "2000AD_Regened_001"
                    potential_filename = "_".join(parts[-i:])
                    
                    lookup_matches = manifest_lookup.get(potential_filename)
                    if lookup_matches:
                        # We found a candidate list! 
                        # Now we must verify if the FULL path matches (reconstructed)
                        # Local: A_B_C -> Reconstructed: A/B/C (roughly)
                        
                        # We convert the *prefix* (the part we didn't use for filename) to a path
                        prefix_parts = parts[:-i]
                        prefix_path = "/".join(prefix_parts)
                        
                        # Check if any candidate ends with "prefix/filename"
                        # OR if prefix is empty, just "filename"
                        
                        best_match = None
                        
                        # Filter down candidates if we have a prefix
                        # Local: A_B_C_D_Filename
                        # Candidate: A/B/C/D/Filename
                        
                        if len(lookup_matches) == 1:
                            best_match = lookup_matches[0]
                        else:
                            # Disambiguation needed
                            # Check if our reconstructed prefix is a SUBSTRING of the candidate path
                            # This handles truncated local names: 'Abe Sapien The Devil Does Not J' in 'Abe Sapien The Devil Does Not Jest'
                            
                            # Reconstruct prefix from parts, replacing _ with space or nothing since truncation might cut words
                            # Better: Check if the *raw local stem* (minus filename) matches the start of the candidate
                            
                            local_prefix_raw = "_".join(prefix_parts) # e.g. "Abe_Sapien_The_..."
                            # Normalize local prefix for matching (replace _ with space? or just ignore non-alnum?)
                            # Let's try simple normalized substring.
                            
                            norm_local = local_prefix_raw.lower().replace('_', ' ').replace('-', ' ').strip()
                            
                            for cid in lookup_matches:
                                # Check if CID path (minus filename) roughly matches local prefix
                                cid_parent = str(Path(cid).parent).lower().replace('_', ' ').replace('-', ' ').strip()
                                
                                # Check containment both ways to handle truncations
                                if norm_local in cid_parent or cid_parent in norm_local:
                                    best_match = cid
                                    break
                                
                                # Fallback: Check checking raw prefix path as constructed
                                if prefix_path and prefix_path in cid:
                                    best_match = cid
                                    break

                        if best_match:
                            candidates = [best_match]
                            match_type = 'suffix_reassembly'
                            break # Stop once we find a valid match
                        
                        # If we found candidates by filename but the path didn't match, 
                        # we continue loop to try a longer filename suffix.

            if not candidates:
                if debug and no_match < 10:
                    print(f"DEBUG NO MATCH: Local '{clean_stem}' (Original: {stem}). Path: {json_file}")
                no_match += 1
                continue
                
            # If match found
            if len(candidates) == 1:
                writer.writerow([candidates[0], str(json_file), match_type])
                matches.append(candidates[0])
            else:
                # Ambiguous: Multiple canonical_ids share this filename
                # Try to disambiguate using parent folder or prefix
                
                # Check if the 'prefix' (from Strategy 2) matches the parent folder in canonical_id
                best_match = None
                
                if '_' in clean_stem:
                     prefix = clean_stem.split('_', 1)[0]
                     # Look for candidate whose parent folder contains this prefix
                     for c in candidates:
                         if prefix in c:
                             best_match = c
                             break
                
                # If still ambiguous, try parent directory of local file
                if not best_match:
                     parent_dir = json_file.parent.name
                     for c in candidates:
                         if parent_dir in c:
                             best_match = c
                             break

                if best_match:
                     writer.writerow([best_match, str(json_file), 'disambiguated'])
                     matches.append(best_match)
                else:
                    ambiguous += 1
                    if debug and ambiguous < 5:
                         print(f"DEBUG AMBIGUOUS: {clean_stem} -> {candidates}")

    print(f"\n--- Inventory Complete ---")
    print(f"Total JSONs Scanned: {len(files)}")
    print(f"Matched: {len(matches)}")
    print(f"Ambiguous (skipped): {ambiguous}")
    print(f"No Match: {no_match}")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inventory Local Analyses')
    parser.add_argument('--manifest', required=True, help='Path to master manifest CSV')
    parser.add_argument('--analysis-dir', required=True, help='Root directory of local JSON analyses')
    parser.add_argument('--output', default='recovered_analyses.csv', help='Output inventory CSV')
    parser.add_argument('--debug', action='store_true', help='Print debug info for mismatches')
    
    args = parser.parse_args()
    
    lookup = load_manifest_lookup(args.manifest)
    scan_and_match(args.analysis_dir, lookup, args.output, args.debug)