#!/usr/bin/env python3
"""
Recursive Zarr Search Test
Reads a `search_results.json` from a previous search, and uses the exact 
canonical_id and panel_idx to perform a 100% accurate recursive query 
directly against the Zarr embeddings.
"""

import os
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default="local_test_output/search_results/vampire_bite/search_results.json", help="Path to the search_results.json from a previous query")
    parser.add_argument('--output_root', default="local_test_output/recursive_search")
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: Could not find {args.input_json}")
        return

    with open(args.input_json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print("No results found in the JSON file.")
        return

    print(f"Found {len(results)} exact panels to use as recursive queries.")

    for item in results:
        rank = item.get("rank")
        cid = item.get("canonical_id")
        p_idx = item.get("panel_idx")
        
        if not cid or p_idx is None:
            print(f"Skipping Rank {rank} due to missing metadata.")
            continue
            
        page_name = Path(cid).stem.replace(' ', '_')
        folder_name = f"r{rank}_{page_name}_p{p_idx}"
        
        print(f"\n>>> Running exact Zarr recursive search for: {folder_name}")
        
        # Call visualize_search_results.py with exact coordinates
        cmd = (
            f"python src/tools/visualize_search_results.py "
            f"--query_canonical_id \"{cid}\" "
            f"--query_panel_idx {p_idx} "
            f"--output_folder_name \"{folder_name}\" "
            f"--output_dir \"{args.output_root}\" "
            f"--top_k {args.top_k}"
        )
        
        os.system(cmd)

    print(f"\nDone. Results saved to {args.output_root}")

if __name__ == "__main__":
    main()
