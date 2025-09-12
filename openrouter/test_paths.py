#!/usr/bin/env python3

import os
from pathlib import Path

def test_paths():
    # Test Windows to WSL path conversion
    json_dir = "E:/amazon_datacontract-test"
    image_root = "E:/amazon"
    
    print(f"Original paths:")
    print(f"  JSON dir: {json_dir}")
    print(f"  Image root: {image_root}")
    
    # Convert to WSL paths
    if json_dir.startswith('E:/') or json_dir.startswith('E:\\'):
        json_dir = "/mnt/e" + json_dir[2:].replace('\\', '/')
    if image_root.startswith('E:/') or image_root.startswith('E:\\'):
        image_root = "/mnt/e" + image_root[2:].replace('\\', '/')
    
    print(f"\nConverted paths:")
    print(f"  JSON dir: {json_dir}")
    print(f"  Image root: {image_root}")
    
    # Check if directories exist
    print(f"\nDirectory checks:")
    print(f"  JSON dir exists: {os.path.exists(json_dir)}")
    print(f"  Image root exists: {os.path.exists(image_root)}")
    
    # List JSON files
    if os.path.exists(json_dir):
        json_files = list(Path(json_dir).glob("*.json"))
        print(f"  Found {len(json_files)} JSON files")
        if json_files:
            print(f"  First few files:")
            for f in json_files[:3]:
                print(f"    {f}")
    else:
        print(f"  JSON directory does not exist!")

if __name__ == "__main__":
    test_paths()
