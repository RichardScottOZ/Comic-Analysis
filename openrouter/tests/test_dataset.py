#!/usr/bin/env python3

import sys
import os
sys.path.append('benchmarks/detections/openrouter')

from closure_lite_dataset import create_dataloader

def test_dataset():
    json_dir = "E:/amazon_datacontract-test"
    image_root = "E:/amazon"
    
    print(f"Testing dataset with:")
    print(f"  JSON dir: {json_dir}")
    print(f"  Image root: {image_root}")
    
    try:
        dataloader = create_dataloader(json_dir, image_root, batch_size=1, num_workers=0)
        print(f"Dataset created successfully!")
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        # Test loading one batch
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Panel mask: {batch['panel_mask'].sum(dim=1)}")
            if i >= 0:  # Just test first batch
                break
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
