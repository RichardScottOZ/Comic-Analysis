#!/usr/bin/env python3
import zarr
import numpy as np
import os
import sys

def main():
    zarr_path = "E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"
    if not os.path.exists(zarr_path):
        zarr_path = "E:/stage4_embeddings.zarr"
        
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr path not found at {zarr_path}")
        return

    print(f"Loading Zarr store at: {zarr_path}")
    try:
        store = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"Failed to open Zarr: {e}")
        return

    keys = list(store.keys())
    print(f"Keys in Zarr store: {keys}")

    for key in keys:
        arr = store[key]
        print(f"\n--- Key: {key} ---")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        
        # Read a sample chunk to inspect
        sample_size = min(100, arr.shape[0])
        sample = arr[:sample_size]
        
        print(f"Sample range: min={np.min(sample)}, max={np.max(sample)}")
        print(f"Mean: {np.mean(sample):.6f}")
        print(f"Std:  {np.std(sample):.6f}")
        print(f"Any NaNs? {np.isnan(sample).any()}")
        print(f"Any Infs? {np.isinf(sample).any()}")
        
        # Check if they are all zeros
        num_zeros = np.sum(sample == 0)
        total_elements = sample.size
        print(f"Zero elements: {num_zeros} / {total_elements} ({num_zeros/total_elements*100:.2f}%)")

if __name__ == "__main__":
    main()
