"""
Test the new efficient dataset loading
"""

import time
import torch
from closure_lite_dataset import create_dataloader

def test_efficient_loading():
    """Test the new efficient loading with sampling"""
    
    print("🧪 Testing Efficient Dataset Loading")
    print("=" * 50)
    
    # Test 1: Small sample
    print("\n1️⃣ Testing with 100 samples...")
    start_time = time.time()
    
    dataloader = create_dataloader(
        "E:/amazon_datacontract-test",
        "E:/amazon",
        batch_size=2,
        max_panels=8,
        max_samples=100,
        num_workers=0
    )
    
    load_time = time.time() - start_time
    print(f"   ⏱️  Load time: {load_time:.2f} seconds")
    print(f"   📊 Dataset size: {len(dataloader.dataset)} pages")
    print(f"   🔄 Batches: {len(dataloader)}")
    
    # Test 2: Load a few batches
    print("\n2️⃣ Testing batch loading...")
    start_time = time.time()
    
    batch_count = 0
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Test first 3 batches
            break
        batch_count += 1
        print(f"   Batch {i+1}: images={batch['images'].shape}, panels={batch['panel_mask'].sum().item()}")
    
    batch_time = time.time() - start_time
    print(f"   ⏱️  Batch loading time: {batch_time:.2f} seconds")
    print(f"   📦 Loaded {batch_count} batches")
    
    # Test 3: Memory usage
    print("\n3️⃣ Memory usage check...")
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"   💾 Memory usage: {memory_mb:.1f} MB")
    
    # Test 4: Compare with different sample sizes
    print("\n4️⃣ Testing different sample sizes...")
    
    for samples in [1000, 5000, 10000]:
        print(f"\n   Testing with {samples} samples...")
        start_time = time.time()
        
        dataloader = create_dataloader(
            "E:/amazon_datacontract-test",
            "E:/amazon",
            batch_size=2,
            max_panels=8,
            max_samples=samples,
            num_workers=0
        )
        
        load_time = time.time() - start_time
        print(f"      ⏱️  Load time: {load_time:.2f} seconds")
        print(f"      📊 Dataset size: {len(dataloader.dataset)} pages")
    
    print("\n✅ Efficient loading test completed!")
    print("\nKey improvements:")
    print("  - No more loading 800K files into memory")
    print("  - On-demand loading of individual pages")
    print("  - Random sampling before loading")
    print("  - Much faster startup time")

if __name__ == "__main__":
    test_efficient_loading()
