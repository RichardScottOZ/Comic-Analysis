#!/usr/bin/env python3

import subprocess
import sys

def restart_training():
    """Restart training with conservative settings"""
    
    # Use the best checkpoint from epoch 1
    checkpoint_path = "./closure_lite_output/best_checkpoint.pth"
    
    # Much more conservative settings
    cmd = [
        "python", "benchmarks/detections/openrouter/train_closure_lite.py",
        "--json_dir", "E:/amazon_datacontract-test",
        "--image_root", "E:/amazon", 
        "--output_dir", "./closure_lite_output",
        "--batch_size", "2",  # Smaller batch size
        "--epochs", "5",
        "--lr", "1e-4",  # Much lower learning rate
        # "--max_samples", "10000",  # Uncomment to sample only 10K pages for faster training
        "--resume", checkpoint_path
    ]
    
    print("Restarting training with conservative settings:")
    print("  - Learning rate: 1e-4 (much lower)")
    print("  - Batch size: 2 (smaller)")
    print("  - Resuming from best checkpoint")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    restart_training()
