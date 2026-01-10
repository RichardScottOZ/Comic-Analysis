import json
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input PSS Labels JSON')
    parser.add_argument('--train-out', default='train_pss.json')
    parser.add_argument('--val-out', default='val_pss.json')
    parser.add_argument('--val-ratio', type=float, default=0.05)
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Filter for 'story' only (just to be safe)
    story_items = {k: v for k, v in data.items() if v in ['story', 'narrative']}
    keys = list(story_items.keys())
    
    print(f"Found {len(keys)} story pages.")
    
    # Shuffle and Split
    random.shuffle(keys)
    split_idx = int(len(keys) * (1 - args.val_ratio))
    
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    
    train_data = {k: story_items[k] for k in train_keys}
    val_data = {k: story_items[k] for k in val_keys}
    
    print(f"Training set: {len(train_data)}")
    print(f"Validation set: {len(val_data)}")
    
    with open(args.train_out, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
        
    with open(args.val_out, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
        
    print("Split complete.")

if __name__ == "__main__":
    main()
