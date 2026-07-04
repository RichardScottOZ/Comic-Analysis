"""
Split stage3_metadata_vlm.json into train/val subsets for Stage 4 training.

Both output files reference the SAME Zarr store — sequence_index values are
preserved so Stage4SequenceDataset can load the correct rows.

Usage:
    python split_stage4_metadata.py \
        --metadata stage3_metadata_vlm.json \
        --train_pss train_pss.json \
        --val_pss val_pss.json \
        --train_out stage4_train_metadata.json \
        --val_out stage4_val_metadata.json
"""

import argparse
import json
from pathlib import Path


def main(args):
    print(f"Loading metadata: {args.metadata}")
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"  {len(metadata):,} total entries")

    print(f"Loading train PSS labels: {args.train_pss}")
    with open(args.train_pss, 'r', encoding='utf-8') as f:
        train_ids = set(json.load(f).keys())
    print(f"  {len(train_ids):,} train canonical_ids")

    print(f"Loading val PSS labels: {args.val_pss}")
    with open(args.val_pss, 'r', encoding='utf-8') as f:
        val_ids = set(json.load(f).keys())
    print(f"  {len(val_ids):,} val canonical_ids")

    train_meta, val_meta, neither = [], [], []
    for entry in metadata:
        cid = entry['canonical_id']
        if cid in val_ids:
            val_meta.append(entry)
        elif cid in train_ids:
            train_meta.append(entry)
        else:
            neither.append(entry)

    print(f"\n--- Split Summary ---")
    print(f"  Train : {len(train_meta):,}")
    print(f"  Val   : {len(val_meta):,}")
    print(f"  Neither (not in either PSS file): {len(neither):,}")

    print(f"\nSaving train metadata: {args.train_out}")
    with open(args.train_out, 'w', encoding='utf-8') as f:
        json.dump(train_meta, f)

    print(f"Saving val metadata:   {args.val_out}")
    with open(args.val_out, 'w', encoding='utf-8') as f:
        json.dump(val_meta, f)

    print("\n✅ Done. Both files reference the same Zarr — pass identical")
    print("   --train_embeddings and --val_embeddings paths to train_stage4.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Stage 3 metadata into train/val for Stage 4")
    parser.add_argument('--metadata',   type=str, default='stage3_metadata_vlm.json')
    parser.add_argument('--train_pss',  type=str, default='train_pss.json')
    parser.add_argument('--val_pss',    type=str, default='val_pss.json')
    parser.add_argument('--train_out',  type=str, default='stage4_train_metadata.json')
    parser.add_argument('--val_out',    type=str, default='stage4_val_metadata.json')
    args = parser.parse_args()
    main(args)
