#!/usr/bin/env python3
"""
Verify Stage 4 Embeddings
Loads the generated Stage 4 Zarr to verify data integrity, shapes, 
and perform strip-level similarity sanity checks.
"""

import zarr
import json
import numpy as np
import argparse
import random
import os
from pathlib import Path
from collections import defaultdict

def cosine_similarity(a, b):
    # a: (D,), b: (D,)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', type=str, default="E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr")
    parser.add_argument('--metadata', type=str, default="E:/Comic_Analysis_Results_v2/stage4_metadata.json")
    args = parser.parse_args()

    if not os.path.exists(args.zarr):
        print(f"Error: Zarr store not found at {args.zarr}")
        return
        
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata JSON not found at {args.metadata}")
        return

    print(f"Loading Zarr: {args.zarr}")
    try:
        store = zarr.open(args.zarr, mode='r')
        context_embs = store['contextualized_panels']
        strip_embs = store['strip_embeddings']
        masks = store['panel_masks']
        print(f"✅ Contextualized Panels: {context_embs.shape}, dtype: {context_embs.dtype}")
        print(f"✅ Strip Embeddings:      {strip_embs.shape}, dtype: {strip_embs.dtype}")
        print(f"✅ Masks shape:           {masks.shape}, dtype: {masks.dtype}")
    except Exception as e:
        print(f"❌ Failed to load Zarr: {e}")
        return

    print(f"\nLoading Metadata: {args.metadata}")
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✅ Metadata entries: {len(metadata)}")

    if len(metadata) != strip_embs.shape[0]:
        print(f"⚠️ WARNING: Length mismatch! Metadata ({len(metadata)}) != Zarr ({strip_embs.shape[0]})")
    else:
        print("✅ Lengths match perfectly.")

    # Group by Book
    books = defaultdict(list)
    for i, meta in enumerate(metadata):
        # Extract Book ID (parent folder of canonical_id)
        cid = meta.get('canonical_id', '')
        book_id = str(Path(cid).parent)
        books[book_id].append(i)

    print(f"Found {len(books)} unique books/folders.")

    print("\n--- Basic Statistics (First 1000 pages) ---")
    sample_size = min(1000, strip_embs.shape[0])
    sample_strip = strip_embs[:sample_size]
    
    print(f"Strip Embeddings (Page Level):")
    print(f"  Mean: {np.mean(sample_strip):.4f}")
    print(f"  Std:  {np.std(sample_strip):.4f}")
    
    if np.isnan(sample_strip).any():
        print("❌ WARNING: NaNs detected in Strip Embeddings!")
    else:
        print("✅ No NaNs detected.")

    print("\n--- Qualitative Strip Similarity Check ---")
    # Find a book with at least 2 pages
    valid_books = {k: v for k, v in books.items() if len(v) >= 2}
    if not valid_books:
        print("Could not find any books with > 1 page.")
        return

    book_a_id = random.choice(list(valid_books.keys()))
    book_a_pages = valid_books[book_a_id]
    
    idx_a1, idx_a2 = random.sample(book_a_pages, 2)
    
    book_b_id = random.choice(list(valid_books.keys()))
    while book_b_id == book_a_id:
        book_b_id = random.choice(list(valid_books.keys()))
    
    idx_b1 = random.choice(valid_books[book_b_id])

    print(f"Book A: {book_a_id}")
    print(f"  Page 1: {metadata[idx_a1]['canonical_id']}")
    print(f"  Page 2: {metadata[idx_a2]['canonical_id']}")
    print(f"Book B: {book_b_id}")
    print(f"  Page 1: {metadata[idx_b1]['canonical_id']}")

    emb_a1 = strip_embs[idx_a1]
    emb_a2 = strip_embs[idx_a2]
    emb_b1 = strip_embs[idx_b1]

    sim_internal = cosine_similarity(emb_a1, emb_a2)
    sim_cross = cosine_similarity(emb_a1, emb_b1)

    print(f"\nInternal Similarity (Book A Page 1 vs Book A Page 2): {sim_internal:.4f}")
    print(f"Cross Similarity  (Book A Page 1 vs Book B Page 1): {sim_cross:.4f}")

    if sim_internal > sim_cross:
        print("\n✅ SUCCESS: The Strip Embedding correctly identifies pages from the same book!")
    else:
        print("\n⚠️ WARNING: Cross-book similarity is higher. The Strip Embedding might not capture global book style well.")

if __name__ == "__main__":
    main()
