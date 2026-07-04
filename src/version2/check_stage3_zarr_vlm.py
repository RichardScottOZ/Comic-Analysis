"""Quick sanity check on a Stage 3 embedding Zarr store."""
import sys
import zarr
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "E:/stage3_embeddings_test.zarr"

z = zarr.open(path, mode='r')
emb = z['panel_embeddings'][:]
mask = z['panel_masks'][:]

print(f"=== {path} ===")
print(f"Embeddings shape : {emb.shape}")
print(f"Mask shape       : {mask.shape}")
print(f"Dtype            : {emb.dtype}")
print()
print(f"All zeros?  {np.all(emb == 0)}")
print(f"Any NaN?    {np.any(np.isnan(emb))}")
print(f"Any Inf?    {np.any(np.isinf(emb))}")
print(f"Min / Max   {emb.min():.4f} / {emb.max():.4f}")
print(f"Mean / Std  {emb.mean():.4f} / {emb.std():.4f}")
print()
print("Valid panels per page (first 10):")
for i in range(min(10, len(mask))):
    print(f"  Page {i:3d}: {int(mask[i].sum())} panels")
print()
print(f"Page 0 == Page 1 (should be False): {np.allclose(emb[0], emb[1])}")
print(f"Page 0 norm: {np.linalg.norm(emb[0,0]):.4f}")
print(f"Page 1 norm: {np.linalg.norm(emb[1,0]):.4f}")
print()

# Cosine similarity between page 0 panel 0 and page 1 panel 0
a = emb[0, 0]
b = emb[1, 0]
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
print(f"Cosine similarity page0[0] vs page1[0]: {cos_sim:.4f}")

# Padded panels should be all zeros
padded = []
for i in range(min(20, len(mask))):
    n_valid = int(mask[i].sum())
    if n_valid < emb.shape[1]:
        padded_emb = emb[i, n_valid:]
        padded.append(np.all(padded_emb == 0))
if padded:
    print(f"Padded panel slots are zero: {all(padded)} ({len(padded)} checked)")
else:
    print("No padded slots found in first 20 pages (all pages full)")
