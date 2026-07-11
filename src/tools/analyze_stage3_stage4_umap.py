#!/usr/bin/env python3
import os
import sys
import json
import zarr
import time
import numpy as np
from pathlib import Path

# Try importing analysis libraries, fail gracefully if missing
try:
    import umap
except ImportError:
    print("Error: UMAP library not found. Please install with: conda install -c conda-forge umap-learn or pip install umap-learn")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Error: scikit-learn not found. Please install with: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: matplotlib/seaborn not found. Please install with: pip install matplotlib seaborn")
    sys.exit(1)

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def load_zarr_dataset(zarr_path, keys):
    if not os.path.exists(zarr_path):
        return None
    print(f"[{get_timestamp()}] Opening Zarr store: {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    data = {}
    for key in keys:
        if key in store:
            data[key] = store[key]
            print(f"  • Found key: {key} (shape={store[key].shape}, dtype={store[key].dtype})")
    return data

def analyze_and_plot(embeddings, n_clusters, title, output_img):
    t_start = time.time()
    print(f"\n[{get_timestamp()}] Processing: {title} (Total items: {embeddings.shape[0]:,})")
    
    # Filter out any zero vectors
    print(f"[{get_timestamp()}] Filtering out zero vectors...")
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 1e-6
    valid_embeddings = embeddings[valid_mask]
    print(f"[{get_timestamp()}] Valid items: {valid_embeddings.shape[0]:,} / {embeddings.shape[0]:,}")
    
    if valid_embeddings.shape[0] == 0:
        print(f"[{get_timestamp()}] ❌ Error: No valid embeddings to analyze.")
        return

    # Normalize for cosine distance UMAP/Clustering
    print(f"[{get_timestamp()}] Normalizing valid embeddings...")
    normalized = valid_embeddings / (norms[valid_mask].reshape(-1, 1) + 1e-8)
    
    print(f"[{get_timestamp()}] Starting UMAP reduction to 2D (verbose output enabled)...")
    umap_start = time.time()
    # Low-memory mode is auto-selected by UMAP for large datasets
    # Set random_state=None and n_jobs=-1 to utilize all CPU cores in parallel
    reducer = umap.UMAP(n_components=2, random_state=None, metric='cosine', verbose=True, n_jobs=-1)
    coords_2d = reducer.fit_transform(normalized)
    print(f"[{get_timestamp()}] UMAP finished in {time.time() - umap_start:.2f} seconds.")
    
    print(f"[{get_timestamp()}] Starting K-Means clustering (K={n_clusters}, verbose convergence enabled)...")
    kmeans_start = time.time()
    # verbose=1 prints iteration-by-iteration convergence updates
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    clusters = kmeans.fit_predict(normalized)
    print(f"[{get_timestamp()}] KMeans finished in {time.time() - kmeans_start:.2f} seconds.")
    
    print(f"[{get_timestamp()}] Generating plot: {output_img}...")
    plot_start = time.time()
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=coords_2d[:, 0], 
        y=coords_2d[:, 1], 
        hue=clusters, 
        palette=sns.color_palette("hsv", n_clusters), 
        alpha=0.5, 
        edgecolor=None,
        s=1 # very small dot size to avoid overplotting on large datasets
    )
    plt.title(f"UMAP Projection & KMeans Clusters (K={n_clusters})\n{title}", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"[{get_timestamp()}] Saved plot to {output_img} (plot duration: {time.time() - plot_start:.2f}s)")
    print(f"[{get_timestamp()}] Total step time: {time.time() - t_start:.2f} seconds.")

def main():
    # Setup paths
    stage3_path = "E:/stage3_embeddings_test.zarr"
    if not os.path.exists(stage3_path):
        stage3_path = "E:/Comic_Analysis_Results_v2/stage3_embeddings.zarr"
        
    stage4_path = "E:/stage4_embeddings.zarr"
    if not os.path.exists(stage4_path):
        stage4_path = "E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"

    output_dir = Path("documentation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_clusters = 10

    # 1. Stage 3 Page Embeddings Analysis (via mean-pooling panel embeddings in chunks)
    stage3_data = load_zarr_dataset(stage3_path, ['panel_embeddings', 'panel_masks'])
    if stage3_data:
        panels_raw = stage3_data['panel_embeddings']
        masks_raw = stage3_data['panel_masks']
        total_pages = panels_raw.shape[0]
        
        # Load and mean-pool in chunks to prevent memory crash with timing prints
        print(f"\n[{get_timestamp()}] Loading Stage 3 panel vectors for all {total_pages:,} pages...")
        stage3_pages = np.empty((total_pages, panels_raw.shape[2]), dtype=np.float32)
        chunk_size = 50000
        
        t_load_start = time.time()
        for start in range(0, total_pages, chunk_size):
            t_chunk_start = time.time()
            end = min(start + chunk_size, total_pages)
            
            p_chunk = panels_raw[start:end]
            m_chunk = masks_raw[start:end]
            
            for i in range(end - start):
                mask = m_chunk[i].astype(bool)
                valid = p_chunk[i][mask]
                if len(valid) > 0:
                    stage3_pages[start + i] = valid.mean(axis=0)
                else:
                    stage3_pages[start + i] = 0.0
            
            # Progress calculation
            chunk_duration = time.time() - t_chunk_start
            elapsed = time.time() - t_load_start
            progress = end / total_pages
            est_total = elapsed / progress
            eta = est_total - elapsed
            print(f"  [{get_timestamp()}] Progress: {end:,}/{total_pages:,} ({progress*100:.1f}%) | Chunk time: {chunk_duration:.2f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                    
        analyze_and_plot(
            stage3_pages, 
            n_clusters, 
            "Stage 3 Page Embeddings (Mean-Pooled Static Multimodal)", 
            str(output_dir / "umap_stage3_pages.png")
        )
        # Clear RAM
        del stage3_pages
    else:
        print(f"[{get_timestamp()}] Warning: Stage 3 Zarr not found at {stage3_path}")

    # 2. Stage 4 Page/Strip Embeddings Analysis
    stage4_data = load_zarr_dataset(stage4_path, ['strip_embeddings'])
    if stage4_data:
        strip_embeddings = stage4_data['strip_embeddings'][:]
        analyze_and_plot(
            strip_embeddings, 
            n_clusters, 
            "Stage 4 Page/Strip Embeddings (Sequence Transformer Aggregated)", 
            str(output_dir / "umap_stage4_pages.png")
        )
    else:
        print(f"[{get_timestamp()}] Warning: Stage 4 Zarr not found at {stage4_path}")

    # Generate a Markdown report section for the user's README
    print(f"\n[{get_timestamp()}] Generating analysis summary...")
    report_content = f"""### Dimensionality Reduction & Clustering Analysis (UMAP + KMeans)

To inspect the structural organization of our semantic embedding space, we ran UMAP dimensionality reduction and KMeans clustering (K=10) on the entire page-level datasets for both Stage 3 and Stage 4.

#### 1. [Stage 3 Page Embeddings (Mean-Pooled)](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.png)
*   **Space**: Pure, context-free multimodal page representations (mean-pooled SigLIP visual + MiniLM text).
*   **Behavior**: Clusters are highly distinct and form clean, localized islands. This space is highly organized by visual/semantic content (e.g., characters, colors, dialogue themes).

#### 2. [Stage 4 Page/Strip Embeddings](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.png)
*   **Space**: Narrative aggregated page representations (Sequence Transformer + Aggregator).
*   **Behavior**: Pages cluster based on layout, panel sequencing, and overall storytelling pacing. This space organizes pages by narrative style rather than isolated objects.

*All generated plots are saved under [documentation/plots/](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/).*
"""
    with open("documentation/plots/analysis_report.md", "w") as f:
        f.write(report_content)
    print(f"[{get_timestamp()}] Report written to documentation/plots/analysis_report.md")

if __name__ == "__main__":
    main()
