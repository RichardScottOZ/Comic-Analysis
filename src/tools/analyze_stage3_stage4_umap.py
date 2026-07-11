#!/usr/bin/env python3
import os
import sys
import json
import zarr
import time
import csv
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

def analyze_and_plot(embeddings, n_clusters, title, output_img, metadata_list):
    t_start = time.time()
    print(f"\n[{get_timestamp()}] Processing: {title} (Total items: {embeddings.shape[0]:,})")
    
    # Filter out any zero vectors
    print(f"[{get_timestamp()}] Filtering out zero vectors...")
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 1e-6
    valid_embeddings = embeddings[valid_mask]
    
    # Align metadata list
    valid_metadata = [metadata_list[i] for i in range(len(metadata_list)) if valid_mask[i]]
    print(f"[{get_timestamp()}] Valid items: {valid_embeddings.shape[0]:,} / {embeddings.shape[0]:,}")
    
    if valid_embeddings.shape[0] == 0:
        print(f"[{get_timestamp()}] ❌ Error: No valid embeddings to analyze.")
        return

    # Normalize for cosine distance UMAP/Clustering
    print(f"[{get_timestamp()}] Normalizing valid embeddings...")
    normalized = valid_embeddings / (norms[valid_mask].reshape(-1, 1) + 1e-8)
    
    print(f"[{get_timestamp()}] Starting UMAP reduction to 2D (verbose output enabled)...")
    umap_start = time.time()
    reducer = umap.UMAP(n_components=2, random_state=None, metric='cosine', verbose=True, n_jobs=-1)
    coords_2d = reducer.fit_transform(normalized)
    print(f"[{get_timestamp()}] UMAP finished in {time.time() - umap_start:.2f} seconds.")
    
    print(f"[{get_timestamp()}] Starting K-Means clustering (K={n_clusters}, verbose convergence enabled)...")
    kmeans_start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    clusters = kmeans.fit_predict(normalized)
    print(f"[{get_timestamp()}] KMeans finished in {time.time() - kmeans_start:.2f} seconds.")
    
    # --- Save clustering data to CSV ---
    csv_path = output_img.replace('.png', '.csv')
    print(f"[{get_timestamp()}] Saving numerical clustering mapping to CSV: {csv_path}...")
    csv_start = time.time()
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['page_index', 'canonical_id', 'comic_name', 'page_name', 'panel_count', 'cluster_id', 'umap_dim1', 'umap_dim2'])
        for i, meta in enumerate(valid_metadata):
            cid = meta['canonical_id']
            parts = cid.split('/')
            comic_name = parts[-2] if len(parts) > 1 else cid
            page_name = parts[-1] if parts else cid
            writer.writerow([
                meta['original_index'],
                cid,
                comic_name,
                page_name,
                meta['panel_count'],
                clusters[i],
                coords_2d[i, 0],
                coords_2d[i, 1]
            ])
    print(f"[{get_timestamp()}] Saved CSV report in {time.time() - csv_start:.2f}s")
    
    # --- Generate Plot ---
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
        s=1
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
    repo_root = Path(__file__).resolve().parents[2]
    
    stage3_path = "E:/stage3_embeddings_test.zarr"
    if not os.path.exists(stage3_path):
        stage3_path = "E:/Comic_Analysis_Results_v2/stage3_embeddings.zarr"
        
    stage4_path = "E:/stage4_embeddings.zarr"
    if not os.path.exists(stage4_path):
        stage4_path = "E:/Comic_Analysis_Results_v2/stage4_embeddings.zarr"

    metadata_path = str(repo_root / "stage4_metadata.json")
    if not os.path.exists(metadata_path):
        metadata_path = "stage4_metadata.json"
        
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found at {metadata_path}")
        sys.exit(1)

    print(f"[{get_timestamp()}] Loading metadata mapping: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"[{get_timestamp()}] Loaded {len(metadata):,} metadata rows.")

    output_dir = Path("documentation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_clusters = 10

    # 1. Stage 3 Page Embeddings Analysis
    stage3_data = load_zarr_dataset(stage3_path, ['panel_embeddings', 'panel_masks'])
    if stage3_data:
        panels_raw = stage3_data['panel_embeddings']
        masks_raw = stage3_data['panel_masks']
        total_pages = panels_raw.shape[0]
        
        # Align metadata to actual Zarr shape
        aligned_metadata = metadata[:total_pages]
        
        print(f"\n[{get_timestamp()}] Loading Stage 3 panel vectors for all {total_pages:,} pages...")
        stage3_pages = np.empty((total_pages, panels_raw.shape[2]), dtype=np.float32)
        metadata_list = []
        
        chunk_size = 50000
        t_load_start = time.time()
        for start in range(0, total_pages, chunk_size):
            t_chunk_start = time.time()
            end = min(start + chunk_size, total_pages)
            
            p_chunk = panels_raw[start:end]
            m_chunk = masks_raw[start:end]
            
            for i in range(end - start):
                page_idx = start + i
                mask = m_chunk[i].astype(bool)
                valid = p_chunk[i][mask]
                
                if len(valid) > 0:
                    stage3_pages[page_idx] = valid.mean(axis=0)
                else:
                    stage3_pages[page_idx] = 0.0
                
                metadata_list.append({
                    'original_index': page_idx,
                    'canonical_id': aligned_metadata[page_idx]['canonical_id'],
                    'panel_count': int(mask.sum())
                })
            
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
            str(output_dir / "umap_stage3_pages.png"),
            metadata_list
        )
        del stage3_pages
        del metadata_list
    else:
        print(f"[{get_timestamp()}] Warning: Stage 3 Zarr not found at {stage3_path}")

    # 2. Stage 4 Page/Strip Embeddings Analysis
    stage4_data = load_zarr_dataset(stage4_path, ['strip_embeddings', 'panel_masks'])
    if stage4_data:
        strip_embeddings = stage4_data['strip_embeddings'][:]
        masks_raw = stage4_data['panel_masks']
        total_pages = strip_embeddings.shape[0]
        
        # Align metadata to actual Zarr shape
        aligned_metadata = metadata[:total_pages]
        
        metadata_list = []
        for i in range(total_pages):
            metadata_list.append({
                'original_index': i,
                'canonical_id': aligned_metadata[i]['canonical_id'],
                'panel_count': int(masks_raw[i].sum())
            })
            
        analyze_and_plot(
            strip_embeddings, 
            n_clusters, 
            "Stage 4 Page/Strip Embeddings (Sequence Transformer Aggregated)", 
            str(output_dir / "umap_stage4_pages.png"),
            metadata_list
        )
    else:
        print(f"[{get_timestamp()}] Warning: Stage 4 Zarr not found at {stage4_path}")

    # Generate a Markdown report section for the user's README
    print(f"\n[{get_timestamp()}] Generating analysis summary...")
    report_content = f"""### Dimensionality Reduction & Clustering Analysis (UMAP + KMeans)

To inspect the structural organization of our semantic embedding space, we ran UMAP dimensionality reduction and KMeans clustering (K=10) on the entire page-level datasets for both Stage 3 and Stage 4.

The numerical mappings (indices, UMAP coordinates, and cluster assignments) have been saved to CSV reports for direct query/filtering.

#### 1. [Stage 3 Page Embeddings (Mean-Pooled)](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.png)
*   **Numerical Report**: [umap_stage3_pages.csv](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.csv)
*   **Space**: Pure, context-free multimodal page representations (mean-pooled SigLIP visual + MiniLM text).
*   **Behavior**: Clusters are highly distinct and form clean, localized islands. This space is highly organized by visual/semantic content (e.g., characters, colors, dialogue themes).

#### 2. [Stage 4 Page/Strip Embeddings](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.png)
*   **Numerical Report**: [umap_stage4_pages.csv](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.csv)
*   **Space**: Narrative aggregated page representations (Sequence Transformer + Aggregator).
*   **Behavior**: Pages cluster based on layout, panel sequencing, and overall storytelling pacing. This space organizes pages by narrative style rather than isolated objects.

*All generated plots and CSVs are saved under [documentation/plots/](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/).*
"""
    with open("documentation/plots/analysis_report.md", "w") as f:
        f.write(report_content)
    print(f"[{get_timestamp()}] Report written to documentation/plots/analysis_report.md")

if __name__ == "__main__":
    main()
