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

try:
    import pandas as pd
    import pyarrow
except ImportError:
    print("Error: pandas and pyarrow are required to write Parquet files. Please install with: pip install pandas pyarrow")
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

def analyze_and_plot(embeddings, n_clusters, title, output_img, metadata_list, zarr_out_path, is_panel=False):
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
    # Set init='random' to prevent spectral layout pairwise distance crashes on disconnected components
    reducer = umap.UMAP(n_components=2, init='random', random_state=None, metric='cosine', verbose=True, n_jobs=-1, low_memory=True)
    coords_2d = reducer.fit_transform(normalized)
    print(f"[{get_timestamp()}] UMAP finished in {time.time() - umap_start:.2f} seconds.")
    
    print(f"[{get_timestamp()}] Starting K-Means clustering (K={n_clusters}, verbose convergence enabled)...")
    kmeans_start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    clusters = kmeans.fit_predict(normalized)
    print(f"[{get_timestamp()}] KMeans finished in {time.time() - kmeans_start:.2f} seconds.")
    
    # --- Save to analysis Zarr dataset ---
    prefix = "panel" if is_panel else "page"
    print(f"[{get_timestamp()}] Writing {prefix} clusters and UMAP to Zarr: {zarr_out_path}...")
    zarr_start = time.time()
    z_store = zarr.open(zarr_out_path, mode='a')
    z_store[f"{prefix}_umap"] = coords_2d
    z_store[f"{prefix}_clusters"] = clusters
    print(f"[{get_timestamp()}] Saved Zarr arrays in {time.time() - zarr_start:.2f}s")
    
    # Assemble pandas dataframe for high-performance file output
    print(f"[{get_timestamp()}] Assembling pandas dataframe...")
    df_start = time.time()
    data = []
    for i, meta in enumerate(valid_metadata):
        cid = meta['canonical_id']
        parts = cid.split('/')
        comic_name = parts[-2] if len(parts) > 1 else cid
        page_name = parts[-1] if parts else cid
        
        row = {
            'page_index': meta['original_index'],
            'canonical_id': cid,
            'comic_name': comic_name,
            'page_name': page_name,
            'panel_count': meta['panel_count'],
            'cluster_id': clusters[i],
            'umap_dim1': coords_2d[i, 0],
            'umap_dim2': coords_2d[i, 1]
        }
        if is_panel:
            row['panel_index'] = meta['panel_index']
        data.append(row)
        
    df = pd.DataFrame(data)
    if is_panel:
        cols = ['page_index', 'panel_index', 'canonical_id', 'comic_name', 'page_name', 'panel_count', 'cluster_id', 'umap_dim1', 'umap_dim2']
    else:
        cols = ['page_index', 'canonical_id', 'comic_name', 'page_name', 'panel_count', 'cluster_id', 'umap_dim1', 'umap_dim2']
    df = df[cols]
    
    # --- Save to Parquet ---
    parquet_path = output_img.replace('.png', '.parquet')
    print(f"[{get_timestamp()}] Saving numerical clustering mapping to Parquet: {parquet_path}...")
    parquet_start = time.time()
    df.to_parquet(parquet_path, index=False, compression='snappy')
    print(f"[{get_timestamp()}] Saved Parquet file in {time.time() - parquet_start:.2f}s")
    
    # --- Save to CSV (Pages only - skip panels to save storage and disk write time) ---
    if not is_panel:
        csv_path = output_img.replace('.png', '.csv')
        print(f"[{get_timestamp()}] Saving numerical clustering mapping to CSV: {csv_path}...")
        csv_start = time.time()
        df.to_csv(csv_path, index=False)
        print(f"[{get_timestamp()}] Saved CSV report in {time.time() - csv_start:.2f}s")
    else:
        print(f"[{get_timestamp()}] (Skipped writing CSV for panels to save disk space; parquet is available).")
        
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

    # Set output analysis Zarr locations next to the source files
    stage3_analysis_zarr = str(Path(stage3_path).parent / "stage3_analysis.zarr")
    stage4_analysis_zarr = str(Path(stage4_path).parent / "stage4_analysis.zarr")

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

    # ------------------ STAGE 3 ANALYSIS ------------------
    stage3_data = load_zarr_dataset(stage3_path, ['panel_embeddings', 'panel_masks'])
    if stage3_data:
        panels_raw = stage3_data['panel_embeddings']
        masks_raw = stage3_data['panel_masks']
        total_pages = panels_raw.shape[0]
        aligned_metadata = metadata[:total_pages]
        
        # 1a. Stage 3 Page Embeddings (972K Pages)
        print(f"\n[{get_timestamp()}] Loading Stage 3 panel vectors for all {total_pages:,} pages...")
        stage3_pages = np.empty((total_pages, panels_raw.shape[2]), dtype=np.float32)
        metadata_list_pages = []
        
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
                
                metadata_list_pages.append({
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
            metadata_list_pages,
            stage3_analysis_zarr,
            is_panel=False
        )
        del stage3_pages
        del metadata_list_pages
        
        # 1b. Stage 3 Panel Embeddings (All Panels - No Sampling)
        print(f"\n[{get_timestamp()}] Counting total valid Stage 3 panels...")
        total_panels = int(masks_raw[:].sum())
        print(f"[{get_timestamp()}] Total panels to load and process: {total_panels:,}")
        
        stage3_panels = np.empty((total_panels, panels_raw.shape[2]), dtype=np.float32)
        metadata_list_panels = []
        
        idx_counter = 0
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
                n_valid = len(valid)
                if n_valid > 0:
                    stage3_panels[idx_counter : idx_counter + n_valid] = valid
                    for panel_idx in range(n_valid):
                        metadata_list_panels.append({
                            'original_index': page_idx,
                            'panel_index': panel_idx,
                            'canonical_id': aligned_metadata[page_idx]['canonical_id'],
                            'panel_count': n_valid
                        })
                    idx_counter += n_valid
            
            chunk_duration = time.time() - t_chunk_start
            elapsed = time.time() - t_load_start
            progress = end / total_pages
            print(f"  [{get_timestamp()}] Loading panels: {end:,}/{total_pages:,} ({progress*100:.1f}%) | Chunk time: {chunk_duration:.2f}s | Elapsed: {elapsed:.1f}s")
            
        analyze_and_plot(
            stage3_panels, 
            n_clusters, 
            "Stage 3 Panel Embeddings (Static Multimodal All Panels)", 
            str(output_dir / "umap_stage3_panels.png"),
            metadata_list_panels,
            stage3_analysis_zarr,
            is_panel=True
        )
        del stage3_panels
        del metadata_list_panels
        
    else:
        print(f"[{get_timestamp()}] Warning: Stage 3 Zarr not found at {stage3_path}")

    # ------------------ STAGE 4 ANALYSIS ------------------
    stage4_data = load_zarr_dataset(stage4_path, ['strip_embeddings', 'contextualized_panels', 'panel_masks'])
    if stage4_data:
        strip_embeddings = stage4_data['strip_embeddings'][:]
        context_raw = stage4_data['contextualized_panels']
        masks_raw = stage4_data['panel_masks']
        total_pages = strip_embeddings.shape[0]
        aligned_metadata = metadata[:total_pages]
        
        # 2a. Stage 4 Page/Strip Embeddings (972K Pages)
        metadata_list_pages = []
        for i in range(total_pages):
            metadata_list_pages.append({
                'original_index': i,
                'canonical_id': aligned_metadata[i]['canonical_id'],
                'panel_count': int(masks_raw[i].sum())
            })
            
        analyze_and_plot(
            strip_embeddings, 
            n_clusters, 
            "Stage 4 Page/Strip Embeddings (Sequence Transformer Aggregated)", 
            str(output_dir / "umap_stage4_pages.png"),
            metadata_list_pages,
            stage4_analysis_zarr,
            is_panel=False
        )
        del strip_embeddings
        del metadata_list_pages
        
        # 2b. Stage 4 Panel Embeddings (All Panels - No Sampling)
        print(f"\n[{get_timestamp()}] Counting total valid Stage 4 panels...")
        total_panels = int(masks_raw[:].sum())
        print(f"[{get_timestamp()}] Total panels to load and process: {total_panels:,}")
        
        stage4_panels = np.empty((total_panels, context_raw.shape[2]), dtype=np.float32)
        metadata_list_panels = []
        
        idx_counter = 0
        chunk_size = 50000
        t_load_start = time.time()
        for start in range(0, total_pages, chunk_size):
            t_chunk_start = time.time()
            end = min(start + chunk_size, total_pages)
            
            p_chunk = context_raw[start:end]
            m_chunk = masks_raw[start:end]
            
            for i in range(end - start):
                page_idx = start + i
                mask = m_chunk[i].astype(bool)
                valid = p_chunk[i][mask]
                n_valid = len(valid)
                if n_valid > 0:
                    stage4_panels[idx_counter : idx_counter + n_valid] = valid
                    for panel_idx in range(n_valid):
                        metadata_list_panels.append({
                            'original_index': page_idx,
                            'panel_index': panel_idx,
                            'canonical_id': aligned_metadata[page_idx]['canonical_id'],
                            'panel_count': n_valid
                        })
                    idx_counter += n_valid
            
            chunk_duration = time.time() - t_chunk_start
            elapsed = time.time() - t_load_start
            progress = end / total_pages
            print(f"  [{get_timestamp()}] Loading panels: {end:,}/{total_pages:,} ({progress*100:.1f}%) | Chunk time: {chunk_duration:.2f}s | Elapsed: {elapsed:.1f}s")
            
        analyze_and_plot(
            stage4_panels, 
            n_clusters, 
            "Stage 4 Contextualized Panel Embeddings (Transformer All Panels)", 
            str(output_dir / "umap_stage4_panels.png"),
            metadata_list_panels,
            stage4_analysis_zarr,
            is_panel=True
        )
        del stage4_panels
        del metadata_list_panels
        
    else:
        print(f"[{get_timestamp()}] Warning: Stage 4 Zarr not found at {stage4_path}")

    # Generate a Markdown report section for the user's README
    print(f"\n[{get_timestamp()}] Generating analysis summary...")
    report_content = f"""### Dimensionality Reduction & Clustering Analysis (UMAP + KMeans)

To inspect the structural organization of our semantic embedding space, we ran UMAP dimensionality reduction and KMeans clustering (K=10) on the entire page-level and panel-level datasets for both Stage 3 and Stage 4.

The numerical mappings (indices, UMAP coordinates, and cluster assignments) have been saved to high-performance Zarr stores and Parquet datasets (for fast, structured loading in Pandas/Polars).

#### 1. Page Embeddings
*   **Stage 3 (Mean-Pooled)**: [umap_stage3_pages.png](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.png) | [umap_stage3_pages.parquet](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.parquet) (also backup [umap_stage3_pages.csv](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_pages.csv))
*   **Stage 4 (Sequence Aggregated)**: [umap_stage4_pages.png](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.png) | [umap_stage4_pages.parquet](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.parquet) (also backup [umap_stage4_pages.csv](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_pages.csv))

#### 2. Panel Embeddings (All Panels - No Sampling)
*   **Stage 3 (Static Multimodal)**: [umap_stage3_panels.png](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_panels.png) | [umap_stage3_panels.parquet](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage3_panels.parquet)
*   **Stage 4 (Transformer)**: [umap_stage4_panels.png](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_panels.png) | [umap_stage4_panels.parquet](file:///C:/Users/Richard/OneDrive/GIT/Comic-Analysis/documentation/plots/umap_stage4_panels.parquet)

#### 3. High-Performance Zarr Outputs
*   **Stage 3 Analysis**: `E:/stage3_analysis.zarr` (contains arrays: `page_umap`, `page_clusters`, `panel_umap`, `panel_clusters`)
*   **Stage 4 Analysis**: `E:/stage4_analysis.zarr` (contains arrays: `page_umap`, `page_clusters`, `panel_umap`, `panel_clusters`)

*All generated plots, Parquets, and Zarrs are saved in the project.*
"""
    with open("documentation/plots/analysis_report.md", "w") as f:
        f.write(report_content)
    print(f"[{get_timestamp()}] Report written to documentation/plots/analysis_report.md")

if __name__ == "__main__":
    main()
