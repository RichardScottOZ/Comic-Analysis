import os
import numpy as np
import xarray as xr
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

def _to_wsl_path(p: str) -> str:
    """Convert Windows drive paths (e.g., E:\foo or E:/foo or E:foo) to WSL (/mnt/e/foo) when on POSIX.
    If not POSIX or not a drive path, return unchanged.
    """
    try:
        if not isinstance(p, str):
            return p
        if os.name != 'posix':
            return p
        m = re.match(r'^[A-Za-z]:(.*)$', p)
        if m:
            drive = p[0].lower()
            rest = m.group(1)
            # Insert leading slash if missing
            if rest and not (rest.startswith('\\') or rest.startswith('/')):
                rest = '/' + rest
            rest = rest.replace('\\', '/')
            return f"/mnt/{drive}{rest}"
        return p
    except Exception:
        return p

# Configuration
ZARR_PATH = _to_wsl_path("E:\\calibre3\\combined_embeddings.zarr") # Placeholder, user needs to confirm
N_COMPONENTS = 2 # For UMAP (2D visualization)
N_CLUSTERS = 10 # For KMeans (can be adjusted)

def load_zarr_dataset(zarr_path: str) -> xr.Dataset:
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr dataset not found at: {zarr_path}")
    return xr.open_zarr(zarr_path)

def main():
    print(f"Loading Zarr dataset from {ZARR_PATH}...")
    ds = load_zarr_dataset(ZARR_PATH)

    # --- Page Embeddings Analysis ---
    print("Analyzing page_embeddings...")
    page_embeddings = ds['page_embeddings'].values
    page_ids = ds['page_id'].values

    # Filter out zero/empty embeddings if any
    norms = np.linalg.norm(page_embeddings, axis=1)
    valid_indices = norms > 1e-6
    valid_page_embeddings = page_embeddings[valid_indices]
    valid_page_ids = page_ids[valid_indices]

    if len(valid_page_embeddings) == 0:
        print("No valid page embeddings found for analysis.")
        return

    print(f"Found {len(valid_page_embeddings)} valid page embeddings.")

    # UMAP Dimensionality Reduction
    print(f"Performing UMAP reduction to {N_COMPONENTS} dimensions...")
    reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=42)
    page_embeddings_2d = reducer.fit_transform(valid_page_embeddings)

    # K-Means Clustering
    print(f"Performing K-Means clustering with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    page_clusters = kmeans.fit_predict(valid_page_embeddings) # Cluster on high-dim embeddings

    # Prepare data for CSV output
    # Extract panel coordinates for valid pages
    valid_panel_coords = ds['panel_coordinates'].values[valid_indices]
    valid_panel_mask = ds['panel_mask'].values[valid_indices]

    # Calculate compositional features
    panel_widths = np.where(valid_panel_mask, valid_panel_coords[:, :, 2], np.nan)
    panel_heights = np.where(valid_panel_mask, valid_panel_coords[:, :, 3], np.nan)
    panel_areas = panel_widths * panel_heights
    panel_aspect_ratios = np.where(panel_heights > 0, panel_widths / panel_heights, np.nan)

    # Compute mean and std, ignoring NaNs (for padded panels)
    avg_panel_width_ratio = np.nanmean(panel_widths, axis=1)
    std_panel_width_ratio = np.nanstd(panel_widths, axis=1)
    avg_panel_height_ratio = np.nanmean(panel_heights, axis=1)
    std_panel_height_ratio = np.nanstd(panel_heights, axis=1)
    avg_panel_aspect_ratio = np.nanmean(panel_aspect_ratios, axis=1)
    std_panel_aspect_ratio = np.nanstd(panel_aspect_ratios, axis=1)
    avg_panel_area_ratio = np.nanmean(panel_areas, axis=1)
    std_panel_area_ratio = np.nanstd(panel_areas, axis=1)

    results_df = pd.DataFrame({
        'page_id': valid_page_ids,
        'manifest_path': ds['manifest_path'].values[valid_indices],
        'UMAP_dim1': page_embeddings_2d[:, 0],
        'UMAP_dim2': page_embeddings_2d[:, 1],
        'cluster_id': page_clusters,
        'panel_count': ds['panel_mask'].values[valid_indices].sum(axis=1),
        'has_text': (ds['text_content'].values[valid_indices] != '').any(axis=1),
        'total_text_length': np.array([sum(len(t) for t in texts if isinstance(t, str)) for texts in ds['text_content'].values[valid_indices]]),
        'source': ds['source'].values[valid_indices],
        'series': ds['series'].values[valid_indices],
        'volume': ds['volume'].values[valid_indices],
        'issue': ds['issue'].values[valid_indices],
        'page_num': ds['page_num'].values[valid_indices],
        'avg_panel_width_ratio': avg_panel_width_ratio,
        'std_panel_width_ratio': std_panel_width_ratio,
        'avg_panel_height_ratio': avg_panel_height_ratio,
        'std_panel_height_ratio': std_panel_height_ratio,
        'avg_panel_aspect_ratio': avg_panel_aspect_ratio,
        'std_panel_aspect_ratio': std_panel_aspect_ratio,
        'avg_panel_area_ratio': avg_panel_area_ratio,
        'std_panel_area_ratio': std_panel_area_ratio,
    })

    # Calculate compositional features
    # Ensure to handle division by zero for aspect ratio where height might be 0 or very small
    panel_widths = np.where(valid_panel_mask, valid_panel_coords[:, :, 2], np.nan)
    panel_heights = np.where(valid_panel_mask, valid_panel_coords[:, :, 3], np.nan)
    
    # Calculate aspect ratios carefully to avoid division by zero
    # Replace 0 or near-zero heights with NaN for division
    panel_heights_safe = np.where(panel_heights > 1e-6, panel_heights, np.nan)
    panel_aspect_ratios = np.where(~np.isnan(panel_heights_safe), panel_widths / panel_heights_safe, np.nan)
    
    panel_areas = panel_widths * panel_heights

    # Compute mean and std, ignoring NaNs (for padded panels or invalid aspect ratios)
    avg_panel_width_ratio = np.nanmean(panel_widths, axis=1)
    std_panel_width_ratio = np.nanstd(panel_widths, axis=1)
    avg_panel_height_ratio = np.nanmean(panel_heights, axis=1)
    std_panel_height_ratio = np.nanstd(panel_heights, axis=1)
    avg_panel_aspect_ratio = np.nanmean(panel_aspect_ratios, axis=1)
    std_panel_aspect_ratio = np.nanstd(panel_aspect_ratios, axis=1)
    avg_panel_area_ratio = np.nanmean(panel_areas, axis=1)
    std_panel_area_ratio = np.nanstd(panel_areas, axis=1)

    # Calculate text length per panel
    text_content_lengths = np.array([[len(t) if isinstance(t, str) else 0 for t in panel_texts] for panel_texts in ds['text_content'].values[valid_indices]])
    # Mask out lengths for invalid panels
    text_content_lengths_masked = np.where(valid_panel_mask, text_content_lengths, np.nan)

    avg_text_length_per_panel = np.nanmean(text_content_lengths_masked, axis=1)
    max_text_length_per_panel = np.nanmax(text_content_lengths_masked, axis=1)
    # Handle pages with no valid panels (nanmean/nanmax would return NaN)
    avg_text_length_per_panel = np.nan_to_num(avg_text_length_per_panel, nan=0.0)
    max_text_length_per_panel = np.nan_to_num(max_text_length_per_panel, nan=0.0)

    results_df = pd.DataFrame({
        'page_id': valid_page_ids,
        'manifest_path': ds['manifest_path'].values[valid_indices],
        'UMAP_dim1': page_embeddings_2d[:, 0],
        'UMAP_dim2': page_embeddings_2d[:, 1],
        'cluster_id': page_clusters,
        'panel_count': ds['panel_mask'].values[valid_indices].sum(axis=1),
        'has_text': (ds['text_content'].values[valid_indices] != '').any(axis=1),
        'total_text_length': np.array([sum(len(t) for t in texts if isinstance(t, str)) for texts in ds['text_content'].values[valid_indices]]),
        'source': ds['source'].values[valid_indices],
        'series': ds['series'].values[valid_indices],
        'volume': ds['volume'].values[valid_indices],
        'issue': ds['issue'].values[valid_indices],
        'page_num': ds['page_num'].values[valid_indices],
        'avg_panel_width_ratio': avg_panel_width_ratio,
        'std_panel_width_ratio': std_panel_width_ratio,
        'avg_panel_height_ratio': avg_panel_height_ratio,
        'std_panel_height_ratio': std_panel_height_ratio,
        'avg_panel_aspect_ratio': avg_panel_aspect_ratio,
        'std_panel_aspect_ratio': std_panel_aspect_ratio,
        'avg_panel_area_ratio': avg_panel_area_ratio,
        'std_panel_area_ratio': std_panel_area_ratio,
        'avg_text_length_per_panel': avg_text_length_per_panel,
        'max_text_length_per_panel': max_text_length_per_panel,
    })

    # Save results to CSV
    csv_output_path = 'page_embeddings_clusters_metadata.csv'
    results_df.to_csv(csv_output_path, index=False)
    print(f"Saved cluster metadata to {csv_output_path}")

    # --- Plotting ---
    # Plot 1: UMAP Projection with Clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='cluster_id',
        palette=sns.color_palette("hsv", N_CLUSTERS),
        alpha=0.7,
        data=results_df
    )
    plt.title(f'UMAP Projection of Page Embeddings (Clustered with KMeans, {N_CLUSTERS} clusters)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_clusters.png')
    plt.close() # Close plot to free memory

    # Plot 2: Box Plot - Total Text Length per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='total_text_length', data=results_df)
    plt.title('Total Text Length Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Total Text Length')
    plt.savefig('total_text_length_per_cluster_boxplot.png')
    plt.close()

    # Plot 3: Box Plot - Panel Count per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='panel_count', data=results_df)
    plt.title('Panel Count Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Panel Count')
    plt.savefig('panel_count_per_cluster_boxplot.png')
    plt.close()

    # Plot 4: Box Plot - Average Panel Aspect Ratio per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='avg_panel_aspect_ratio', data=results_df)
    plt.title('Average Panel Aspect Ratio Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Panel Aspect Ratio')
    plt.savefig('avg_panel_aspect_ratio_per_cluster_boxplot.png')
    plt.close()

    # Plot 5: Box Plot - Average Panel Width Ratio per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='avg_panel_width_ratio', data=results_df)
    plt.title('Average Panel Width Ratio Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Panel Width Ratio')
    plt.savefig('avg_panel_width_ratio_per_cluster_boxplot.png')
    plt.close()

    # Plot 6: Box Plot - Average Panel Height Ratio per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='avg_panel_height_ratio', data=results_df)
    plt.title('Average Panel Height Ratio Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Panel Height Ratio')
    plt.savefig('avg_panel_height_ratio_per_cluster_boxplot.png')
    plt.close()

    # Plot 7: Scatter Plot - UMAP colored by Panel Count
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='panel_count',
        palette='viridis', # Use a sequential palette
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Panel Count Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_panel_count.png')
    plt.close()

    # Plot 8: Scatter Plot - UMAP colored by Total Text Length
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='total_text_length',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Total Text Length Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_total_text_length.png')
    plt.close()

    # Plot 9: Scatter Plot - UMAP colored by Average Panel Width Ratio
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='avg_panel_width_ratio',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Average Panel Width Ratio Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_avg_panel_width_ratio.png')
    plt.close()

    # Plot 10: Scatter Plot - UMAP colored by Average Panel Height Ratio
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='avg_panel_height_ratio',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Average Panel Height Ratio Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_avg_panel_height_ratio.png')
    plt.close()

    # Plot 11: Scatter Plot - UMAP colored by Average Panel Aspect Ratio
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='avg_panel_aspect_ratio',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Average Panel Aspect Ratio Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_avg_panel_aspect_ratio.png')
    plt.close()

    # Plot 12: Box Plot - Average Text Length per Panel per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='avg_text_length_per_panel', data=results_df)
    plt.title('Average Text Length per Panel Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Text Length per Panel')
    plt.savefig('avg_text_length_per_panel_boxplot.png')
    plt.close()

    # Plot 13: Box Plot - Max Text Length per Panel per Cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster_id', y='max_text_length_per_panel', data=results_df)
    plt.title('Max Text Length per Panel Distribution per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Max Text Length per Panel')
    plt.savefig('max_text_length_per_panel_boxplot.png')
    plt.close()

    # Plot 14: Scatter Plot - UMAP colored by Average Text Length per Panel
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='avg_text_length_per_panel',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Average Text Length per Panel Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_avg_text_length_per_panel.png')
    plt.close()

    # Plot 15: Scatter Plot - UMAP colored by Max Text Length per Panel
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='UMAP_dim1',
        y='UMAP_dim2',
        hue='max_text_length_per_panel',
        palette='viridis',
        alpha=0.7,
        data=results_df
    )
    plt.title('UMAP Projection with Max Text Length per Panel Coloring')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig('page_embeddings_umap_max_text_length_per_panel.png')
    plt.close()

if __name__ == "__main__":
    main()

    # --- Panel Embeddings Analysis (Optional, can be added later if needed) ---
    # This would follow a similar pattern

if __name__ == "__main__":
    main()
