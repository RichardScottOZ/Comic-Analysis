
import os
import csv
import numpy as np
import xarray as xr
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from tqdm import tqdm

def analyze_panel_text(panel):
    """Analyzes the text field of a single panel and returns a category, stats, and keys."""
    text_info = panel.get('text', {})
    if not text_info or not isinstance(text_info, dict):
        return 'no_text', False, 0, 0, 0, set()

    keys = set(text_info.keys())
    has_dialogue = bool(text_info.get('dialogue'))
    has_narration = bool(text_info.get('narration'))
    has_sfx = bool(text_info.get('sfx'))

    dialogue_len = sum(len(t) for t in text_info.get('dialogue', []))
    narration_len = sum(len(t) for t in text_info.get('narration', []))
    sfx_len = sum(len(t) for t in text_info.get('sfx', []))

    category = 'no_text'
    if has_dialogue and not has_narration and not has_sfx:
        category = 'dialogue_only'
    elif not has_dialogue and has_narration and not has_sfx:
        category = 'narration_only'
    elif not has_dialogue and not has_narration and has_sfx:
        category = 'sfx_only'
    elif has_dialogue and has_narration and not has_sfx:
        category = 'dialogue_and_narration'
    elif has_dialogue and not has_narration and has_sfx:
        category = 'dialogue_and_sfx'
    elif not has_dialogue and has_narration and has_sfx:
        category = 'narration_and_sfx'
    elif has_dialogue and has_narration and has_sfx:
        category = 'all_three'
    elif keys:
        category = 'custom_keys_only'

    is_descriptive = False
    if has_narration:
        narration_text = ' '.join(text_info.get('narration', [])).lower()
        descriptive_words = ['shows', 'depicts', 'image of', 'close-up', 'background', 'figure', 'scene']
        if narration_len > 100 and any(word in narration_text for word in descriptive_words):
            is_descriptive = True

    return category, is_descriptive, dialogue_len, narration_len, sfx_len, keys

def main():
    parser = argparse.ArgumentParser(description='Perform a combined analysis of embeddings, clustering, and text content.')
    parser.add_argument('--zarr_path', required=True, help='Path to the combined_embeddings.zarr file.')
    parser.add_argument('--output_csv', required=True, help='Path to save the output CSV report.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-Means.')
    parser.add_argument('--save_plots', action='store_true', help='If set, save UMAP and distribution plots.')
    parser.add_argument('--plots_dir', default='./combo_plots', help='Directory to save plots.')
    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        raise FileNotFoundError(f"Zarr dataset not found at: {args.zarr_path}")

    print("Loading Zarr dataset...")
    ds = xr.open_zarr(args.zarr_path)

    print("Analyzing page embeddings...")
    page_embeddings = ds['page_embeddings'].values
    page_ids = ds['page_id'].values

    norms = np.linalg.norm(page_embeddings, axis=1)
    valid_indices = np.where(norms > 1e-6)[0]
    valid_page_embeddings = page_embeddings[valid_indices]
    valid_page_ids = page_ids[valid_indices]

    if len(valid_page_embeddings) == 0:
        print("No valid page embeddings found.")
        return

    print(f"Found {len(valid_page_embeddings)} valid page embeddings.")

    print("Performing UMAP reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    page_embeddings_2d = reducer.fit_transform(valid_page_embeddings)

    print(f"Performing K-Means clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    page_clusters = kmeans.fit_predict(valid_page_embeddings)

    # Create a mapping from valid page_id to its cluster and UMAP data
    page_id_to_analysis = {}
    for i, page_id in enumerate(valid_page_ids):
        page_id_to_analysis[page_id] = {
            'cluster_id': page_clusters[i],
            'UMAP_dim1': page_embeddings_2d[i, 0],
            'UMAP_dim2': page_embeddings_2d[i, 1]
        }

    # Now, iterate through the full dataset to generate the per-panel report
    all_manifest_paths = ds['manifest_path'].values
    all_page_ids = ds['page_id'].values
    all_panel_embeddings = ds['panel_embeddings'].values
    all_panel_masks = ds['panel_mask'].values

    unique_text_keys = set()
    missing_manifest_count = 0
    
    output_rows = []

    print(f"Generating detailed report for {len(all_manifest_paths)} pages...")
    for i, manifest_path in enumerate(tqdm(all_manifest_paths, desc="Processing pages")):
        page_id = all_page_ids[i]
        page_embedding_norm = float(np.linalg.norm(page_embeddings[i]))
        panel_mask_sum = int(all_panel_masks[i].sum())

        analysis_data = page_id_to_analysis.get(page_id, {})
        cluster_id = analysis_data.get('cluster_id', -1)
        umap_dim1 = analysis_data.get('UMAP_dim1', 0.0)
        umap_dim2 = analysis_data.get('UMAP_dim2', 0.0)

        if not os.path.exists(manifest_path):
            missing_manifest_count += 1
            if missing_manifest_count <= 20:
                print(f"Warning: Manifest path not found: {manifest_path}")
            # Add a single row for the page with error info
            output_rows.append({
                'page_index': i, 'manifest_path': manifest_path, 'page_id': page_id,
                'cluster_id': cluster_id, 'UMAP_dim1': umap_dim1, 'UMAP_dim2': umap_dim2,
                'page_embedding_norm': page_embedding_norm, 'panel_mask_sum': panel_mask_sum,
                'panel_index': 'N/A', 'panel_embedding_mean': 0.0, 'text_category': 'MANIFEST_NOT_FOUND',
                'is_descriptive_narration': False, 'dialogue_char_count': 0, 'narration_char_count': 0, 'sfx_char_count': 0
            })
            continue

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
        except Exception as e:
            output_rows.append({
                'page_index': i, 'manifest_path': manifest_path, 'page_id': page_id,
                'cluster_id': cluster_id, 'UMAP_dim1': umap_dim1, 'UMAP_dim2': umap_dim2,
                'page_embedding_norm': page_embedding_norm, 'panel_mask_sum': panel_mask_sum,
                'panel_index': 'N/A', 'panel_embedding_mean': 0.0, 'text_category': f'JSON_LOAD_ERROR: {e}',
                'is_descriptive_narration': False, 'dialogue_char_count': 0, 'narration_char_count': 0, 'sfx_char_count': 0
            })
            continue

        panels = page_data.get('panels', [])
        if not panels:
            output_rows.append({
                'page_index': i, 'manifest_path': manifest_path, 'page_id': page_id,
                'cluster_id': cluster_id, 'UMAP_dim1': umap_dim1, 'UMAP_dim2': umap_dim2,
                'page_embedding_norm': page_embedding_norm, 'panel_mask_sum': panel_mask_sum,
                'panel_index': 'N/A', 'panel_embedding_mean': 0.0, 'text_category': 'no_panels_in_json',
                'is_descriptive_narration': False, 'dialogue_char_count': 0, 'narration_char_count': 0, 'sfx_char_count': 0
            })
        else:
            for panel_idx, panel in enumerate(panels):
                if panel_idx >= all_panel_embeddings.shape[1]:
                    break

                category, is_descriptive, dialogue_len, narration_len, sfx_len, keys = analyze_panel_text(panel)
                unique_text_keys.update(keys)

                panel_embedding_mean = float(np.mean(all_panel_embeddings[i, panel_idx]))

                output_rows.append({
                    'page_index': i, 'manifest_path': manifest_path, 'page_id': page_id,
                    'cluster_id': cluster_id, 'UMAP_dim1': umap_dim1, 'UMAP_dim2': umap_dim2,
                    'page_embedding_norm': page_embedding_norm, 'panel_mask_sum': panel_mask_sum,
                    'panel_index': panel_idx, 'panel_embedding_mean': panel_embedding_mean, 'text_category': category,
                    'is_descriptive_narration': is_descriptive, 'dialogue_char_count': dialogue_len, 
                    'narration_char_count': narration_len, 'sfx_char_count': sfx_len
                })

    # Write to CSV
    print(f"\nWriting {len(output_rows)} rows to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = output_rows[0].keys() if output_rows else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nAnalysis complete!")
    if missing_manifest_count > 0:
        print(f"Warning: {missing_manifest_count} manifest paths were not found.")
    print(f"Found the following unique text keys across all panels: {unique_text_keys}")

    if args.save_plots:
        print("Generating and saving plots...")
        if not os.path.exists(args.plots_dir):
            os.makedirs(args.plots_dir)
        
        report_df = pd.DataFrame(output_rows)
        # For page-level plots, we need to aggregate the per-panel data
        # We also need to sum the character counts for the page-level plots
        page_level_df = report_df.groupby('page_id').agg({
            'cluster_id': 'first',
            'UMAP_dim1': 'first',
            'UMAP_dim2': 'first',
            'narration_char_count': 'sum',
            'dialogue_char_count': 'sum',
            'sfx_char_count': 'sum'
        }).reset_index()

        # Create log-scaled versions of the counts for better visualization
        page_level_df['log_narration_char_count'] = np.log1p(page_level_df['narration_char_count'])
        page_level_df['log_dialogue_char_count'] = np.log1p(page_level_df['dialogue_char_count'])
        page_level_df['log_sfx_char_count'] = np.log1p(page_level_df['sfx_char_count'])

        # Plot 1: UMAP Projection with Clusters
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='UMAP_dim1', y='UMAP_dim2', hue='cluster_id', palette=sns.color_palette("hsv", args.n_clusters), alpha=0.7, data=page_level_df)
        plt.title(f'UMAP Projection of Page Embeddings (Clustered with KMeans, {args.n_clusters} clusters)')
        plt.savefig(os.path.join(args.plots_dir, 'umap_clusters.png'))
        plt.close()

        # Plot: UMAP colored by Narration Character Count (Log Scale)
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='UMAP_dim1', y='UMAP_dim2', hue='log_narration_char_count', palette='viridis', alpha=0.7, data=page_level_df)
        plt.title('UMAP Projection with Narration Character Count (Log Scale)')
        plt.savefig(os.path.join(args.plots_dir, 'umap_narration_char_count_log.png'))
        plt.close()

        # Plot: UMAP colored by Dialogue Character Count (Log Scale)
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='UMAP_dim1', y='UMAP_dim2', hue='log_dialogue_char_count', palette='viridis', alpha=0.7, data=page_level_df)
        plt.title('UMAP Projection with Dialogue Character Count (Log Scale)')
        plt.savefig(os.path.join(args.plots_dir, 'umap_dialogue_char_count_log.png'))
        plt.close()

        # Plot: UMAP colored by SFX Character Count (Log Scale)
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='UMAP_dim1', y='UMAP_dim2', hue='log_sfx_char_count', palette='viridis', alpha=0.7, data=page_level_df)
        plt.title('UMAP Projection with SFX Character Count (Log Scale)')
        plt.savefig(os.path.join(args.plots_dir, 'umap_sfx_char_count_log.png'))
        plt.close()

        print(f"Plots saved to {args.plots_dir}")

if __name__ == '__main__':
    main()
