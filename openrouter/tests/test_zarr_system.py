"""
Test script for Zarr-based embedding system
"""

import os
import numpy as np
import xarray as xr
import json
from pathlib import Path

def test_zarr_creation():
    """Test creating a small Zarr dataset"""
    
    print("Testing Zarr dataset creation...")
    
    # Create test data
    n_pages = 100
    max_panels = 12
    embedding_dim = 384
    
    # Create coordinates
    page_ids = [f"amazon_test_series_v01_{i:03d}_p001" for i in range(n_pages)]
    sources = ['amazon'] * n_pages
    series = ['test_series'] * n_pages
    volumes = ['v01'] * n_pages
    issues = [f"{i:03d}" for i in range(n_pages)]
    pages = ['p001'] * n_pages
    
    coords = {
        'page_id': page_ids,
        'source': ('page_id', sources),
        'series': ('page_id', series),
        'volume': ('page_id', volumes),
        'issue': ('page_id', issues),
        'page_num': ('page_id', pages),
        'panel_id': range(max_panels),
        'embedding_dim': range(embedding_dim),
        'coord_dim': range(4)
    }
    
    # Create XArray dataset
    ds = xr.Dataset(coords=coords)
    
    # Add data variables
    ds['panel_embeddings'] = (['page_id', 'panel_id', 'embedding_dim'], 
                              np.random.randn(n_pages, max_panels, embedding_dim).astype(np.float32))
    ds['page_embeddings'] = (['page_id', 'embedding_dim'], 
                             np.random.randn(n_pages, embedding_dim).astype(np.float32))
    ds['attention_weights'] = (['page_id', 'panel_id'], 
                               np.random.rand(n_pages, max_panels).astype(np.float32))
    ds['reading_order'] = (['page_id', 'panel_id'], 
                           np.random.randn(n_pages, max_panels).astype(np.float32))
    ds['panel_coordinates'] = (['page_id', 'panel_id', 'coord_dim'], 
                               np.random.rand(n_pages, max_panels, 4).astype(np.float32))
    ds['text_content'] = (['page_id', 'panel_id'], 
                          np.array([['test text'] * max_panels] * n_pages, dtype=object))
    ds['panel_mask'] = (['page_id', 'panel_id'], 
                        np.random.rand(n_pages, max_panels) > 0.5)
    
    # Add attributes
    ds.attrs['model_name'] = 'CLOSURE-Lite-Simple-Test'
    ds.attrs['embedding_dim'] = embedding_dim
    ds.attrs['max_panels'] = max_panels
    ds.attrs['test_dataset'] = True
    
    # Save to Zarr
    output_path = 'test_embeddings.zarr'
    ds.to_zarr(output_path, mode='w')
    
    print(f"Test dataset created: {output_path}")
    print(f"Dataset shape: {ds.dims}")
    print(f"Total pages: {len(ds.page_id)}")
    
    return output_path

def test_zarr_loading(zarr_path):
    """Test loading and querying Zarr dataset"""
    
    print(f"\nTesting Zarr dataset loading: {zarr_path}")
    
    # Load dataset
    ds = xr.open_zarr(zarr_path)
    
    print(f"Dataset loaded successfully")
    print(f"Dimensions: {ds.dims}")
    print(f"Data variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    
    # Test basic queries
    print(f"\nTesting basic queries...")
    
    # Get first page
    first_page = ds.isel(page_id=0)
    print(f"First page ID: {first_page.page_id.values}")
    print(f"First page source: {first_page.source.values}")
    print(f"First page series: {first_page.series.values}")
    
    # Get page embeddings shape
    page_embeddings = first_page['page_embeddings'].values
    print(f"Page embedding shape: {page_embeddings.shape}")
    
    # Get panel embeddings shape
    panel_embeddings = first_page['panel_embeddings'].values
    print(f"Panel embeddings shape: {panel_embeddings.shape}")
    
    # Test filtering
    print(f"\nTesting filtering...")
    
    # Filter by source (using where instead of sel for non-indexed coordinates)
    amazon_pages = ds.where(ds['source'] == 'amazon', drop=True)
    print(f"Amazon pages: {len(amazon_pages.page_id)}")
    
    # Filter by series
    test_series = ds.where(ds['series'] == 'test_series', drop=True)
    print(f"Test series pages: {len(test_series.page_id)}")
    
    # Test similarity calculation
    print(f"\nTesting similarity calculation...")
    
    # Get two page embeddings
    page1_emb = ds['page_embeddings'].isel(page_id=0).values
    page2_emb = ds['page_embeddings'].isel(page_id=1).values
    
    # Calculate cosine similarity
    similarity = np.dot(page1_emb, page2_emb) / (np.linalg.norm(page1_emb) * np.linalg.norm(page2_emb))
    print(f"Cosine similarity between page 0 and page 1: {similarity:.4f}")
    
    return ds

def test_standardization():
    """Test path standardization functions"""
    
    print(f"\nTesting path standardization...")
    
    # Test Amazon paths
    amazon_paths = [
        r"E:\amazon\Batman The Dark Knight Detective v03\Batman The Dark Knight Detective v03 - p001.jpg",
        r"E:\amazon\X-Men Uncanny v01\X-Men Uncanny v01 - p005.jpg",
        r"E:\amazon\Spider-Man Amazing v02\Spider-Man Amazing v02 - p012.jpg"
    ]
    
    for path in amazon_paths:
        # Import the standardization function
        import sys
        sys.path.append('benchmarks/detections/openrouter')
        from generate_embeddings_zarr import standardize_path
        
        standardized = standardize_path(path, 'amazon')
        print(f"Amazon: {path}")
        print(f"  -> {standardized}")
    
    # Test CalibreComics paths
    calibre_paths = [
        r"E:\CalibreComics\Justice League (2016-2018)\Justice League (2016-2018) 012 - p019.jpg",
        r"E:\CalibreComics\Batman (2016-2021)\Batman (2016-2021) 001 - p001.jpg",
        r"E:\CalibreComics\Wonder Woman (2016-2021)\Wonder Woman (2016-2021) 005 - p008.jpg"
    ]
    
    for path in calibre_paths:
        standardized = standardize_path(path, 'calibre')
        print(f"CalibreComics: {path}")
        print(f"  -> {standardized}")

def main():
    """Run all tests"""
    
    print("🧪 Testing Zarr-based embedding system")
    print("=" * 50)
    
    # Test path standardization
    test_standardization()
    
    # Test Zarr creation
    zarr_path = test_zarr_creation()
    
    # Test Zarr loading and querying
    ds = test_zarr_loading(zarr_path)
    
    print(f"\n✅ All tests completed successfully!")
    print(f"Test dataset saved to: {zarr_path}")
    
    # Clean up (commented out to keep test dataset)
    # import shutil
    # if os.path.exists(zarr_path):
    #     shutil.rmtree(zarr_path)
    #     print(f"Test dataset cleaned up")
    print(f"Test dataset kept at: {zarr_path}")

if __name__ == "__main__":
    main()
