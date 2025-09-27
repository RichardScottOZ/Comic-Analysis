"""
Generate embeddings for comic datasets and store in Zarr format
Supports both Amazon and CalibreComics datasets with standardized naming
"""

import os
import re
import json
import torch
import numpy as np
import xarray as xr
import zarr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

from closure_lite_simple_framework import ClosureLiteSimple
from closure_lite_dataset import create_dataloader

def standardize_path(original_path: str, source: str) -> str:
    """Convert various path formats to standardized naming"""
    
    path_parts = Path(original_path).parts
    
    if source == 'amazon':
        # E:\amazon\Batman The Dark Knight Detective v03\Batman The Dark Knight Detective v03 - p001.jpg
        # -> amazon_batman_dark_knight_detective_v03_001_p001
        
        if len(path_parts) < 2:
            return f"amazon_unknown_unknown_000_p000"
            
        series_dir = path_parts[-2]  # "Batman The Dark Knight Detective v03"
        filename = path_parts[-1]    # "Batman The Dark Knight Detective v03 - p001.jpg"
        
        # Extract series, volume, issue, page
        series = clean_series_name(series_dir)
        volume = extract_volume(series_dir)
        issue = extract_issue(filename)
        page = extract_page(filename)
        
    elif source == 'calibre':
        # E:\CalibreComics\Justice League (2016-2018)\Justice League (2016-2018) 012 - p019.jpg
        # -> calibre_justice_league_2016_012_p019
        
        if len(path_parts) < 2:
            return f"calibre_unknown_unknown_000_p000"
            
        series_dir = path_parts[-2]  # "Justice League (2016-2018)"
        filename = path_parts[-1]    # "Justice League (2016-2018) 012 - p019.jpg"
        
        # Extract series, volume, issue, page
        series = clean_series_name(series_dir)
        volume = extract_volume(series_dir)
        issue = extract_issue(filename)
        page = extract_page(filename)
    
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return f"{source}_{series}_{volume}_{issue}_{page}"

def clean_series_name(name: str) -> str:
    """Clean and standardize series names"""
    # Remove special characters, normalize spaces
    cleaned = re.sub(r'[^\w\s]', '', name)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned.lower()

def extract_volume(name: str) -> str:
    """Extract volume information"""
    # Look for patterns like "v03", "2016-2018", etc.
    volume_match = re.search(r'(v\d+|\(\d{4}-\d{4}\)|\d{4})', name)
    return volume_match.group(1) if volume_match else 'unknown'

def extract_issue(filename: str) -> str:
    """Extract issue number"""
    issue_match = re.search(r'(\d{3})', filename)
    return issue_match.group(1) if issue_match else '000'

def extract_page(filename: str) -> str:
    """Extract page number"""
    page_match = re.search(r'p(\d{3})', filename)
    return f"p{page_match.group(1)}" if page_match else 'p000'

def extract_series(page_id: str) -> str:
    """Extract series from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:-3])  # Everything between source and volume
    return 'unknown'

def extract_volume_from_id(page_id: str) -> str:
    """Extract volume from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-3]  # Volume is third from end
    return 'unknown'

def extract_issue_from_id(page_id: str) -> str:
    """Extract issue from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-2]  # Issue is second from end
    return '000'

def extract_page_from_id(page_id: str) -> str:
    """Extract page from standardized page ID"""
    parts = page_id.split('_')
    if len(parts) >= 2:
        return parts[-1]  # Page is last
    return 'p000'

def load_model(checkpoint_path: str, device: torch.device, num_heads: int = 4, temperature: float = 0.1):
    """Load the trained simple model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ClosureLiteSimple(d=384, num_heads=num_heads, temperature=temperature).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_batch(model, batch, device):
    """Process a batch and extract embeddings"""
    with torch.no_grad():
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Get model outputs
        B, N, _, _, _ = batch['images'].shape
        images = batch['images'].flatten(0, 1)
        input_ids = batch['input_ids'].flatten(0, 1)
        attention_mask = batch['attention_mask'].flatten(0, 1)
        comp_feats = batch['comp_feats'].flatten(0, 1)
        
        # 1. Panel Analysis (raw embeddings)
        P_flat = model.atom(images, input_ids, attention_mask, comp_feats)
        P = P_flat.view(B, N, -1)
        
        # 2. Page-level Understanding
        E_page, attention_weights = model.han.panels_to_page(P, batch['panel_mask'])
        
        # 3. Reading Order Prediction
        logits_neighbors = model.next_head(P)
        
        return {
            'panel_embeddings': P.cpu().numpy(),
            'page_embeddings': E_page.cpu().numpy(),
            'attention_weights': attention_weights.cpu().numpy(),
            'reading_order': logits_neighbors.cpu().numpy(),
            'panel_mask': batch['panel_mask'].cpu().numpy(),
            'next_idx': batch['next_idx'].cpu().numpy(),
            'original_pages': batch['original_page'],
            'json_files': batch['json_file']
        }

def create_zarr_dataset(output_path: str, amazon_data: List[Dict], calibre_data: List[Dict] = None):
    """Create Zarr dataset with standardized coordinates"""
    
    print("Creating standardized coordinates...")
    
    # Combine all page IDs
    all_page_ids = []
    all_sources = []
    all_series = []
    all_volumes = []
    all_issues = []
    all_pages = []
    
    # Process Amazon data
    for page_data in amazon_data:
        page_id = standardize_path(page_data['path'], 'amazon')
        all_page_ids.append(page_id)
        all_sources.append('amazon')
        all_series.append(extract_series(page_id))
        all_volumes.append(extract_volume_from_id(page_id))
        all_issues.append(extract_issue_from_id(page_id))
        all_pages.append(extract_page_from_id(page_id))
    
    # Process CalibreComics data if provided
    if calibre_data:
        for page_data in calibre_data:
            page_id = standardize_path(page_data['path'], 'calibre')
            all_page_ids.append(page_id)
            all_sources.append('calibre')
            all_series.append(extract_series(page_id))
            all_volumes.append(extract_volume_from_id(page_id))
            all_issues.append(extract_issue_from_id(page_id))
            all_pages.append(extract_page_from_id(page_id))
    
    print(f"Total pages: {len(all_page_ids)}")
    print(f"Amazon pages: {len(amazon_data)}")
    if calibre_data:
        print(f"CalibreComics pages: {len(calibre_data)}")
    
    # Create coordinates
    coords = {
        'page_id': all_page_ids,
        'source': ('page_id', all_sources),
        'series': ('page_id', all_series),
        'volume': ('page_id', all_volumes),
        'issue': ('page_id', all_issues),
        'page_num': ('page_id', all_pages),
        'panel_id': range(12),  # Max panels
        'embedding_dim': range(384),
        'coord_dim': range(4)  # x, y, width, height
    }
    
    # Create XArray dataset
    ds = xr.Dataset(coords=coords)
    
    # Add data variables with proper shapes
    n_pages = len(all_page_ids)
    max_panels = 12
    embedding_dim = 384
    
    ds['panel_embeddings'] = (['page_id', 'panel_id', 'embedding_dim'], 
                              np.zeros((n_pages, max_panels, embedding_dim), dtype=np.float32))
    ds['page_embeddings'] = (['page_id', 'embedding_dim'], 
                             np.zeros((n_pages, embedding_dim), dtype=np.float32))
    ds['attention_weights'] = (['page_id', 'panel_id'], 
                               np.zeros((n_pages, max_panels), dtype=np.float32))
    ds['reading_order'] = (['page_id', 'panel_id'], 
                           np.zeros((n_pages, max_panels), dtype=np.float32))
    ds['panel_coordinates'] = (['page_id', 'panel_id', 'coord_dim'], 
                               np.zeros((n_pages, max_panels, 4), dtype=np.float32))
    ds['text_content'] = (['page_id', 'panel_id'], 
                          np.zeros((n_pages, max_panels), dtype=object))
    ds['panel_mask'] = (['page_id', 'panel_id'], 
                        np.zeros((n_pages, max_panels), dtype=bool))
    
    # Configure Zarr backend with compression
    encoding = {
        'panel_embeddings': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12, 384)  # Chunk by 1000 pages
        },
        'page_embeddings': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 384)
        },
        'attention_weights': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'reading_order': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'panel_coordinates': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12, 4)
        },
        'text_content': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        },
        'panel_mask': {
            'compressor': zarr.Blosc(cname='zstd', clevel=6, shuffle=2),
            'chunks': (1000, 12)
        }
    }
    
    # Add global attributes
    ds.attrs['model_name'] = 'CLOSURE-Lite-Simple'
    ds.attrs['embedding_dim'] = 384
    ds.attrs['max_panels'] = 12
    ds.attrs['created_date'] = str(Path().cwd())
    ds.attrs['amazon_pages'] = len(amazon_data)
    if calibre_data:
        ds.attrs['calibre_pages'] = len(calibre_data)
    
    # Save to Zarr
    print(f"Saving to Zarr: {output_path}")
    ds.to_zarr(output_path, encoding=encoding, mode='w')
    
    return ds

def generate_embeddings(model, dataloader, device, output_path: str, amazon_data: List[Dict], calibre_data: List[Dict] = None):
    """Generate embeddings for all pages and save to Zarr"""
    
    # Create Zarr dataset
    ds = create_zarr_dataset(output_path, amazon_data, calibre_data)
    
    print("Generating embeddings...")
    
    # Process batches
    page_idx = 0
    amazon_idx = 0
    calibre_idx = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Process batch
        results = process_batch(model, batch, device)
        
        batch_size = len(batch['original_page'])
        
        for i in range(batch_size):
            # Get page data
            page_data = results['original_pages'][i]
            json_file = results['json_files'][i]
            
            # Determine source and get standardized ID
            if amazon_idx < len(amazon_data) and amazon_data[amazon_idx]['path'] in json_file:
                source = 'amazon'
                page_id = standardize_path(amazon_data[amazon_idx]['path'], 'amazon')
                amazon_idx += 1
            elif calibre_data and calibre_idx < len(calibre_data) and calibre_data[calibre_idx]['path'] in json_file:
                source = 'calibre'
                page_id = standardize_path(calibre_data[calibre_idx]['path'], 'calibre')
                calibre_idx += 1
            else:
                # Fallback - try to determine from path
                if 'amazon' in json_file.lower():
                    source = 'amazon'
                    page_id = standardize_path(json_file, 'amazon')
                else:
                    source = 'calibre'
                    page_id = standardize_path(json_file, 'calibre')
            
            # Find page index in dataset
            try:
                page_index = ds.page_id.values.tolist().index(page_id)
            except ValueError:
                print(f"Warning: Page ID {page_id} not found in dataset, skipping...")
                continue
            
            # Extract data
            panel_embeddings = results['panel_embeddings'][i]  # Shape: (12, 384)
            page_embedding = results['page_embeddings'][i]     # Shape: (384,)
            attention_weights = results['attention_weights'][i] # Shape: (12,)
            reading_order = results['reading_order'][i]        # Shape: (12,)
            panel_mask = results['panel_mask'][i]              # Shape: (12,)
            next_idx = results['next_idx'][i]                  # Shape: (12,)
            
            # Extract panel coordinates and text
            panel_coords = np.zeros((12, 4), dtype=np.float32)
            text_content = np.zeros(12, dtype=object)
            
            for j, panel in enumerate(page_data['panels']):
                if j < 12:  # Max panels
                    panel_coords[j] = panel['panel_coords']
                    text_content[j] = str(panel.get('text', ''))
            
            # Store in dataset
            ds['panel_embeddings'][page_index] = panel_embeddings
            ds['page_embeddings'][page_index] = page_embedding
            ds['attention_weights'][page_index] = attention_weights
            ds['reading_order'][page_index] = reading_order
            ds['panel_coordinates'][page_index] = panel_coords
            ds['text_content'][page_index] = text_content
            ds['panel_mask'][page_index] = panel_mask
            
            page_idx += 1
    
    print(f"Generated embeddings for {page_idx} pages")
    print(f"Saved to: {output_path}")
    
    return ds

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings and store in Zarr format')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--amazon_json_list', type=str, required=True,
                       help='Path to Amazon JSON list file')
    parser.add_argument('--amazon_image_root', type=str, required=True,
                       help='Root directory for Amazon images')
    parser.add_argument('--calibre_json_list', type=str, default=None,
                       help='Path to CalibreComics JSON list file (optional)')
    parser.add_argument('--calibre_image_root', type=str, default=None,
                       help='Root directory for CalibreComics images (optional)')
    parser.add_argument('--output_dir', type=str, default='embeddings_output',
                       help='Output directory for Zarr files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Attention temperature')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device, args.num_heads, args.temperature)
    
    # Load Amazon data
    print("Loading Amazon data...")
    amazon_data = []
    with open(args.amazon_json_list, 'r', encoding='utf-8') as f:
        for line in f:
            json_path = line.strip()
            if json_path:
                amazon_data.append({'path': json_path})
    
    if args.max_samples:
        amazon_data = amazon_data[:args.max_samples]
    
    # Load CalibreComics data if provided
    calibre_data = None
    if args.calibre_json_list and args.calibre_image_root:
        print("Loading CalibreComics data...")
        calibre_data = []
        with open(args.calibre_json_list, 'r', encoding='utf-8') as f:
            for line in f:
                json_path = line.strip()
                if json_path:
                    calibre_data.append({'path': json_path})
        
        if args.max_samples:
            calibre_data = calibre_data[:args.max_samples]
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(
        args.amazon_json_list, 
        args.amazon_image_root, 
        batch_size=args.batch_size, 
        max_panels=12, 
        num_workers=0,
        max_samples=args.max_samples
    )
    
    # Generate embeddings
    output_path = os.path.join(args.output_dir, 'combined_embeddings.zarr')
    ds = generate_embeddings(model, dataloader, device, output_path, amazon_data, calibre_data)
    
    print("Embedding generation complete!")
    print(f"Dataset saved to: {output_path}")
    print(f"Total pages: {len(ds.page_id)}")
    print(f"Amazon pages: {len(amazon_data)}")
    if calibre_data:
        print(f"CalibreComics pages: {len(calibre_data)}")

if __name__ == "__main__":
    main()
