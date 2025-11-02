#!/usr/bin/env python3
"""
OpenRouter Models Dataset Parser

This script fetches the OpenRouter models API and converts it into a structured dataset
for analysis and comparison of different models.
"""

import requests
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import time

def fetch_openrouter_models():
    """Fetch models from OpenRouter API."""
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return None

def parse_model_data(models_data):
    """Parse the models data into a structured format."""
    if not models_data or 'data' not in models_data:
        print("Invalid API response format")
        return None
    
    models = models_data['data']
    parsed_models = []
    
    for model in models:
        # Extract pricing information
        pricing = model.get('pricing', {})
        
        # Extract architecture information
        architecture = model.get('architecture', {})
        
        # Extract top provider information
        top_provider = model.get('top_provider', {})
        
        # Parse the model data
        parsed_model = {
            'id': model.get('id', ''),
            'canonical_slug': model.get('canonical_slug', ''),
            'name': model.get('name', ''),
            'description': model.get('description', ''),
            'created': model.get('created', 0),
            'context_length': model.get('context_length', 0),
            
            # Architecture details
            'modality': architecture.get('modality', ''),
            'input_modalities': architecture.get('input_modalities', []),
            'output_modalities': architecture.get('output_modalities', []),
            'tokenizer': architecture.get('tokenizer', ''),
            'instruct_type': architecture.get('instruct_type', ''),
            
            # Pricing details
            'prompt_price': float(pricing.get('prompt', 0)),
            'completion_price': float(pricing.get('completion', 0)),
            'request_price': float(pricing.get('request', 0)),
            'image_price': float(pricing.get('image', 0)),
            'web_search_price': float(pricing.get('web_search', 0)),
            'internal_reasoning_price': float(pricing.get('internal_reasoning', 0)),
            
            # Provider details
            'provider_context_length': top_provider.get('context_length', 0),
            'provider_max_completion_tokens': top_provider.get('max_completion_tokens', 0),
            'is_moderated': top_provider.get('is_moderated', False),
            
            # Supported parameters
            'supported_parameters': model.get('supported_parameters', []),
            
            # Per request limits
            'per_request_limits': model.get('per_request_limits', {})
        }
        
        parsed_models.append(parsed_model)
    
    return parsed_models

def create_dataframe(parsed_models):
    """Convert parsed models into a pandas DataFrame."""
    df = pd.DataFrame(parsed_models)
    
    # Convert timestamp to datetime
    if 'created' in df.columns:
        df['created_date'] = pd.to_datetime(df['created'], unit='s')
    
    # Convert lists to strings for CSV compatibility
    for col in ['input_modalities', 'output_modalities', 'supported_parameters']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return df

def analyze_models(df):
    """Perform basic analysis on the models dataset."""
    print("\n=== OpenRouter Models Analysis ===")
    print(f"Total models: {len(df)}")
    
    # Model types by modality
    print(f"\nModel modalities:")
    modality_counts = df['modality'].value_counts()
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count}")
    
    # Free vs paid models
    free_models = df[df['prompt_price'] == 0]
    paid_models = df[df['prompt_price'] > 0]
    print(f"\nFree models: {len(free_models)}")
    print(f"Paid models: {len(paid_models)}")
    
    # Context length analysis
    print(f"\nContext length statistics:")
    print(f"  Average: {df['context_length'].mean():.0f}")
    print(f"  Median: {df['context_length'].median():.0f}")
    print(f"  Max: {df['context_length'].max()}")
    print(f"  Min: {df['context_length'].min()}")
    
    # Price analysis for paid models
    if len(paid_models) > 0:
        print(f"\nPricing analysis (paid models):")
        print(f"  Average prompt price: ${paid_models['prompt_price'].mean():.8f}")
        print(f"  Average completion price: ${paid_models['completion_price'].mean():.8f}")
        print(f"  Most expensive prompt: ${paid_models['prompt_price'].max():.8f}")
        print(f"  Cheapest prompt: ${paid_models['prompt_price'].min():.8f}")
    
    # Vision models
    vision_models = df[df['input_modalities'].str.contains('image', na=False)]
    print(f"\nVision models: {len(vision_models)}")
    
    # Multimodal models
    multimodal_models = df[df['modality'].str.contains('image', na=False)]
    print(f"Multimodal models: {len(multimodal_models)}")
    
    return {
        'total_models': len(df),
        'free_models': len(free_models),
        'paid_models': len(paid_models),
        'vision_models': len(vision_models),
        'multimodal_models': len(multimodal_models)
    }

def save_dataset(df, output_dir, analysis_results):
    """Save the dataset and analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV file
    csv_path = output_path / f"openrouter_models_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to: {csv_path}")
    
    # JSON file (with full structure)
    json_path = output_path / f"openrouter_models_{timestamp}.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"JSON dataset saved to: {json_path}")
    
    # Analysis summary
    summary_path = output_path / f"openrouter_analysis_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Analysis summary saved to: {summary_path}")
    
    # Create filtered datasets
    # Free models
    free_models = df[df['prompt_price'] == 0]
    free_path = output_path / f"openrouter_free_models_{timestamp}.csv"
    free_models.to_csv(free_path, index=False)
    print(f"Free models saved to: {free_path}")
    
    # Vision models
    vision_models = df[df['input_modalities'].str.contains('image', na=False)]
    vision_path = output_path / f"openrouter_vision_models_{timestamp}.csv"
    vision_models.to_csv(vision_path, index=False)
    print(f"Vision models saved to: {vision_path}")
    
    # Paid models sorted by price
    paid_models = df[df['prompt_price'] > 0].sort_values('prompt_price')
    paid_path = output_path / f"openrouter_paid_models_{timestamp}.csv"
    paid_models.to_csv(paid_path, index=False)
    print(f"Paid models (sorted by price) saved to: {paid_path}")

def main():
    parser = argparse.ArgumentParser(description='Parse OpenRouter models API into a dataset')
    parser.add_argument('--output-dir', type=str, default='benchmarks/detections/openrouter/models_dataset',
                       help='Directory to save the dataset')
    parser.add_argument('--show-analysis', action='store_true',
                       help='Show detailed analysis of the models')
    parser.add_argument('--filter-free', action='store_true',
                       help='Show only free models')
    parser.add_argument('--filter-vision', action='store_true',
                       help='Show only vision/multimodal models')
    
    args = parser.parse_args()
    
    print("Fetching OpenRouter models...")
    models_data = fetch_openrouter_models()
    
    if not models_data:
        print("Failed to fetch models data")
        return
    
    print("Parsing models data...")
    parsed_models = parse_model_data(models_data)
    
    if not parsed_models:
        print("Failed to parse models data")
        return
    
    print("Creating dataset...")
    df = create_dataframe(parsed_models)
    
    # Perform analysis
    analysis_results = analyze_models(df)
    
    # Save dataset
    save_dataset(df, args.output_dir, analysis_results)
    
    # Display filtered results if requested
    if args.filter_free:
        print("\n=== Free Models ===")
        free_models = df[df['prompt_price'] == 0]
        for _, model in free_models.iterrows():
            print(f"  {model['name']} ({model['id']})")
    
    if args.filter_vision:
        print("\n=== Vision/Multimodal Models ===")
        vision_models = df[df['input_modalities'].str.contains('image', na=False)]
        for _, model in vision_models.iterrows():
            print(f"  {model['name']} ({model['id']}) - {model['modality']}")
    
    print(f"\nDataset creation complete! Found {len(df)} models.")

if __name__ == "__main__":
    main() 