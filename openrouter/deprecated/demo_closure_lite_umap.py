"""
Demo script for CLOSURE-Lite model with UMAP visualization
Showcase the model's comic understanding capabilities using UMAP instead of t-SNE
"""

import os
import torch
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from closure_lite_framework import ClosureLite
from closure_lite_dataset import create_dataloader

def load_model(checkpoint_path, device):
    """Load the trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ClosureLite(d=384).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def demo_comic_understanding(model, dataloader, device, output_dir="demo_output"):
    """Demo the model's comic understanding capabilities"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 CLOSURE-Lite Comic Understanding Demo (UMAP)")
    print("=" * 50)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Demo first 3 pages
                break
                
            print(f"\n📖 Analyzing Page {i+1}...")
            
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
            
            # 1. Panel Analysis
            P_flat = model.atom(images, input_ids, attention_mask, comp_feats)
            P = P_flat.view(B, N, -1)
            
            # 2. Reading Order Prediction
            S = model.seq(P, attn_mask=batch['panel_mask'])
            logits_neighbors = model.next_head(S)
            
            # 3. Page-level Understanding
            E_page, attention_weights = model.han.panels_to_page(S, batch['panel_mask'])
            
            # Extract information
            num_panels = batch['panel_mask'][0].sum().item()
            reading_order = []
            for j in range(N):
                if batch['panel_mask'][0, j]:
                    next_idx = batch['next_idx'][0, j].item()
                    if next_idx != -100:
                        reading_order.append((j, next_idx))
            
            # Get page info for display
            page_data = batch['original_page'][0]
            page_name = os.path.basename(page_data.get('page_image_path', 'unknown'))
            json_file = batch['json_file'][0]  # Get the actual JSON file name from the batch
            
            # Display results
            print(f"\n📖 Page {i+1}: {page_name}")
            print(f"  📄 Source: {json_file}")
            print(f"  📊 Detected {num_panels} panels")
            print(f"  🔢 Reading order: {[f'{j}→{next_j}' for j, next_j in reading_order]}")
            print(f"  🧠 Page attention weights: {attention_weights[0].cpu().numpy()[:num_panels]}")
            
            # Panel similarity analysis
            panel_embeddings = P[0][:num_panels].cpu().numpy()
            similarities = []
            for j in range(num_panels):
                for k in range(j+1, num_panels):
                    sim = torch.cosine_similarity(
                        P[0, j:j+1], P[0, k:k+1], dim=1
                    ).item()
                    similarities.append((j, k, sim))
            
            # Find most similar panels
            if similarities:
                most_similar = max(similarities, key=lambda x: x[2])
                print(f"  🔗 Most similar panels: {most_similar[0]} ↔ {most_similar[1]} (similarity: {most_similar[2]:.3f})")
            
            # Get page data for visualization
            page_data = batch['original_page'][0]  # Get original page data from batch
            
            # Create visualization
            create_demo_visualization(
                page_data, 
                P[0], 
                reading_order, 
                attention_weights[0],
                f"{output_dir}/demo_page_{i+1}.png"
            )
            
            print(f"  💾 Visualization saved to demo_page_{i+1}.png")

def create_demo_visualization(page_data, panel_embeddings, reading_order, attention_weights, output_path):
    """Create a comprehensive visualization of the model's analysis using UMAP"""
    
    # Load image
    img_path = page_data['page_image_path']
    
    # Convert Windows paths to WSL paths if needed
    if img_path.startswith('E:/') or img_path.startswith('E:\\'):
        img_path = "/mnt/e" + img_path[2:].replace('\\', '/')
    
    # If it's a relative path, try to find it in the image root
    if not os.path.isabs(img_path):
        # Try different possible locations
        possible_paths = [
            img_path,  # Try as-is first
            os.path.join("/mnt/e/amazon", img_path),  # Try in amazon directory
            os.path.join("/mnt/e/amazon", os.path.basename(img_path)),  # Try just filename
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        else:
            print(f"Warning: Could not find image at any of these paths: {possible_paths}")
            # Create a placeholder image
            img = Image.new('RGB', (800, 1200), color='lightgray')
            W, H = img.size
            # Continue with placeholder image
    
    try:
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        # Create a placeholder image
        img = Image.new('RGB', (800, 1200), color='lightgray')
        W, H = img.size
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main image with panels
    ax1 = plt.subplot(2, 3, (1, 4))
    ax1.imshow(img)
    ax1.set_title('Panel Detection & Reading Order', fontsize=14, fontweight='bold')
    
    # Draw panels with reading order
    colors = plt.cm.Set3(np.linspace(0, 1, len(panel_embeddings)))
    for i, (panel, color) in enumerate(zip(page_data['panels'], colors)):
        x, y, w, h = panel['panel_coords']
        
        # Draw panel box
        rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # Add reading order number
        order_idx = i  # Simplified for demo
        ax1.text(x + 10, y + 30, str(order_idx), fontsize=16, fontweight='bold', 
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
    
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.axis('off')
    
    # Panel similarity heatmap
    ax2 = plt.subplot(2, 3, 2)
    if len(panel_embeddings) > 1:
        similarity_matrix = torch.mm(panel_embeddings, panel_embeddings.t()).cpu().numpy()
        im = ax2.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Panel Similarity Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Panel Index')
        ax2.set_ylabel('Panel Index')
        plt.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'Single Panel', ha='center', va='center', fontsize=14)
        ax2.set_title('Panel Similarity', fontsize=12, fontweight='bold')
    
    # Attention weights
    ax3 = plt.subplot(2, 3, 3)
    valid_attention = attention_weights[:len(panel_embeddings)].cpu().numpy()
    bars = ax3.bar(range(len(valid_attention)), valid_attention, color=colors[:len(valid_attention)])
    ax3.set_title('Panel Attention Weights', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Panel Index')
    ax3.set_ylabel('Attention Weight')
    ax3.set_xticks(range(len(valid_attention)))
    
    # Panel embeddings UMAP (if enough panels)
    ax4 = plt.subplot(2, 3, 5)
    
    # Filter to only actual panels (not padding)
    num_actual_panels = len(page_data['panels'])
    actual_panel_embeddings = panel_embeddings[:num_actual_panels]
    
    if num_actual_panels > 1:  # UMAP can work with fewer samples than t-SNE
        try:
            import umap
            # UMAP parameters for better visualization with small datasets
            reducer = umap.UMAP(
                n_components=2, 
                random_state=42,
                n_neighbors=min(2, num_actual_panels - 1),  # Use fewer neighbors for small datasets
                min_dist=0.3,  # Increase min_dist for small datasets
                spread=1.0,
                metric='cosine'  # Use cosine distance which works better for embeddings
            )
            embeddings_2d = reducer.fit_transform(actual_panel_embeddings.cpu().numpy())
            
            scatter = ax4.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=range(num_actual_panels), cmap='Set3', s=100)
            ax4.set_title('Panel Embeddings (UMAP)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('UMAP 1')
            ax4.set_ylabel('UMAP 2')
            
            # Add panel labels
            for i, (x, y) in enumerate(embeddings_2d):
                ax4.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')
                
        except ImportError:
            ax4.text(0.5, 0.5, 'UMAP not installed.\nInstall with:\npip install umap-learn', 
                    ha='center', va='center', fontsize=10)
            ax4.set_title('Panel Embeddings (UMAP)', fontsize=12, fontweight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, f'UMAP failed:\n{str(e)[:50]}...', ha='center', va='center', fontsize=10)
            ax4.set_title('Panel Embeddings (UMAP)', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, f'Need 2+ panels\nfor UMAP\n(has {num_actual_panels})', ha='center', va='center', fontsize=12)
        ax4.set_title('Panel Embeddings (UMAP)', fontsize=12, fontweight='bold')
    
    # Reading order flow
    ax5 = plt.subplot(2, 3, 6)
    if len(reading_order) > 1:
        # Create reading order flow diagram
        order_nodes = list(range(len(panel_embeddings)))
        order_edges = [(j, next_j) for j, next_j in reading_order if next_j < len(panel_embeddings)]
        
        # Simple flow visualization
        y_pos = np.linspace(0, 1, len(order_nodes))
        ax5.scatter([0] * len(order_nodes), y_pos, c=colors, s=200, alpha=0.7)
        
        for i, y in enumerate(y_pos):
            ax5.text(-0.1, y, f'P{i}', ha='right', va='center', fontweight='bold')
        
        # Draw arrows for reading order
        for j, next_j in order_edges:
            if j < len(y_pos) and next_j < len(y_pos):
                ax5.annotate('', xy=(0.1, y_pos[next_j]), xytext=(0, y_pos[j]),
                           arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax5.set_xlim(-0.3, 0.3)
        ax5.set_ylim(-0.1, 1.1)
        ax5.set_title('Reading Order Flow', fontsize=12, fontweight='bold')
        ax5.axis('off')
    else:
        ax5.text(0.5, 0.5, 'Single panel\nor no order', ha='center', va='center', fontsize=12)
        ax5.set_title('Reading Order', fontsize=12, fontweight='bold')
        ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Comprehensive analysis saved to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CLOSURE-Lite Demo with UMAP')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--json_dir', type=str, required=True,
                       help='Directory containing DataSpec JSON files')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Root directory for comic images')
    parser.add_argument('--output_dir', type=str, default='demo_output_umap',
                       help='Directory to save demo outputs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create dataset
    dataloader = create_dataloader(
        args.json_dir, 
        args.image_root, 
        batch_size=1, 
        max_panels=12, 
        num_workers=0,
        max_samples=100  # Sample only 100 pages for demo
    )
    
    # Run demo
    demo_comic_understanding(model, dataloader, device, args.output_dir)
    
    print(f"\n🎉 Demo completed! Check {args.output_dir} for visualizations.")

if __name__ == "__main__":
    main()
