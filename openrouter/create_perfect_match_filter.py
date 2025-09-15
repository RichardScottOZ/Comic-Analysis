"""
Create perfect match filter for training (1.0 panel ratio)
"""

import csv

def create_perfect_match_filter(csv_path: str, output_path: str):
    """Create filter for images with perfect R-CNN/VLM panel alignment"""
    
    perfect_matches = []
    near_perfect = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio = float(row['panel_count_ratio'])
            if ratio == 1.0:
                perfect_matches.append(row['image_path'])
            elif 0.9 <= ratio <= 1.1:
                near_perfect.append(row['image_path'])
    
    # Save perfect matches
    with open(output_path, 'w') as f:
        f.write('\n'.join(perfect_matches))
    
    # Save near-perfect matches
    near_perfect_path = output_path.replace('.txt', '_near_perfect.txt')
    with open(near_perfect_path, 'w') as f:
        f.write('\n'.join(near_perfect))
    
    print(f"Perfect matches (1.0 ratio): {len(perfect_matches)} images")
    print(f"Near-perfect matches (0.9-1.1): {len(near_perfect)} images")
    print(f"Perfect matches saved to: {output_path}")
    print(f"Near-perfect matches saved to: {near_perfect_path}")
    
    return perfect_matches, near_perfect

if __name__ == "__main__":
    perfect, near_perfect = create_perfect_match_filter(
        'amazon_rcnn_vlm_analysis.csv', 
        'amazon_perfect_matches.txt'
    )

