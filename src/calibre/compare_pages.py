
import argparse
import json
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as T
from transformers import AutoTokenizer

# It's assumed that this script is run from the root of the CoMix project,
# so we can add the path to the openrouter directory to the python path.
import sys
sys.path.append('./benchmarks/detections/openrouter')

from closure_lite_simple_framework import ClosureLiteSimple

def load_model(ckpt_path, device):
    """Loads the ClosureLiteSimple model from a checkpoint."""
    print(f"Loading model from {ckpt_path}...")
    model = ClosureLiteSimple().to(device)
    state = torch.load(ckpt_path, map_location=device)
    mstate = state.get('model_state_dict', state)
    model.load_state_dict(mstate, strict=False)
    model.eval()
    return model

def load_page_data(json_path):
    """Loads page data from a JSON manifest file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_image_path(page_data):
    """Extracts the image path from the page data."""
    return page_data.get('page_image_path')

def draw_bounding_boxes_and_text(image, page_data):
    """Draws panel bounding boxes and text on the image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    panels = page_data.get('panels', [])
    for i, panel in enumerate(panels):
        coords = panel.get('panel_coords')
        if coords:
            x, y, w, h = coords
            draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
            draw.text((x + 5, y + 5), f"Panel {i}", fill='red', font=font)

        text_info = panel.get('text', {})
        text_to_draw = []
        for text_type, text_list in text_info.items():
            if text_list:
                text_to_draw.append(f"{text_type.upper()}:")
                # Handle cases where text_list contains strings or lists of strings
                for item in text_list:
                    if isinstance(item, str):
                        text_to_draw.append(f"- {item}")
                    elif isinstance(item, list):
                        text_to_draw.extend([f"- {sub_item}" for sub_item in item])

        if text_to_draw:
            draw.text((x + 5, y + 30), "\n".join(text_to_draw), fill='blue', font=font)
            
    return image

def comp_features_for_panel(panel, page_w, page_h):
    x,y,w,h = panel['panel_coords']
    aspect = (w+1e-6)/(h+1e-6)
    size_ratio = (w*h) / (page_w*page_h + 1e-6)
    chars = panel.get('character_coords', []) or []
    char_count = len(chars)
    if char_count > 0:
        ratios = [ (cw*ch)/(w*h+1e-6) for (cx,cy,cw,ch) in chars ]
        shot_mean = float(np.mean(ratios))
        shot_max = float(np.max(ratios))
    else:
        shot_mean = 0.0; shot_max = 0.0
    if char_count > 0:
        centers = [ ((cx+cw/2 - x)/(w+1e-6), (cy+ch/2 - y)/(h+1e-6)) for (cx,cy,cw,ch) in chars ]
        mean_cx = float(np.mean([c[0] for c in centers]))
        mean_cy = float(np.mean([c[1] for c in centers]))
    else:
        mean_cx = 0.5; mean_cy = 0.5
    return np.array([aspect, size_ratio, char_count, shot_mean, shot_max, mean_cx, mean_cy], dtype=np.float32)

def prepare_page_data_for_model(page_data, image_path, tokenizer, device, max_panels=12):
    """Prepares a single page's data into a model-ready batch."""
    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    panels = page_data.get('panels', [])
    
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    crops, texts, comps = [], [], []
    for p in panels[:max_panels]:
        x, y, w, h = p['panel_coords']
        crop = img.crop((x, y, x + w, y + h))
        crops.append(tf(crop))

        tdict = p.get('text', {}) or {}
        parts = []
        for k in ('dialogue', 'narration', 'sfx'):
            vals = tdict.get(k) or []
            for v in vals:
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
                elif isinstance(v, list):
                    parts.extend([str(item).strip() for item in v if str(item).strip()])
        text = ' | '.join(parts) if parts else ''
        texts.append(text)
        
        comps.append(comp_features_for_panel(p, W, H))

    N = len(crops)
    padN = max_panels - N
    if padN > 0:
        pad_img = torch.zeros(3, 224, 224)
        crops += [pad_img] * padN
        texts += [''] * padN
        comps += [np.zeros(7, dtype=np.float32)] * padN

    tok = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    
    batch = {
        'images': torch.stack(crops).unsqueeze(0).to(device),
        'input_ids': tok['input_ids'].unsqueeze(0).to(device),
        'attention_mask': tok['attention_mask'].unsqueeze(0).to(device),
        'comp_feats': torch.tensor(np.stack(comps), dtype=torch.float32).unsqueeze(0).to(device),
        'panel_mask': torch.zeros(1, max_panels, dtype=torch.bool).index_fill_(1, torch.arange(N), True).to(device),
    }
    return batch

def get_embedding_stats(model, batch):
    """Analyzes a page to get embedding statistics."""
    return model.analyze(batch)

def main():
    parser = argparse.ArgumentParser(description="Visualize panel detections and compare embeddings for one or two comic pages.")
    parser.add_argument('--json_paths', nargs='+', required=True, help='Path(s) to one or two JSON manifest files.')
    parser.add_argument('--model_ckpt', required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--output_dir', default='./compare_out', help='Directory to save the output images.')
    args = parser.parse_args()

    if len(args.json_paths) > 2:
        print("Please provide at most two JSON paths.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    model = load_model(args.model_ckpt, device)

    os.makedirs(args.output_dir, exist_ok=True)

    page_embeddings = []

    for i, json_path in enumerate(args.json_paths):
        print(f"--- Processing page {i+1}: {json_path} ---")
        
        page_data = load_page_data(json_path)
        image_path = get_image_path(page_data)
        
        if not image_path or not os.path.exists(image_path):
            print(f"Image path not found or invalid in {json_path}")
            continue
            
        image = Image.open(image_path).convert('RGB')

        annotated_image = draw_bounding_boxes_and_text(image.copy(), page_data)
        output_image_path = os.path.join(args.output_dir, f"page_{i+1}_annotated.png")
        annotated_image.save(output_image_path)
        print(f"Saved annotated image to {output_image_path}")

        if model:
            batch = prepare_page_data_for_model(page_data, image_path, tokenizer, device)
            stats = get_embedding_stats(model, batch)
            page_embeddings.append(stats['E_page'].squeeze().cpu().numpy())
            
            print(f"Page {i+1} Analysis:")
            print(f"  - Page Embedding Norm: {torch.linalg.norm(stats['E_page']).item():.4f}")
            print(f"  - Panel Embeddings Shape: {stats['P'].shape}")
            print(f"  - Attention Weights: {stats['attention'].squeeze().cpu().numpy()}")
        else:
            print("Model not loaded. Skipping analysis.")

    if len(page_embeddings) == 2:
        sim = np.dot(page_embeddings[0], page_embeddings[1]) / (np.linalg.norm(page_embeddings[0]) * np.linalg.norm(page_embeddings[1]))
        print(f"\n--- Comparison ---")
        print(f"Cosine Similarity between page embeddings: {sim:.4f}")

if __name__ == '__main__':
    main()
