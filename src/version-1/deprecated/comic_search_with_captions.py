"""
comic_search_with_captions_fixed.py

Modified version that handles separate image and analysis directories:
- Images: E:\CalibreComics_extracted\
- Captions: E:\CalibreComics_analysis\

This script:
1. Creates image embeddings for comic pages (two modes: FAST thumbnail mode, FULL high-res mode)
2. Loads per-page captions from JSON files in the analysis directory
3. Creates FAISS indexes for:
   - Image embeddings
   - Caption embeddings
4. Allows searching:
   - Text → captions
   - Image → images
   - Hybrid: combine both
5. Displays browser preview with thumbnails, captions, and clickable links to full-size pages
"""

import os
import json
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import faiss

try:
    import clip
except ImportError:
    raise ImportError("Please install OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git")

# --- Config ---
IMAGES_DIR = Path("E:/CalibreComics_extracted")  # Original comic pages
ANALYSIS_DIR = Path("E:/CalibreComics_analysis")  # Caption JSON files
EMBED_DIR = Path("pages_embed")  # thumbnails for FAST mode
EMBED_DIR.mkdir(exist_ok=True)
INDEX_DIR = Path("faiss_indexes")
INDEX_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "fast"  # or "full"

ALPHA_IMAGE = 0.5
BETA_CAPTION = 0.5
SEARCH_K = 100

# --- Load CLIP ---
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# --- Utils ---
def scan_image_files():
    """Scan for image files in the images directory."""
    image_files = []
    for comic_folder in IMAGES_DIR.iterdir():
        if comic_folder.is_dir():
            for img_file in comic_folder.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]:
                    image_files.append(img_file)
    return sorted(image_files)

def prepare_thumbnails(files):
    """Create thumbnails for fast mode."""
    mapping = {}
    for img_path in files:
        # Create thumbnail path in embed directory
        comic_name = img_path.parent.name
        thumb_path = EMBED_DIR / f"{comic_name}_{img_path.name}"
        
        if not thumb_path.exists():
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((512, 512))
            img.save(thumb_path)
        mapping[img_path] = thumb_path
    return mapping

def load_caption_for_image(img_path):
    """Load caption from the analysis directory based on image path."""
    # Get comic folder name and image filename
    comic_name = img_path.parent.name
    img_filename = img_path.stem  # filename without extension
    
    # Look for corresponding JSON file in analysis directory
    json_path = ANALYSIS_DIR / comic_name / f"{img_filename}.json"
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract caption from the JSON structure
            if isinstance(data, dict):
                # Try different possible caption fields
                if "overall_summary" in data:
                    return data["overall_summary"]
                elif "caption" in data:
                    return data["caption"]
                elif "summary" in data and isinstance(data["summary"], dict) and "plot" in data["summary"]:
                    return data["summary"]["plot"]
                else:
                    # Return a summary of the available data
                    return str(data)[:500]  # First 500 chars
            elif isinstance(data, str):
                return data
        except Exception as e:
            print(f"Error loading caption for {img_path}: {e}")
            return ""
    else:
        print(f"No caption file found for {img_path}")
        return ""

def embed_images(image_paths):
    """Generate CLIP embeddings for images."""
    all_embeds = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu().numpy())
    return np.vstack(all_embeds)

def embed_texts(texts):
    """Generate CLIP embeddings for text captions."""
    all_embeds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        tokens = clip.tokenize(batch_texts, truncate=True).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        all_embeds.append(feats.cpu().numpy())
    return np.vstack(all_embeds)

def build_index(embeds, dim):
    """Build FAISS index for embeddings."""
    index = faiss.IndexFlatIP(dim)
    index.add(embeds.astype('float32'))
    return index

# --- Build embeddings & indexes ---
print("Scanning for image files...")
files = scan_image_files()
print(f"Found {len(files)} image files")

if MODE == "fast":
    print("Creating thumbnails for fast mode...")
    mapping = prepare_thumbnails(files)
    embed_paths = list(mapping.values())
else:
    embed_paths = files

print("Loading captions...")
captions = [load_caption_for_image(p) for p in files]

# Filter out files with no captions
valid_indices = [i for i, caption in enumerate(captions) if caption.strip()]
files = [files[i] for i in valid_indices]
embed_paths = [embed_paths[i] for i in valid_indices]
captions = [captions[i] for i in valid_indices]

print(f"Processing {len(files)} files with captions")

print("Embedding images...")
img_embeds = embed_images(embed_paths)

print("Embedding captions...")
cap_embeds = embed_texts(captions)

print("Building FAISS indexes...")
img_index = build_index(img_embeds, img_embeds.shape[1])
cap_index = build_index(cap_embeds, cap_embeds.shape[1])

meta = {
    "full_paths": [str(p) for p in files],
    "embed_paths": [str(p) for p in embed_paths],
    "captions": captions
}

# --- Search functions ---
def preview_results(results):
    """Display search results in browser."""
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        html_path = f.name
        f.write('<html><body><h2>Search Results</h2><div style="display:flex;flex-wrap:wrap;gap:16px;">')
        for idx, score in results:
            full_p = Path(meta['full_paths'][idx])
            embed_p = Path(meta['embed_paths'][idx])
            caption = meta['captions'][idx]
            show_p = embed_p if embed_p.exists() else full_p
            f.write(f"<div style='width:320px'>")
            f.write(f"<a href='file://{full_p.resolve()}' target='_blank'>")
            f.write(f"<img src='file://{show_p.resolve()}' style='max-width:300px;border:1px solid #ccc;padding:4px;'></a><br>")
            f.write(f"Score: {score:.3f}<br>")
            f.write(f"<div style='font-size:12px;color:#333'>{caption[:200]}...</div>")
            f.write("</div>")
        f.write('</div></body></html>')
    webbrowser.open(f"file://{html_path}")

def search_text(query, k=10, hybrid=False):
    """Search by text query."""
    tokens = clip.tokenize([query], truncate=True).to(DEVICE)
    with torch.no_grad():
        q = model.encode_text(tokens)
        q /= q.norm(dim=-1, keepdim=True)
    q_np = q.cpu().numpy().astype('float32')

    scores = np.zeros(len(meta['full_paths']), dtype=np.float32)
    D, I = cap_index.search(q_np, SEARCH_K)
    for score, idx in zip(D[0], I[0]):
        scores[idx] += score * BETA_CAPTION

    if hybrid:
        D, I = img_index.search(q_np, SEARCH_K)
        for score, idx in zip(D[0], I[0]):
            scores[idx] += score * ALPHA_IMAGE

    topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    preview_results(topk)

def search_image(query_path, k=10, hybrid=False):
    """Search by image query."""
    with Image.open(query_path) as im:
        x = preprocess(im.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        qimg = model.encode_image(x)
        qimg /= qimg.norm(dim=-1, keepdim=True)
    q_np = qimg.cpu().numpy().astype('float32')

    scores = np.zeros(len(meta['full_paths']), dtype=np.float32)
    D, I = img_index.search(q_np, SEARCH_K)
    for score, idx in zip(D[0], I[0]):
        scores[idx] += score * ALPHA_IMAGE

    if hybrid:
        D, I = cap_index.search(q_np, SEARCH_K)
        for score, idx in zip(D[0], I[0]):
            scores[idx] += score * BETA_CAPTION

    topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    preview_results(topk)

if __name__ == "__main__":
    print("Ready. Try:")
    print(" search_text('spaceship battle', hybrid=True)")
    print(" search_image('some_page.jpg', hybrid=True)") 