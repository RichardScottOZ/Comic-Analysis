#!/usr/bin/env python3
"""
Compare multiple Closure-Lite models (Simple/Full variants) on the same set of pages.
Outputs the same diagnostics as the original simple demo (image with panel boxes and order, similarity heatmap, attention weights, and 2D embedding), but repeated per-model so you can compare variants on the exact same pages.

Usage example (Windows PowerShell):
  python benchmarks/detections/openrouter/demo_compare_models.py \
    --json_list_file path\to\calibre_perfect_list.txt \
    --image_root E:\\calibre \
    --models "name=base_simple;ckpt=path\\to\\base\\best_checkpoint.pth;variant=simple" \
             "name=denoise;ckpt=path\\to\\denoise\\best_checkpoint.pth;variant=simple" \
             "name=context;ckpt=path\\to\\context\\best_checkpoint.pth;variant=simple" \
    --max_pages 16 --seed 42 --out_dir ./demo_out

This script keeps computation light and focuses on extracting analyzable outputs via model.analyze(...).
"""

import os
import re
import json
import argparse
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
try:
    from sklearn.manifold import TSNE  # type: ignore
except Exception:
    TSNE = None
try:
    import umap  # type: ignore
except Exception:
    umap = None

from closure_lite_dataset import create_dataloader_from_list
from closure_lite_framework import ClosureLite as ClosureLiteFull
from closure_lite_simple_framework import ClosureLiteSimple


def parse_model_arg(s: str):
    # Format: name=foo;ckpt=path;variant=simple|full;extra=mpm_denoise:1,mpm_context:0
    parts = [p for p in re.split(r";", s) if p]
    out = {}
    for p in parts:
        if "=" in p:
            k,v = p.split("=",1)
            out[k.strip()] = v.strip()
    if 'name' not in out or 'ckpt' not in out or 'variant' not in out:
        raise ValueError(f"Invalid --models entry: {s}")
    return out


def load_model(entry: dict, device: torch.device):
    variant = entry['variant']
    ckpt = entry['ckpt']
    if variant == 'simple':
        # We load in strict=False to allow for optional heads
        model = ClosureLiteSimple().to(device)
    elif variant == 'full':
        model = ClosureLiteFull().to(device)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    mstate = state.get('model_state_dict', state)
    missing, unexpected = model.load_state_dict(mstate, strict=False)
    print(f"Loaded {entry['name']} ({variant}). Missing: {len(missing)} Unexpected: {len(unexpected)}")
    model.eval()
    return model


def batch_analyze(model, batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    if hasattr(model, 'analyze'):
        return model.analyze(batch)
    # Full model: emulate analyze by using sequence output
    with torch.no_grad():
        B,N,_,_,_ = batch['images'].shape
        images = batch['images'].flatten(0,1)
        input_ids = batch['input_ids'].flatten(0,1)
        attention_mask = batch['attention_mask'].flatten(0,1)
        comp_feats = batch['comp_feats'].flatten(0,1)
        P_flat = model.atom(images, input_ids, attention_mask, comp_feats)
        P = P_flat.view(B, N, -1)
        S = model.seq(P, attn_mask=batch['panel_mask'])
        E_page, attn = model.han.panels_to_page(S, batch['panel_mask'])
        return {'P': P, 'E_page': E_page, 'attention': attn, 'panel_mask': batch['panel_mask']}


def _open_image(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        try:
            # Normalize EXIF orientation
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img.convert('RGB')
    except Exception as e:
        # Fallback: placeholder so figure still renders
        import warnings
        warnings.warn(f"Failed to open image '{path}': {e}")
        return Image.new('RGB', (800, 1200), color='lightgray')


def _resolve_used_panels(page_data: dict, boxes_norm, image_path: str):
    """Return (used_boxes_px:list[[x,y,w,h]], source:str, n_json:int, n_batch_boxes:int).
    If page_data contains panels, use those (absolute px). Otherwise, if boxes_norm
    (normalized [x,y,w,h]) is provided, convert to px using the image size.
    """
    img = _open_image(image_path)
    W, H = img.size
    panels = page_data.get('panels', []) or []
    if isinstance(panels, list) and len(panels) > 0:
        out = []
        for p in panels:
            try:
                if 'panel_coords' in p and isinstance(p.get('panel_coords'), (list, tuple)) and len(p['panel_coords']) == 4:
                    # DataSpec convention: [x, y, w, h] in pixels
                    x, y, w, h = p['panel_coords']
                    out.append([int(x), int(y), int(w), int(h)])
                elif 'bbox' in p and isinstance(p.get('bbox'), (list, tuple)) and len(p['bbox']) == 4:
                    # Heuristics for generic bbox: could be [x,y,w,h], [x1,y1,x2,y2], or normalized
                    b0, b1, b2, b3 = [float(v) for v in p['bbox']]
                    # Normalized [0..1] → scale
                    if max(b0, b1, b2, b3) <= 1.1:
                        x, y, w, h = b0 * W, b1 * H, b2 * W, b3 * H
                    else:
                        # Decide between [x,y,w,h] vs [x1,y1,x2,y2]
                        looks_x2y2 = (b2 > b0 and b3 > b1 and b2 <= W and b3 <= H)
                        if looks_x2y2:
                            x, y, w, h = b0, b1, max(1.0, b2 - b0), max(1.0, b3 - b1)
                        else:
                            x, y, w, h = b0, b1, b2, b3
                    out.append([int(x), int(y), int(w), int(h)])
            except Exception:
                continue
        return out, 'json_panels', len(panels), int((boxes_norm.sum(axis=-1) > 0).sum()) if isinstance(boxes_norm, np.ndarray) else 0
    # Fallback to normalized boxes if provided
    if boxes_norm is not None:
        try:
            bn = np.asarray(boxes_norm)
            # Filter out zero rows
            keep = (bn.sum(axis=-1) > 0)
            bn = bn[keep]
            out = []
            for b in bn:
                x, y, w, h = float(b[0]) * W, float(b[1]) * H, float(b[2]) * W, float(b[3]) * H
                out.append([int(x), int(y), int(w), int(h)])
            return out, 'batch_boxes', 0, int(len(out))
        except Exception:
            pass
    return [], 'none', 0, 0


def _plot_page_like_simple(fig, axes, page_img_path, page_data, attn, P, title, boxes_norm=None):
    # Axes layout matches the original simple demo: image + similarity + attention + 2D emb + flow
    ax_img, ax_sim, ax_attn, ax_umap, ax_flow = axes

    # Load and show image
    img = _open_image(page_img_path)
    W, H = img.size
    ax_img.imshow(img)
    ax_img.set_title('Panel Detection & Reading Order', fontsize=12, fontweight='bold')
    ax_img.axis('off')

    # Draw panels with simple index labels (0..N-1)
    # Use the same panel source/coords as the CSV by resolving here too
    used_boxes_px, source, n_json, n_bb = _resolve_used_panels(page_data, np.asarray(boxes_norm) if boxes_norm is not None else None, page_img_path)
    panels = [{'panel_coords': b} for b in used_boxes_px]
    colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(panels))))
    for i, (panel, color) in enumerate(zip(panels, colors)):
        try:
            x, y, w, h = panel.get('panel_coords', panel.get('bbox', [0, 0, 1, 1]))
        except Exception:
            x, y, w, h = 0, 0, 1, 1
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)
        ax_img.text(x + 8, y + 24, str(i), fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)

    # Determine actual panels count from JSON for consistent slicing
    num_actual = len(panels)
    Pm = P[0, :num_actual].detach().cpu().numpy() if num_actual > 0 else np.zeros((0, P.shape[-1]))

    # Similarity heatmap (cosine)
    if Pm.shape[0] > 1:
        Pn = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(Pn @ Pn.T, 0.0, 1.0)
        im = ax_sim.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        ax_sim.set_title('Panel Similarity Matrix (Cosine)', fontsize=11, fontweight='bold')
        ax_sim.set_xlabel('Panel Index')
        ax_sim.set_ylabel('Panel Index')
        fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
    else:
        ax_sim.text(0.5, 0.5, 'Single Panel', ha='center', va='center')
        ax_sim.set_title('Panel Similarity', fontsize=11, fontweight='bold')

    # Attention weights (slice to actual panels)
    a = attn[0, :num_actual].detach().cpu().numpy() if attn is not None else np.zeros((num_actual,))
    ax_attn.bar(np.arange(len(a)), a, color=colors[:len(a)])
    ax_attn.set_title('Panel Attention Weights', fontsize=11, fontweight='bold')
    ax_attn.set_xlabel('Panel Index')
    ax_attn.set_ylabel('Attention Weight')

    # 2D embedding: UMAP if available; else t-SNE; fallback to centered PCA (SVD)
    try:
        if Pm.shape[0] >= 2:
            if umap is not None:
                reducer = umap.UMAP(n_components=2, random_state=42,
                                     n_neighbors=min(5, len(Pm)-1), min_dist=0.3, metric='cosine')
                pts = reducer.fit_transform(Pm)
                ttl = 'Panel Embeddings (UMAP)'
            elif TSNE is not None:
                pts = TSNE(n_components=2, perplexity=min(5, len(Pm)-1), init='random', random_state=0).fit_transform(Pm)
                ttl = 'Panel Embeddings (t-SNE)'
            else:
                X = Pm - Pm.mean(axis=0, keepdims=True)
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                pts = U[:, :2] * S[:2]
                ttl = 'Panel Embeddings (PCA)'
            sc = ax_umap.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(Pm)), cmap='Set3', s=60)
            ax_umap.set_title(ttl, fontsize=11, fontweight='bold')
            for i, (x, y) in enumerate(pts):
                ax_umap.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        else:
            ax_umap.text(0.5, 0.5, f'Need 2+ panels (has {Pm.shape[0]})', ha='center', va='center')
            ax_umap.set_title('Panel Embeddings', fontsize=11, fontweight='bold')
    except Exception as e:
        ax_umap.text(0.5, 0.5, f'proj err: {str(e)[:40]}', ha='center', va='center')
        ax_umap.set_title('Panel Embeddings', fontsize=11, fontweight='bold')

    # Reading order flow (derive from simple next_idx style if present)
    ax_flow.set_title('Reading Order Flow', fontsize=11, fontweight='bold')
    order_nodes = list(range(num_actual))
    y_pos = np.linspace(0, 1, max(1, len(order_nodes)))
    ax_flow.scatter([0] * len(order_nodes), y_pos, c=colors[:len(order_nodes)], s=120, alpha=0.8)
    for i, y in enumerate(y_pos):
        ax_flow.text(-0.08, y, f'P{i}', ha='right', va='center', fontweight='bold')
    # Try to draw arrows using simple linear order 0..N-1
    for j in range(len(order_nodes) - 1):
        ax_flow.annotate('', xy=(0.1, y_pos[j + 1]), xytext=(0, y_pos[j]),
                         arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax_flow.set_xlim(-0.2, 0.3)
    ax_flow.set_ylim(-0.1, 1.1)
    ax_flow.axis('off')

    # Add a compact super-title with model/page info
    fig.suptitle(title, fontsize=11, y=0.98)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_list_file', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--models', nargs='+', required=True, help='One or more entries: name=ID;ckpt=PATH;variant=simple|full')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_pages', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='./demo_compare_out')
    ap.add_argument('--report_csv', default=None, help='Optional path to write a CSV summary; defaults to <out_dir>/report.csv')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    dl = create_dataloader_from_list(
        args.json_list_file,
        args.image_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_pages,
        seed=args.seed,
        dedupe=True,
        sample_without_replacement=True,
    )

    # Models
    entries = [parse_model_arg(s) for s in args.models]
    models = [load_model(e, device) for e in entries]

    # Prepare CSV report
    report_path = args.report_csv or str(Path(args.out_dir) / 'report.csv')
    first_row = True
    os.makedirs(args.out_dir, exist_ok=True)
    csv_f = open(report_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    header = [
        'batch', 'model', 'variant',
        'page_name', 'image_path', 'comic_dir',
        'json_file', 'json_path',
        'panel_source', 'n_json_panels', 'n_batch_boxes', 'num_panels_used', 'used_boxes_px',
        'max_similarity', 'max_sim_pair', 'mean_similarity',
        'attention_max', 'attention_max_idx', 'attention_entropy', 'attention_vector'
    ]
    writer.writerow(header)

    # Iterate pages and render per-model comparisons (separate PNG per model, per page)
    for bidx, batch in enumerate(dl):
        for midx, (entry, model) in enumerate(zip(entries, models)):
            res = batch_analyze(model, dict(batch), device)
            # Prepare figure layout (image + 4 diagnostics), matching the simple demo style
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2,3)
            ax_img = fig.add_subplot(gs[:,0])
            ax_sim = fig.add_subplot(gs[0,1])
            ax_attn = fig.add_subplot(gs[0,2])
            ax_umap = fig.add_subplot(gs[1,1])
            ax_flow = fig.add_subplot(gs[1,2])
            page_data = batch.get('original_page', [{}])[0]
            img_path = batch.get('image_path', [None])[0]
            page_name = os.path.basename(page_data.get('page_image_path') or img_path or f'page_{bidx}.png')
            # Render page like the original simple demo
            _plot_page_like_simple(
                fig,
                (ax_img, ax_sim, ax_attn, ax_umap, ax_flow),
                img_path,
                page_data,
                res.get('attention'),
                res.get('P'),
                f"{entry['name']} · {page_name} · batch{bidx}",
                boxes_norm=(batch.get('boxes')[0].detach().cpu().numpy() if isinstance(batch.get('boxes'), torch.Tensor) else None)
            )
            out = Path(args.out_dir) / f"batch{bidx}_{entry['name']}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # --- CSV metrics (mirror original simple demo prints) ---
            # Resolve the exact boxes used for this figure
            bx = batch.get('boxes')
            bx_np = bx[0].detach().cpu().numpy() if isinstance(bx, torch.Tensor) else None
            used_boxes_px, src, n_json, n_bb = _resolve_used_panels(page_data, bx_np, img_path)
            num_actual = len(used_boxes_px)

            P = res.get('P')
            a = res.get('attention')
            Pm = P[0, :num_actual].detach().cpu().numpy() if (P is not None and num_actual > 0) else np.zeros((0, 1))
            av = a[0, :num_actual].detach().cpu().numpy() if (a is not None and num_actual > 0) else np.zeros((0,))

            # Pairwise cosine sims summary
            max_sim = 0.0
            max_pair = (-1, -1)
            mean_sim = 0.0
            if Pm.shape[0] > 1:
                Pn = Pm / (np.linalg.norm(Pm, axis=1, keepdims=True) + 1e-8)
                sim = np.clip(Pn @ Pn.T, 0.0, 1.0)
                iu = np.triu_indices(sim.shape[0], k=1)
                flat = sim[iu]
                if flat.size > 0:
                    max_idx = int(flat.argmax())
                    max_sim = float(flat[max_idx])
                    # Map back to pair
                    pairs = list(zip(iu[0], iu[1]))
                    max_pair = pairs[max_idx]
                    mean_sim = float(flat.mean())

            # Attention summaries
            if av.size > 0:
                att_max = float(av.max())
                att_max_idx = int(av.argmax())
                # entropy with small eps
                p = av / (av.sum() + 1e-8)
                att_entropy = float(-(p * np.log(p + 1e-12)).sum())
                att_vec_str = '|'.join(f"{x:.6f}" for x in av.tolist())
            else:
                att_max = 0.0
                att_max_idx = -1
                att_entropy = 0.0
                att_vec_str = ''

            comic_dir = os.path.basename(os.path.dirname(img_path)) if img_path else ''
            used_boxes_str = '|'.join(','.join(str(int(v)) for v in b) for b in used_boxes_px)
            row = [
                bidx,
                entry.get('name',''),
                entry.get('variant',''),
                page_name,
                img_path,
                comic_dir,
                batch.get('json_file',[None])[0],
                batch.get('json_path',[None])[0],
                src,
                n_json,
                n_bb,
                num_actual,
                used_boxes_str,
                f"{max_sim:.6f}",
                f"({max_pair[0]},{max_pair[1]})",
                f"{mean_sim:.6f}",
                f"{att_max:.6f}",
                att_max_idx,
                f"{att_entropy:.6f}",
                att_vec_str,
            ]
            writer.writerow(row)
        if (bidx+1) * args.batch_size >= args.max_pages:
            break

    csv_f.close()
    print(f"Saved figures under: {args.out_dir}")
    print(f"CSV report: {report_path}")


if __name__ == '__main__':
    main()
