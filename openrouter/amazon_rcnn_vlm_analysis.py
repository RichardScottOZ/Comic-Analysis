"""
Comprehensive analysis of R-CNN vs VLM alignment
Creates a CSV filter for good training samples
"""

import json
import os
import re
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

def load_coco_data(coco_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load COCO detection data"""
    print("Loading COCO data...")
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    # Categories
    cat_map = {}
    for c in coco.get('categories', []):
        cat_map[c['id']] = c['name']
    
    # Group annotations per image_id
    anns_by_img = defaultdict(list)
    for a in coco.get('annotations', []):
        anns_by_img[a['image_id']].append(a)
    
    # Build image info map
    imginfo = {}
    if isinstance(coco.get('images'), list) and coco['images'] and isinstance(coco['images'][0], dict):
        for im in coco['images']:
            iid = im['id']
            p = im.get('file_name', iid)
            imginfo[iid] = {
                'path': p.replace('\\', '/'),
                'width': im.get('width'),
                'height': im.get('height')
            }
    
    return cat_map, anns_by_img, imginfo

def load_vlm_data(vlm_pages_dir: str) -> Dict:
    """Load VLM data by image path"""
    print("Loading VLM data...")
    vlm_data = {}
    
    if not os.path.exists(vlm_pages_dir):
        print(f"VLM directory not found: {vlm_pages_dir}")
        return vlm_data
    
    for f in tqdm(os.listdir(vlm_pages_dir), desc="Loading VLM files"):
        if f.endswith('.json'):
            try:
                with open(os.path.join(vlm_pages_dir, f), 'r', encoding='utf-8') as vf:
                    vlm = json.load(vf)
                    # Store by filename stem for matching
                    stem = os.path.splitext(f)[0]
                    vlm_data[stem] = vlm
            except Exception as e:
                print(f"Error loading VLM file {f}: {e}")
    
    return vlm_data

def analyze_single_image(img_id, anns, cat_map, imginfo, vlm_data, 
                        panel_thr=0.75, panel_nms=0.25, text_thr=0.5, char_thr=0.6, balloon_thr=0.5):
    """Analyze a single image for R-CNN vs VLM alignment"""
    
    # Get image info
    if img_id in imginfo:
        path = imginfo[img_id]['path']
        W, H = imginfo[img_id]['width'], imginfo[img_id]['height']
    else:
        path = str(img_id)
        W, H = 1000, 1000  # Default fallback
    
    # Extract filename stem for VLM matching
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'\.(png|jpg|jpeg)$', '', stem, flags=re.IGNORECASE)
    
    # Find matching VLM data
    vlm_match = None
    vlm_file = None
    for vlm_stem, vlm_data_item in vlm_data.items():
        if stem in vlm_stem or vlm_stem in stem:
            vlm_match = vlm_data_item
            vlm_file = vlm_stem + '.json'
            break
    
    # Collect R-CNN detections
    panels, panel_scores = [], []
    texts, text_scores = [], []
    balloons, balloon_scores = [], []
    chars, char_scores = [], []
    faces, face_scores = [], []
    
    for a in anns:
        cid = a['category_id']
        b = [float(x) for x in a['bbox']]
        s = float(a.get('score', 1.0))
        name = cat_map.get(cid, str(cid))
        
        if name == 'panel' and s >= panel_thr:
            panels.append(b); panel_scores.append(s)
        elif name == 'balloon' and s >= balloon_thr:
            balloons.append(b); balloon_scores.append(s)
        elif name == 'text' and s >= text_thr:
            texts.append(b); text_scores.append(s)
        elif name == 'onomatopoeia' and s >= text_thr:
            texts.append(b); text_scores.append(s)
        elif name == 'character' and s >= char_thr:
            chars.append(b); char_scores.append(s)
        elif name == 'face' and s >= char_thr:
            faces.append(b); face_scores.append(s)
    
    # Apply NMS to panels
    if len(panels) > 1:
        from collections import defaultdict
        import numpy as np
        
        def iou_xywh(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ax2, ay2 = ax+aw, ay+ah
            bx2, by2 = bx+bw, by+bh
            inter_w = max(0, min(ax2, bx2) - max(ax, bx))
            inter_h = max(0, min(ay2, by2) - max(ay, by))
            inter = inter_w * inter_h
            ua = aw*ah + bw*bh - inter + 1e-6
            return inter / ua
        
        def nms(boxes, scores, iou_thr):
            idxs = np.argsort(scores)[::-1]
            keep = []
            while len(idxs) > 0:
                i = idxs[0]
                keep.append(int(i))
                if len(idxs) == 1: break
                rest = idxs[1:]
                ious = np.array([iou_xywh(boxes[i], boxes[j]) for j in rest])
                idxs = rest[ious < iou_thr]
            return keep
        
        keep = nms(panels, panel_scores, iou_thr=panel_nms)
        panels = [panels[i] for i in keep]
        panel_scores = [panel_scores[i] for i in keep]
    
    # Filter tiny panels
    clean_panels = []
    for b in panels:
        x, y, w, h = b
        if w < 32 or h < 32: continue
        if w * h < 0.01 * (W * H): continue
        clean_panels.append(b)
    panels = clean_panels
    
    # Analyze VLM data
    vlm_panels = []
    vlm_panels_with_text = 0
    vlm_total_dialogue = 0
    vlm_total_narration = 0
    
    if vlm_match:
        vlm_panels = vlm_match.get('panels', [])
        for p in vlm_panels:
            has_text = False
            # Check dialogue
            for s in p.get('speakers', []):
                if 'dialogue' in s and s['dialogue']:
                    vlm_total_dialogue += 1
                    has_text = True
            # Check narration
            cap = p.get('caption')
            if cap:
                vlm_total_narration += 1
                has_text = True
            if has_text:
                vlm_panels_with_text += 1
    
    # Calculate alignment metrics
    rcnn_panel_count = len(panels)
    vlm_panel_count = len(vlm_panels)
    vlm_text_panel_count = vlm_panels_with_text
    
    # Panel count ratio
    panel_count_ratio = vlm_panel_count / rcnn_panel_count if rcnn_panel_count > 0 else 0
    
    # Text coverage ratio
    text_coverage_ratio = vlm_text_panel_count / rcnn_panel_count if rcnn_panel_count > 0 else 0
    
    # VLM text density
    vlm_text_density = (vlm_total_dialogue + vlm_total_narration) / vlm_panel_count if vlm_panel_count > 0 else 0
    
    # Quality flags
    has_vlm_data = vlm_match is not None
    has_rcnn_panels = rcnn_panel_count > 0
    has_vlm_panels = vlm_panel_count > 0
    has_text_content = vlm_text_panel_count > 0
    
    # Alignment quality
    panel_alignment_good = 0.7 <= panel_count_ratio <= 1.3 if rcnn_panel_count > 0 else False
    text_coverage_good = text_coverage_ratio >= 0.3 if rcnn_panel_count > 0 else False
    vlm_quality_good = vlm_text_density >= 1.0 if vlm_panel_count > 0 else False
    
    # Overall quality score
    quality_score = 0
    if has_rcnn_panels: quality_score += 1
    if has_vlm_data: quality_score += 1
    if panel_alignment_good: quality_score += 1
    if text_coverage_good: quality_score += 1
    if vlm_quality_good: quality_score += 1
    
    return {
        'image_id': img_id,
        'image_path': path,
        'image_size': f"{W}x{H}",
        'vlm_file': vlm_file,
        'has_vlm_data': has_vlm_data,
        'rcnn_panel_count': rcnn_panel_count,
        'vlm_panel_count': vlm_panel_count,
        'vlm_text_panel_count': vlm_text_panel_count,
        'panel_count_ratio': panel_count_ratio,
        'text_coverage_ratio': text_coverage_ratio,
        'vlm_text_density': vlm_text_density,
        'vlm_total_dialogue': vlm_total_dialogue,
        'vlm_total_narration': vlm_total_narration,
        'rcnn_text_regions': len(texts),
        'rcnn_balloon_regions': len(balloons),
        'rcnn_character_regions': len(chars),
        'rcnn_face_regions': len(faces),
        'panel_alignment_good': panel_alignment_good,
        'text_coverage_good': text_coverage_good,
        'vlm_quality_good': vlm_quality_good,
        'quality_score': quality_score,
        'recommend_training': quality_score >= 4
    }

def analyze_alignment(coco_path: str, vlm_pages_dir: str, output_csv: str, 
                     limit: int = None, panel_thr: float = 0.75, panel_nms: float = 0.25):
    """Analyze R-CNN vs VLM alignment and create CSV filter"""
    
    print("🔍 Analyzing R-CNN vs VLM Alignment")
    print("=" * 50)
    
    # Load data
    cat_map, anns_by_img, imginfo = load_coco_data(coco_path)
    vlm_data = load_vlm_data(vlm_pages_dir)
    
    print(f"Found {len(anns_by_img)} images with R-CNN data")
    print(f"Found {len(vlm_data)} VLM files")
    
    # Analyze each image
    results = []
    work_items = list(anns_by_img.items())
    
    if limit:
        work_items = work_items[:limit]
        print(f"Analyzing first {limit} images (limit mode)")
    
    for img_id, anns in tqdm(work_items, desc="Analyzing images"):
        try:
            result = analyze_single_image(
                img_id, anns, cat_map, imginfo, vlm_data,
                panel_thr, panel_nms
            )
            results.append(result)
        except Exception as e:
            print(f"Error analyzing image {img_id}: {e}")
            continue
    
    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n📊 Analysis Complete!")
        print(f"Results saved to: {output_csv}")
        
        # Summary statistics
        total_images = len(results)
        has_vlm = sum(1 for r in results if r['has_vlm_data'])
        good_alignment = sum(1 for r in results if r['panel_alignment_good'])
        good_text_coverage = sum(1 for r in results if r['text_coverage_good'])
        good_vlm_quality = sum(1 for r in results if r['vlm_quality_good'])
        recommend_training = sum(1 for r in results if r['recommend_training'])
        
        print(f"\n📈 Summary Statistics:")
        print(f"  Total images analyzed: {total_images}")
        print(f"  Images with VLM data: {has_vlm} ({has_vlm/total_images*100:.1f}%)")
        print(f"  Good panel alignment: {good_alignment} ({good_alignment/total_images*100:.1f}%)")
        print(f"  Good text coverage: {good_text_coverage} ({good_text_coverage/total_images*100:.1f}%)")
        print(f"  Good VLM quality: {good_vlm_quality} ({good_vlm_quality/total_images*100:.1f}%)")
        print(f"  Recommended for training: {recommend_training} ({recommend_training/total_images*100:.1f}%)")
        
        # Quality score distribution
        quality_scores = [r['quality_score'] for r in results]
        score_counts = {i: quality_scores.count(i) for i in range(6)}
        print(f"\n🎯 Quality Score Distribution:")
        for score in range(6):
            count = score_counts.get(score, 0)
            print(f"  Score {score}: {count} images ({count/total_images*100:.1f}%)")
        
        # Panel count statistics
        rcnn_counts = [r['rcnn_panel_count'] for r in results if r['rcnn_panel_count'] > 0]
        vlm_counts = [r['vlm_panel_count'] for r in results if r['vlm_panel_count'] > 0]
        
        if rcnn_counts:
            print(f"\n📊 Panel Count Statistics:")
            print(f"  R-CNN panels - Avg: {sum(rcnn_counts)/len(rcnn_counts):.1f}, Min: {min(rcnn_counts)}, Max: {max(rcnn_counts)}")
        if vlm_counts:
            print(f"  VLM panels - Avg: {sum(vlm_counts)/len(vlm_counts):.1f}, Min: {min(vlm_counts)}, Max: {max(vlm_counts)}")
        
        # Create filtered dataset recommendations
        high_quality = [r for r in results if r['quality_score'] >= 4]
        medium_quality = [r for r in results if r['quality_score'] == 3]
        low_quality = [r for r in results if r['quality_score'] <= 2]
        
        print(f"\n🎯 Training Dataset Recommendations:")
        print(f"  High quality (score 4-5): {len(high_quality)} images - Use for training")
        print(f"  Medium quality (score 3): {len(medium_quality)} images - Use with caution")
        print(f"  Low quality (score 0-2): {len(low_quality)} images - Exclude from training")
        
        # Create filtered lists
        high_quality_paths = [r['image_path'] for r in high_quality]
        medium_quality_paths = [r['image_path'] for r in medium_quality]
        
        # Perfect match analysis
        perfect_matches = [r for r in results if r['panel_count_ratio'] == 1.0]
        near_perfect = [r for r in results if 0.9 <= r['panel_count_ratio'] <= 1.1]
        
        perfect_match_paths = [r['image_path'] for r in perfect_matches]
        near_perfect_paths = [r['image_path'] for r in near_perfect]
        
        # Save filtered lists
        with open(output_csv.replace('.csv', '_high_quality.txt'), 'w') as f:
            for path in high_quality_paths:
                f.write(f"{path}\n")
        
        with open(output_csv.replace('.csv', '_medium_quality.txt'), 'w') as f:
            for path in medium_quality_paths:
                f.write(f"{path}\n")
        
        with open(output_csv.replace('.csv', '_perfect_matches.txt'), 'w') as f:
            for path in perfect_match_paths:
                f.write(f"{path}\n")
        
        with open(output_csv.replace('.csv', '_near_perfect.txt'), 'w') as f:
            for path in near_perfect_paths:
                f.write(f"{path}\n")
        
        print(f"\n🎯 Perfect Match Analysis:")
        print(f"  Perfect matches (1.0 ratio): {len(perfect_matches)} images ({len(perfect_matches)/total_images*100:.1f}%)")
        print(f"  Near-perfect (0.9-1.1 ratio): {len(near_perfect)} images ({len(near_perfect)/total_images*100:.1f}%)")
        print(f"  Perfect + Near-perfect: {len(perfect_matches) + len(near_perfect)} images ({(len(perfect_matches) + len(near_perfect))/total_images*100:.1f}%)")
        
        print(f"\n💾 Filtered lists saved:")
        print(f"  High quality: {output_csv.replace('.csv', '_high_quality.txt')}")
        print(f"  Medium quality: {output_csv.replace('.csv', '_medium_quality.txt')}")
        print(f"  Perfect matches: {output_csv.replace('.csv', '_perfect_matches.txt')}")
        print(f"  Near-perfect: {output_csv.replace('.csv', '_near_perfect.txt')}")
        
    else:
        print("❌ No results to analyze!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze R-CNN vs VLM alignment')
    parser.add_argument('--coco', required=True, help='Path to COCO detection JSON')
    parser.add_argument('--vlm_dir', required=True, help='Directory containing VLM JSON files')
    parser.add_argument('--output_csv', required=True, help='Output CSV file for analysis results')
    parser.add_argument('--limit', type=int, default=None, help='Limit analysis to first N images (default: process all)')
    parser.add_argument('--panel_thr', type=float, default=0.75, help='Panel detection threshold')
    parser.add_argument('--panel_nms', type=float, default=0.25, help='Panel NMS IoU threshold')
    
    args = parser.parse_args()
    
    analyze_alignment(
        args.coco, 
        args.vlm_dir, 
        args.output_csv,
        limit=args.limit,
        panel_thr=args.panel_thr,
        panel_nms=args.panel_nms
    )
