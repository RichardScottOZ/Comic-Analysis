#!/usr/bin/env python3
import argparse
import json
import csv
from collections import Counter, defaultdict

def analyze_with_json(pred_path, out_dir=None, report_zero_cat=None):
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    cat_id2name = {c["id"]: c["name"] for c in categories}

    # Count annotations per category
    cat_counts = Counter()
    # Count per-image category counts: {image_id: Counter({cat_id: count})}
    image_counts = defaultdict(Counter)

    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        img = ann.get("image_id")
        cat_counts[cid] += 1
        image_counts[img][cid] += 1

    # Optionally write CSVs
    if out_dir:
        cat_csv = out_dir.rstrip("\\/") + "/category_counts.csv"
        with open(cat_csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(["category_id", "category_name", "count"])
            for cid, cnt in cat_counts.most_common():
                w.writerow([cid, cat_id2name.get(cid, ""), cnt])

        img_csv = out_dir.rstrip("\\/") + "/image_category_counts.csv"
        with open(img_csv, "w", newline="", encoding="utf-8") as imf:
            # discover all category ids present
            all_cids = sorted({cid for c in image_counts.values() for cid in c.keys()})
            header = ["image_id"] + [f"cat_{cid}_{cat_id2name.get(cid,'')}" for cid in all_cids]
            w = csv.writer(imf)
            w.writerow(header)
            for img, cnts in image_counts.items():
                row = [img] + [cnts.get(cid, 0) for cid in all_cids]
                w.writerow(row)

    # Print a readable summary
    print("Categories found (id -> name):")
    for cid, name in sorted(cat_id2name.items()):
        print(f"  {cid} -> {name}")
    print("\nTop categories by annotation count:")
    for cid, cnt in cat_counts.most_common():
        print(f"  {cid} ({cat_id2name.get(cid,'')}): {cnt:,}")

    if report_zero_cat is not None:
        # List images with zero annotations for the requested category id or name
        # Accept numeric id or category name
        target_cid = None
        if isinstance(report_zero_cat, int) or report_zero_cat.isdigit():
            target_cid = int(report_zero_cat)
        else:
            # find cid by name (exact match)
            for cid, name in cat_id2name.items():
                if name == report_zero_cat:
                    target_cid = cid
                    break
        if target_cid is None:
            print(f"\nRequested category '{report_zero_cat}' not found in categories.")
        else:
            no_cat_imgs = [img for img, cnts in image_counts.items() if cnts.get(target_cid, 0) == 0]
            print(f"\nImages with ZERO detections for category {target_cid} ({cat_id2name.get(target_cid)}): {len(no_cat_imgs):,}")
            # print first few examples
            for img in no_cat_imgs[:20]:
                print("  ", img)

    return cat_counts, image_counts, cat_id2name

def analyze_with_ijson(pred_path, out_dir=None, report_zero_cat=None):
    try:
        import ijson
    except Exception as e:
        raise RuntimeError("ijson not available. Install with: pip install ijson") from e

    cat_id2name = {}
    cat_counts = Counter()
    image_counts = defaultdict(Counter)

    with open(pred_path, "rb") as f:
        parser = ijson.parse(f)
        # We will walk and accumulate categories and annotations
        # This is a simple event-based read; ijson's convenience methods could be used
        current_field = None
        # Two-phase read: first collect categories, then process annotations by scanning keys
        # Simpler: use ijson.items for categories and annotations
        f.seek(0)
        for c in ijson.items(f, "categories.item"):
            cid = c.get("id")
            name = c.get("name")
            if cid is not None:
                cat_id2name[cid] = name

    # Now process annotations streaming
    with open(pred_path, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            cid = ann.get("category_id")
            img = ann.get("image_id")
            cat_counts[cid] += 1
            image_counts[img][cid] += 1

    # same reporting/writing as json version
    if out_dir:
        cat_csv = out_dir.rstrip("\\/") + "/category_counts.csv"
        with open(cat_csv, "w", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            w.writerow(["category_id", "category_name", "count"])
            for cid, cnt in cat_counts.most_common():
                w.writerow([cid, cat_id2name.get(cid, ""), cnt])

    print("Categories found (id -> name):")
    for cid, name in sorted(cat_id2name.items()):
        print(f"  {cid} -> {name}")
    print("\nTop categories by annotation count:")
    for cid, cnt in cat_counts.most_common():
        print(f"  {cid} ({cat_id2name.get(cid,'')}): {cnt:,}")

    if report_zero_cat is not None:
        target_cid = None
        if isinstance(report_zero_cat, int) or (isinstance(report_zero_cat, str) and report_zero_cat.isdigit()):
            target_cid = int(report_zero_cat)
        else:
            for cid, name in cat_id2name.items():
                if name == report_zero_cat:
                    target_cid = cid
                    break
        if target_cid is None:
            print(f"\nRequested category '{report_zero_cat}' not found.")
        else:
            no_cat_imgs = [img for img, cnts in image_counts.items() if cnts.get(target_cid, 0) == 0]
            print(f"\nImages with ZERO detections for category {target_cid} ({cat_id2name.get(target_cid)}): {len(no_cat_imgs):,}")
            for img in no_cat_imgs[:20]:
                print("  ", img)

    return cat_counts, image_counts, cat_id2name

def main():
    p = argparse.ArgumentParser(description="Analyze COCO predictions JSON and report counts by category/image.")
    p.add_argument("predictions_json", help="Path to predictions.json")
    p.add_argument("--out_dir", help="Optional directory to write CSV summaries (category_counts.csv, image_category_counts.csv)")
    p.add_argument("--stream", action="store_true", help="Use ijson streaming parser (recommended for very large files).")
    p.add_argument("--zero_cat", help="Category id or name to list images that have ZERO detections for that category (optional).")
    args = p.parse_args()

    if args.stream:
        cat_counts, image_counts, cat_id2name = analyze_with_ijson(args.predictions_json, args.out_dir, args.zero_cat)
    else:
        cat_counts, image_counts, cat_id2name = analyze_with_json(args.predictions_json, args.out_dir, args.zero_cat)

if __name__ == "__main__":
    main()