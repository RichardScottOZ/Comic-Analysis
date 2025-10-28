"""Convert a key mapping CSV (image_path, predictions_json_id, vlm_json_path) into a vlm_map JSON.

The analyzer accepts a vlm_map that maps VLM JSON files to image paths (the canonical link).
This script reads `key_mapping_report_claude.csv` (or similar) and emits a JSON map.

Usage (PowerShell):
  python .\benchmarks\detections\openrouter\convert_key_mapping_to_vlm_map.py --csv key_mapping_report_claude.csv --output vlm_map.json

Options:
  --format vlm_to_image (default) : produce mapping { vlm_json_path: image_path }
  --format predid_to_vlm           : produce mapping { predictions_json_id: vlm_json_path }
  --require_exists                 : only include entries where both vlm_json_path and image_path exist on disk
  --image_col, --vlm_col, --predid_col : column names in the CSV (defaults match your example)

The script is defensive about Windows/WSL path normalization and prints a short summary.
"""
import argparse
import csv
import json
from pathlib import Path


def norm_path(p: str) -> str:
    if p is None:
        return ''
    p = p.strip()
    if not p:
        return ''
    # Keep as-is but try to normalize separators
    return str(Path(p))


def main():
    ap = argparse.ArgumentParser(description='Convert key mapping CSV to vlm_map JSON')
    ap.add_argument('--csv', required=True, help='Input CSV file path')
    ap.add_argument('--output', required=True, help='Output JSON map path')
    ap.add_argument('--format', choices=['vlm_to_image', 'predid_to_vlm'], default='vlm_to_image', help='Output mapping format')
    ap.add_argument('--require_exists', action='store_true', help='Only include rows where both files exist on disk')
    ap.add_argument('--image_col', default='image_path', help='CSV column name for image path')
    ap.add_argument('--vlm_col', default='vlm_json_path', help='CSV column name for vlm json path')
    ap.add_argument('--predid_col', default='predictions_json_id', help='CSV column name for predictions JSON id')
    args = ap.parse_args()

    inp = Path(args.csv)
    if not inp.exists():
        raise SystemExit(f'CSV not found: {inp}')

    mapping = {}
    total = 0
    included = 0
    skipped = 0

    with open(inp, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            total += 1
            image_raw = row.get(args.image_col, '')
            vlm_raw = row.get(args.vlm_col, '')
            predid_raw = row.get(args.predid_col, '')

            image_path = norm_path(image_raw)
            vlm_path = norm_path(vlm_raw)
            predid = predid_raw.strip() if predid_raw else ''

            # Optionally require files to exist
            if args.require_exists:
                exists_ok = True
                if args.format == 'vlm_to_image':
                    if not Path(vlm_path).exists() or not Path(image_path).exists():
                        exists_ok = False
                else:
                    # predid_to_vlm: require vlm_json exists
                    if not Path(vlm_path).exists():
                        exists_ok = False
                if not exists_ok:
                    skipped += 1
                    continue

            if args.format == 'vlm_to_image':
                if not vlm_path:
                    skipped += 1
                    continue
                mapping[vlm_path] = image_path
            else:
                if not predid:
                    skipped += 1
                    continue
                mapping[predid] = vlm_path

            included += 1

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f'Processed {total} rows; included: {included}; skipped: {skipped}; wrote {len(mapping)} mappings to {outp}')


if __name__ == '__main__':
    main()
