# Canonical Mapping Tool - Usage Guide

## Problem

Three different robots created three different naming schemes:

1. **faster_rcnn_calibre.py**: Creates COCO image_id as `{directory}_{filename}`
2. **batch_comic_analysis_multi.py**: Creates VLM JSON as `{relative_path_with_underscores}.json`
3. **COCO predictions.json**: Stores image_id in the images list

This creates a mapping nightmare for downstream tools.

## Solution

`generate_canonical_mapping.py` creates a single CSV that maps:
- Original image path → COCO image_id → VLM JSON path
- With existence checks for all three

## Usage

### Basic Usage

```cmd
python tools\generate_canonical_mapping.py ^
  --image_dir "E:\CalibreComics_extracted" ^
  --vlm_dir "E:\CalibreComics_analysis" ^
  --coco_file "E:\CalibreComics\test_dections\predictions.json" ^
  --output_csv "key_mapping_report_claude.csv"
```

### Test Run (First 1000 images)

```cmd
python tools\generate_canonical_mapping.py ^
  --image_dir "E:\CalibreComics_extracted" ^
  --vlm_dir "E:\CalibreComics_analysis" ^
  --coco_file "E:\CalibreComics\test_dections\predictions.json" ^
  --output_csv "key_mapping_test.csv" ^
  --limit 1000
```

## Output CSV Columns

| Column | Description | Example |
|--------|-------------|---------|
| `image_path` | Full path to actual image file | `E:\CalibreComics_extracted\Series\issue_01.jpg` |
| `image_exists` | Binary check if image exists on disk | `True` |
| `image_rel_path` | Relative path from image_dir | `Series\issue_01.jpg` |
| `coco_image_id` | Image ID as in predictions.json | `Series_issue_01.jpg` |
| `coco_id_exists` | Binary check if ID exists in COCO | `True` |
| `coco_file_name` | file_name from COCO images list | `Series\issue_01.jpg` |
| `vlm_json_filename` | Expected VLM JSON filename | `Series_issue_01.json` |
| `vlm_json_path` | Full path to VLM JSON | `E:\CalibreComics_analysis\Series_issue_01.json` |
| `vlm_json_exists` | Binary check if VLM JSON exists | `True` |
| `vlm_json_rel_path` | Relative path from vlm_dir | `Series_issue_01.json` |

## Example Output

```csv
image_path,image_exists,image_rel_path,coco_image_id,coco_id_exists,coco_file_name,vlm_json_filename,vlm_json_path,vlm_json_exists,vlm_json_rel_path
E:\CalibreComics_extracted\2000AD\issue_001.jpg,True,2000AD\issue_001.jpg,2000AD_issue_001.jpg,True,2000AD\issue_001.jpg,2000AD_issue_001.json,E:\CalibreComics_analysis\2000AD_issue_001.json,True,2000AD_issue_001.json
```

## Summary Statistics

The tool prints summary stats like:

```
Total records: 337,123
Images exist: 337,123 (100.0%)
COCO IDs exist: 336,881 (99.9%)
VLM JSONs exist: 334,567 (99.2%)
All three exist: 334,321 (99.2%)

⚠️  242 images missing from COCO predictions.json
⚠️  2,556 images missing VLM analysis JSONs
```

## What To Do With This Mapping

Once you have the canonical CSV, you can:

1. **Identify missing data**:
   ```python
   import pandas as pd
   df = pd.read_csv('key_mapping_report_claude.csv')
   
   # Missing COCO predictions
   missing_coco = df[~df['coco_id_exists']]
   print(f"{len(missing_coco)} images need R-CNN predictions")
   
   # Missing VLM analysis
   missing_vlm = df[~df['vlm_json_exists']]
   print(f"{len(missing_vlm)} images need VLM analysis")
   ```

2. **Restructure VLM directory** to match COCO structure:
   ```python
   import shutil
   for _, row in df.iterrows():
       if row['vlm_json_exists']:
           # Copy/move VLM JSON to match image structure
           src = row['vlm_json_path']
           # Create dest based on image_rel_path
           dest = os.path.join('E:\\CalibreComics_analysis_restructured', 
                              row['image_rel_path'].replace('.jpg', '.json'))
           os.makedirs(os.path.dirname(dest), exist_ok=True)
           shutil.copy2(src, dest)
   ```

3. **Create a simplified COCO file** with consistent IDs:
   ```python
   # Use image_path as the canonical key
   # Update COCO to use absolute paths or consistent relative paths
   ```

4. **Feed to simpler robots** that don't need complex path resolution:
   ```python
   # Load mapping
   df = pd.read_csv('key_mapping_report_claude.csv')
   
   # Filter to complete records only
   complete = df[df['image_exists'] & df['coco_id_exists'] & df['vlm_json_exists']]
   
   # Now any downstream tool can use:
   for _, row in complete.iterrows():
       image = row['image_path']
       coco_id = row['coco_image_id']
       vlm_json = row['vlm_json_path']
       # Process with all three guaranteed to exist
   ```

## Troubleshooting

### "COCO IDs don't match expected pattern"

The tool reconstructs COCO IDs based on faster_rcnn_calibre.py logic:
- `image_id = f"{directory}_{filename}"`

If your COCO file was generated differently, you may need to adjust the `reconstruct_faster_rcnn_id()` function.

### "VLM JSONs not found"

The tool reconstructs VLM filenames based on batch_comic_analysis_multi.py logic:
- Flattens relative path with underscores
- Removes extension, adds `.json`

If your VLM files were generated differently (e.g., preserving directory structure), you may need to adjust the `reconstruct_vlm_filename()` function.

### "Too slow on 337K images"

The tool walks the entire directory tree once and builds indexes. For 337K images:
- Image indexing: ~30-60 seconds
- VLM indexing: ~10-20 seconds  
- Mapping generation: ~2-3 minutes
- Total: ~3-5 minutes

If you just want to test the logic, use `--limit 1000` first.

## Next Steps

After generating the mapping:

1. **Inspect mismatches**: Look at rows where exists flags don't all match
2. **Re-run failed jobs**: Use the CSV to identify which images need re-processing
3. **Restructure data**: Use the mapping to reorganize files into a consistent structure
4. **Update workflows**: Modify downstream tools to use the canonical mapping instead of guessing paths

The goal is: **One source of truth for all path mappings**.
