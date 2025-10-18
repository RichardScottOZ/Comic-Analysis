# Match create_calibre and training

This note captures the exact contract between the Calibre analyzer that emits DataSpec JSONs and the Closure-Lite training pipeline, plus copy-ready one-liners that we’ve used successfully.

## What produces DataSpec and what consumes it

- Producer: `benchmarks/detections/openrouter/create_perfect_match_filter_calibre_v2.py`
  - Reads: COCO predictions + VLM JSONs (Calibre analysis outputs)
  - Emits: DataSpec JSONs per page with panel boxes from COCO and text aggregated from VLM
  - Also emits: optional subset lists of VLM JSONs (for analysis/cross-checks), and an optional list of emitted DataSpec JSON paths for training

- Consumer: `benchmarks/detections/openrouter/closure_lite_dataset.py` + `closure_lite_simple_framework.py`
  - Loads DataSpec JSONs on-demand (lazy)
  - Builds per-panel crops, text tokens, composition features, and reading order
  - Trains the CLOSURE-Lite Simple model directly on these pages

## DataSpec v0.3 contract (training expects this)

Per file (one page):
- page_image_path: string. Absolute path preferred; relative is allowed but will be resolved against `--image_root`.
- panels: list of objects; each must yield a bbox. Training prefers:
  - panel_coords: [x, y, w, h] in pixels (integers). The loader will also accept VLM-style keys like bbox/box/rect/polygon and normalize them.
- text (optional but recommended):
  - text: { dialogue: [..], narration: [..], sfx: [..] } — arrays of strings; the loader aggregates them into a panel text string.

The loader tolerates schema variants and will normalize bboxes, but providing `panel_coords` avoids ambiguity.

## Analyzer → DataSpec details (what we emit)

- Panel boxes come from COCO detections filtered by panel category ids/names.
- Boxes are sorted top-to-bottom then left-to-right, then paired with VLM panels (if provided). With `--dataspec_require_equal_counts`, non-equal pages are skipped.
- Text is aggregated per panel from VLM JSON (dialogue, narration/caption, sfx, speakers, OCR-like fields).
- Emission filters: `--dataspec_min_det_score` for detection quality; `--require_image_exists` filters pages whose images aren’t found under `--image_roots`.
- Naming: by default we preserve directory structure to prevent collisions and keep training lookups stable.
  - Default `--dataspec_naming preserve_tree`:
    - If VLM JSON is under `--vlm_dir`, we mirror its relative folders in `--dataspec_out_dir`, and keep the same basename.
    - Otherwise we mirror a short tail of the image path (e.g., Series/JPG4CBZ/<image>.json).
  - Optional `--dataspec_naming hash`: flat folder with a short hash suffix added; not recommended for training unless you rely solely on emitted list files.
- Auditing: `--dataspec_debug_skips_out path.csv` logs every skipped page with reason (`missing_image`, `no_dets_after_threshold`, `equal_count_mismatch`, `zero_pairs`, `exception:<Type>`).
- Optional list of emitted JSONs: `--dataspec_list_out path.txt`.

## Training loader behavior (what it accepts now)

- Discovery:
  - If you pass a directory to `create_dataloader(json_dir=...)`, it now recurses (rglob) to find all `*.json`.
  - If you pass a `.txt` file to `json_dir`, it treats each non-empty line as a JSON path (compatible with `--dataspec_list_out`).
- Image path resolution:
  - Uses `page_image_path`/`image_path`/`image`/`IMAGE_PATH` if present.
  - If not present, infers from the JSON filename and parent folder under `--image_root`, with robust fallbacks and a one-time global index.
  - On POSIX with WSL paths, converts `E:\...` to `/mnt/e/...` automatically (when `--image_root` looks like `/mnt/*`).
- Panel normalization:
  - Accepts `panel_coords` directly or infers from `bbox/box/rect/polygon/segmentation`.
  - Detects normalized coordinates and scales to pixels if needed.
- Text aggregation:
  - Merges dialogue/narration/sfx and common variants, plus OCR-like fields.
- Zero panels:
  - Logs up to a few examples; pads tensors to `max_panels` to keep batches stable.

## End-to-end: runbook and one‑liners

1) Generate alignment CSV for Calibre (fast-only is typical for full run):

```powershell
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py --coco "E:\CalibreComics\test_dections\predictions.json" --vlm_dir "E:\CalibreComics_analysis" --vlm_map "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\all_vlm_jsons_calibre_training_map.json" --output_csv "calibre_rcnn_vlm_analysis_v2.csv" --num_workers 16 --fast_only
```

2) Emit a canonical DataSpec corpus for all matched pages (training‑safe naming):

```powershell
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py --coco "E:\CalibreComics\test_dections\predictions.json" --vlm_dir "E:\CalibreComics_analysis" --vlm_map "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\all_vlm_jsons_calibre_training_map.json" --output_csv "calibre_rcnn_vlm_analysis_v2.csv" --num_workers 8 --image_roots "E:\CalibreComics_extracted" --require_image_exists --min_text_coverage 0.6 --emit_dataspec_everything --dataspec_out_dir "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\calibre_dataspec_all" --dataspec_min_det_score 0.5 --dataspec_list_out "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\dataspec_all_list.txt" --dataspec_debug_skips_out "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\dataspec_all_skips.csv" --dataspec_naming preserve_tree
```

Optional: Emit subset VLM lists (for analysis cross‑checks):

```powershell
python benchmarks\detections\openrouter\create_perfect_match_filter_calibre_v2.py --coco "E:\CalibreComics\test_dections\predictions.json" --vlm_dir "E:\CalibreComics_analysis" --output_csv "calibre_rcnn_vlm_analysis_v2.csv" --num_workers 8 --json_require_exists --emit_all_json_lists
```

3) Train (two equivalent ways)
- Using the emitted list file from step 2 (works even with nested folders):

```powershell
python benchmarks\detections\openrouter\train_closure_lite_simple.py --json_list_file "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\dataspec_all_list.txt" --image_root "E:\CalibreComics_extracted" --output_dir "closure_lite_simple_calibre_all" --batch_size 4 --epochs 10 --num_workers 8 --num_heads 4 --temperature 0.1
```

- Or, pointing at the nested DataSpec directory (loader recurses):

```powershell
python benchmarks\detections\openrouter\train_closure_lite_simple.py --json_dir "C:\Users\Richard\OneDrive\GIT\CoMix\perfect_match_training\calibre_dataspec_all" --image_root "E:\CalibreComics_extracted" --output_dir "closure_lite_simple_calibre_all" --batch_size 4 --epochs 10 --num_workers 8 --num_heads 4 --temperature 0.1
```

Strict perfect‑match subset pipeline (from recent Workflow.md runs):
- Create a filtered subset and list (dry‑run shown):

```powershell
cd "C:\Users\Richard\OneDrive\GIT\CoMix"; python benchmarks\detections\openrouter\train_with_perfect_matches_parallel.py --perfect_matches_file "calibre_rcnn_vlm_analysis_perfect_matches.txt" --json_dir "E:\CalibreComics_dataspec_full" --image_root "E:\CalibreComics_extracted" --output_dir "E:\CalibreComics_perfect_match_subset" --create_filtered_data --write_list --strict_paths --dry_run
```

- Train using the strict list:

```powershell
cd "C:\Users\Richard\OneDrive\GIT\CoMix"; python benchmarks\detections\openrouter\train_closure_lite_simple.py --json_list_file "E:\CalibreComics_perfect_match_subset\calibre_perfect_match_jsons.txt" --image_root "E:\CalibreComics_extracted" --output_dir "closure_lite_simple_calibre_perfect_strict" --batch_size 4 --epochs 10 --num_workers 8 --num_heads 4 --temperature 0.1
```

## Expected counts and typical gaps
- In your last full run, the CSV had 336,881 matched pages; with `--require_image_exists`, `exists` was 309,924.
- DataSpec emission iterated those existing pages and wrote ~304,688 files.
  - Typical skips: `no_dets_after_threshold` (raise coverage by lowering `--dataspec_min_det_score`), zero R‑CNN panels, strict equal‑count filtering (if enabled), or rare exceptions.
- Use `--dataspec_debug_skips_out` to audit and decide which knobs to turn.

## Gotchas and troubleshooting
- If training can’t find images:
  - Ensure `--image_root` points to the same drive/folder used during emission.
  - On WSL, the dataset will convert `E:\...` to `/mnt/e/...` if `--image_root` is a `/mnt/*` path.
- If training sees zero JSONs:
  - When pointing to a directory, the loader recurses; verify that JSONs are under nested folders.
  - When pointing to a `.txt` list, make sure the file contains full paths, one per line, to `.json` files.
- Naming collisions:
  - Avoid hash mode unless you rely on list files; default preserve_tree keeps stable, de‑duplicated paths.
- Panel/text mismatch at training time:
  - The dataset normalizes diverse VLM schemas, but DataSpec with `panel_coords` produces the most stable results.

## Rationale for design choices
- Centralize matching and text aggregation in the analyzer to keep the contract single‑sourced.
- Preserve directory structure by default so that training can find JSONs without custom mappers.
- Provide a list‑file option and a skip log so runs are both reproducible and debuggable.

---
Last updated: 2025‑10‑18
