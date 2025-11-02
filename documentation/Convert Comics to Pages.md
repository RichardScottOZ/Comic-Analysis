# Converting Comic Archives to a Master Page Manifest

This document explains the process and logic of the `tools/create_master_manifest.py` script, which serves as the new, robust "Step 0" for the CoMiX data pipeline.

## 1. The Problem: Inconsistent Data Sources

The primary challenge in this project is the variety and inconsistency of the source comic files. The collection includes comics from numerous sources (Humble Bundles, DriveThruComics, Kickstarter, etc.) in multiple formats, including:

*   Container archives like `.cbz`, `.cbr`, `.pdf`, and `.epub`.
*   Pre-extracted directories of loose image files.

These sources often have inconsistent naming schemes and directory structures, making it impossible to reliably link images to their metadata. Attempting to process them with scripts that rely on filename matching is prone to error and leads to data corruption.

## 2. The Solution: A Centralized, Two-Phase Pipeline

To solve this, we designed a new ingestion pipeline centered around a single script: `tools/create_master_manifest.py`. This script acts as a "universal adapter" that handles all the complexity of the different source formats.

Its purpose is to:
1.  Ingest all supported container files and pre-extracted image directories.
2.  Extract images from containers into a single, clean, and organized directory.
3.  Validate every single page from all sources to ensure it's a proper image.
4.  Produce a **Master Manifest CSV file** that serves as the single, definitive source of truth for the entire downstream pipeline.

## 3. The Workflow Step-by-Step

The `create_master_manifest.py` script follows a clear, two-phase process:

### Phase 1: Container Processing

1.  **Scan for Containers:** The script begins by recursively scanning the specified input directories for all supported container files: `.cbz`, `.cbr`, `.pdf`, and `.epub`.

2.  **Extract Images:** For each container file found, it performs an extraction into a designated `--extraction_dir`. This process preserves the relative directory structure from the source to prevent filename collisions.
    *   **CBZ (.zip):** Uses the standard `zipfile` library.
    *   **CBR (.rar):** Uses the `rarfile` library (which requires the `unrar` system utility to be installed).
    *   **PDF:** Uses `PyMuPDF` (`fitz`) to render each page as a high-quality PNG image.
    *   **EPUB:** Uses the `ebooklib` library to parse the book structure and save all image items in order.

3.  **Validate and Manifest:** After extracting a container, the script iterates through the newly created image files. It validates each one with the Pillow library and, if valid, writes its `canonical_id` and `absolute_image_path` to the `master_manifest.csv`.

### Phase 2: Loose Image Processing

1.  **Scan for Loose Images:** After processing all containers, the script performs a second scan of the same input directories, this time looking for loose image files (`.jpg`, `.png`, etc.).

2.  **De-duplicate:** It intelligently ignores any image file whose path matches one that was already extracted from a container. This prevents duplicate entries in the manifest.

3.  **Validate and Manifest:** For every new, valid loose image it finds, it generates a `canonical_id` based on its original path and writes the entry to the `master_manifest.csv`. The `absolute_image_path` for these entries will point to their original location.

## 4. The Manifest-Driven Pipeline (TODO)

The `master_manifest.csv` file produced by this script is now the definitive source of truth for your entire collection. The next step is to use the new `_v2` versions of the downstream processing scripts that read from this manifest.

### Step 4.1: `faster_rcnn_calibre_v2.py`

*   **Input:** Reads `master_manifest.csv`.
*   **Logic:** Iterates through the manifest and processes the image at `absolute_image_path`.
*   **Output:** Creates a `predictions_v2.json` file where the `image_id` is the `canonical_id` from the manifest.

### Step 4.2: `batch_comic_analysis_multi_v2.py`

*   **Input:** Reads `master_manifest.csv`.
*   **Logic:** Iterates through the manifest and processes the image at `absolute_image_path`.
*   **Output:** Saves VLM analysis JSON files named after the `canonical_id` (e.g., `series-a/001.json`).

### Step 4.3: `coco_to_dataspec_calibre_v2.py` (The Final Join)

*   **Input:** The manifest, the new `predictions_v2.json`, and the new VLM analysis directory.
*   **Logic:** Iterates through the manifest and uses the `canonical_id` to perform a direct, error-proof join of the COCO and VLM data.
*   **Result:** A clean, perfectly synchronized `DataSpec` dataset, ready for embedding generation.

## 5. How to Use the Pipeline: A Cross-Platform Guide

This guide explains how to run the full, manifest-driven pipeline on different operating systems.

### Dependencies

Ensure the following are installed in your Python environment:

*   `pip install rarfile PyMuPDF ebooklib Pillow tqdm`

And ensure the following system utility is installed and in your PATH:

*   **Windows:** Install WinRAR and ensure `unrar.exe` is in your PATH, or place it in a known system directory.
*   **Ubuntu/WSL:** `sudo apt-get update && sudo apt-get install unrar`

### Pathing Differences

The only difference when running on Windows vs. WSL/Linux is how you format the paths to your data on external drives.

*   **Windows:** Use standard drive letters (e.g., `E:\Comics`)
*   **WSL/Ubuntu:** Mount the drive and use the `/mnt/` path (e.g., `/mnt/e/Comics`)

### Running the Full Pipeline

Here is the full sequence of commands. **Run these from the root of the `CoMix` project directory.**

#### Example on Windows:

```powershell
# Step 1: Create the master manifest from all your sources
python .\tools\create_master_manifest.py --input_dirs "E:\CalibreComics" "E:\amazon" --extraction_dir "E:\Comics_Extracted_Clean" --output_csv "master_manifest.csv"

# Step 2: Generate panel detections
python .\benchmarks\detections_2000ad\faster_rcnn_calibre_v2.py --manifest_file "master_manifest.csv" --weights_path "path\to\your\weights.pth" --output_path "predictions_v2.json"

# Step 3: Generate VLM analysis
python .\benchmarks\detections\openrouter\batch_comic_analysis_multi_v2.py --manifest_file "master_manifest.csv" --output_dir "VLM_analysis_v2"

# Step 4: Generate the final, clean DataSpec dataset (e.g., for the 'perfect' subset)
python .\benchmarks\detections\openrouter\coco_to_dataspec_calibre_v2.py --manifest_file "master_manifest.csv" --coco_file "predictions_v2.json" --vlm_dir "VLM_analysis_v2" --output_dir "DataSpec_v2_final" --subset perfect --list_output "perfect_match_list.txt"
```

#### Example on WSL or Ubuntu:

(Note the change in paths for `--input_dirs` and `--extraction_dir`)

```bash
# Step 1: Create the master manifest from all your sources
python ./tools/create_master_manifest.py --input_dirs "/mnt/e/CalibreComics" "/mnt/e/amazon" --extraction_dir "/mnt/e/Comics_Extracted_Clean" --output_csv "master_manifest.csv"

# Step 2: Generate panel detections
python ./benchmarks/detections_2000ad/faster_rcnn_calibre_v2.py --manifest_file "master_manifest.csv" --weights_path "path/to/your/weights.pth" --output_path "predictions_v2.json"

# Step 3: Generate VLM analysis
python ./benchmarks/detections/openrouter/batch_comic_analysis_multi_v2.py --manifest_file "master_manifest.csv" --output_dir "VLM_analysis_v2"

# Step 4: Generate the final, clean DataSpec dataset (e.g., for the 'perfect' subset)
python ./benchmarks/detections/openrouter/coco_to_dataspec_calibre_v2.py --manifest_file "master_manifest.csv" --coco_file "predictions_v2.json" --vlm_dir "VLM_analysis_v2" --output_dir "DataSpec_v2_final" --subset perfect --list_output "perfect_match_list.txt"
```