# Plan: A Visual Embedding Search Interface

This document explains the process and logic of the CoMiX visual search UI, including its evolution from a simple search tool to a multi-stage exploration interface.

## 1. Backend (Using Flask)

A simple Python web server is the core of the application.

*   **File:** `search_server.py`
*   **On Startup:** The server loads the Zarr dataset, the embedding model, and the master manifest CSV into memory for fast query responses.
*   **API Endpoint `/search`:** This endpoint accepts various query types (`image`, `text`, `page_id`, `embedding`) and search modes to perform different kinds of analysis.

## 2. Frontend (HTML & JavaScript)

A single web page serves as the entire user interface.

*   **File:** `templates/index.html`
*   **UI Components:** The page has inputs for text queries, image uploads, and Page ID searches.
*   **JavaScript Logic:** On form submission, JavaScript sends the query to the `/search` backend. Upon receiving results, it dynamically generates a visual grid of images or a list of page IDs.

## 3. Initial Implementation: Core Search Modes

The first version of the interface provided three primary ways to search the dataset.

### Image Search
*   **Page Mode:** Finds whole pages that are visually and structurally similar to a query image.
*   **Panel Mode:** Finds individual panels across the entire dataset that are visually similar to a query image.

### Text Search
*   **Semantic Mode:** Converts a text query (e.g., "a hero fighting a monster") into an embedding and finds panels that are conceptually similar.
*   **Keyword Mode:** Performs a fast, literal string search for an exact word or phrase (e.g., "Vampirella").

## 4. Phase 2 Upgrade: Multi-Stage Search & Exploration

To enable deeper exploration and more precise queries, the interface was upgraded with a multi-stage workflow.

### Feature 1: Page ID Search

This feature allows a user to quickly find specific pages if they know part of the page's canonical ID.

*   **UI:** A new "Page ID Search" radio button reveals a text input.
*   **Logic:** The user enters a fragment of a `canonical_id` (e.g., `vampirella/001`). The backend searches the `master_manifest.csv` and returns a list of all matching IDs.
*   **Result:** The UI displays a simple, clickable list of the full `canonical_id`s that were found.

### Feature 2: Full Embedding Search ("Use as Query")

This is the most powerful search mode. It uses a known, existing page from the dataset as a "perfect" query to find the most holistically similar pages.

*   **UI:** Every image result now has a **"Use as Query"** button.
*   **Logic:**
    1.  When the user clicks this button, the frontend sends the `canonical_id` of that result back to the `/search` endpoint with the new `query_type` of `embedding`.
    2.  The backend receives this ID and uses the `get_embedding_by_page_id` function to look up the pre-calculated, rich, multi-modal `page_embedding` for that page directly from the Zarr dataset.
    3.  This "perfect" embedding is then used as the query vector to find other pages with the most similar embeddings.
*   **Benefit:** This provides the most accurate possible similarity search, finding pages that are similar in content, style, and layout, because it uses a complete, multi-modal vector from the dataset itself as the query.

## 5. Refactored Logic (`search_utils.py`)

This file contains the core functions for the application:

*   **Loading:** `load_zarr_dataset`, `load_model`, `load_manifest`.
*   **Embedding Generation:** `get_embedding_for_image`, `get_embedding_for_text`.
*   **Search Functions:** `find_similar_pages`, `find_similar_panels`, `keyword_search_panels`.
*   **New Multi-Stage Functions:**
    *   `page_id_search`: Searches the loaded manifest data by `canonical_id`.
    *   `get_embedding_by_page_id`: Retrieves a specific page's embedding vector from the Zarr dataset using its ID.

## 6. How to Run the CoMiX Visual Interface

Follow these steps to run the application after the files have been created.

### 1. Install Dependencies
The web interface requires the Flask library. Install it using pip:
```bash
pip install Flask
```

### 2. Configure Paths
**This is a critical step.** Open the `search_server.py` file in a text editor and update the following configuration variables at the top of the file to match your local environment:
*   `ZARR_PATH`: The full path to your `combined_embeddings.zarr` directory.
*   `CHECKPOINT_PATH`: The full path to your `.pth` model checkpoint file.
*   `IMAGE_ROOT`: The root directory where your comic images are stored (e.g., `E:\CalibreComics_extracted`).
*   `MANIFEST_PATH`: The path to your `master_manifest.csv` file.

### 3. Launch the Server
Simply double-click and run the `run_web_ui.bat` file in the project's root directory. This will start the web server.

### 4. Access the Interface
Open your web browser and navigate to the following address:
[http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## 7. Guide to Running the Full Data Pipeline

This guide explains how to run the new, complete, manifest-driven data pipeline on different operating systems.

### Dependencies

Ensure the following are installed in your Python environment:

*   `pip install rarfile PyMuPDF ebooklib Pillow tqdm pandas`

And ensure the following system utility is installed and in your PATH:

*   **Windows:** Install WinRAR and ensure `unrar.exe` is in your PATH, or place it in a known system directory.
*   **Ubuntu/WSL:** `sudo apt-get update && sudo apt-get install unrar`

### Pathing Differences

The only difference when running on Windows vs. WSL/Linux is how you format the paths to your data on external drives.

*   **Windows:** Use standard drive letters (e.g., `E:\Comics\Input`)
*   **WSL/Ubuntu:** Mount the drive and use the `/mnt/` path (e.g., `/mnt/e/Comics/Input`)

### The Full Pipeline Command Sequence

Run these commands from the root of the `CoMiX` project directory.

#### **Example on Windows:**

```powershell
# Step 1: Create the master manifest from all your sources
python .\tools\create_master_manifest.py --input_dirs "E:\CalibreComics" "E:\amazon" --extraction_dir "E:\Comics_Extracted_Clean" --output_csv "master_manifest.csv"

# Step 2: Generate panel detections using the manifest
python .\benchmarks\detections_2000ad\faster_rcnn_calibre_v2.py --manifest_file "master_manifest.csv" --weights_path "path\to\your\weights.pth" --output_path "predictions_v2.json"

# Step 3: Generate VLM analysis using the manifest
python .\benchmarks\detections\openrouter\batch_comic_analysis_multi_v2.py --manifest_file "master_manifest.csv" --output_dir "VLM_analysis_v2"

# Step 4: Generate the final, clean DataSpec dataset using the manifest and new outputs
python .\benchmarks\detections\openrouter\coco_to_dataspec_calibre_v2.py --manifest_file "master_manifest.csv" --coco_file "predictions_v2.json" --vlm_dir "VLM_analysis_v2" --output_dir "DataSpec_v2_final" --subset perfect --list_output "perfect_match_list.txt"
```

#### **Example on WSL or Ubuntu:**

(Note the change in paths for `--input_dirs`, `--extraction_dir`, and `--weights_path`)

```bash
# Step 1: Create the master manifest from all your sources
python ./tools/create_master_manifest.py --input_dirs "/mnt/e/CalibreComics" "/mnt/e/amazon" --extraction_dir "/mnt/e/Comics_Extracted_Clean" --output_csv "master_manifest.csv"

# Step 2: Generate panel detections using the manifest
python ./benchmarks/detections_2000ad/faster_rcnn_calibre_v2.py --manifest_file "master_manifest.csv" --weights_path "path/to/your/weights.pth" --output_path "predictions_v2.json"

# Step 3: Generate VLM analysis using the manifest
python ./benchmarks/detections/openrouter/batch_comic_analysis_multi_v2.py --manifest_file "master_manifest.csv" --output_dir "VLM_analysis_v2"

# Step 4: Generate the final, clean DataSpec dataset using the manifest and new outputs
python ./benchmarks/detections/openrouter/coco_to_dataspec_calibre_v2.py --manifest_file "master_manifest.csv" --coco_file "predictions_v2.json" --vlm_dir "VLM_analysis_v2" --output_dir "DataSpec_v2_final" --subset perfect --list_output "perfect_match_list.txt"
```