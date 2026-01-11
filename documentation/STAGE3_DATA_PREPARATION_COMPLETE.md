# Stage 3 Data Preparation: Complete Workflow & Verification

**Date:** Jan 2026  
**Status:** COMPLETE  
**Dataset Size:** 1,215,620 Pages

This document details the exact workflow used to prepare the 1.2 million page dataset for Stage 3 (Panel Feature Generation). It resolves the complexities of distributed storage (S3 vs Local), manifest ID mismatches, and data integrity verification.

## 1. Embedding Generation (Stage 2)
**Goal:** Generate semantic embeddings (SigLIP + Qwen) for all pages.

*   **Script:** `src/cosmo/generate_embeddings_zarr_batch.py`
*   **Manifest:** `manifests/calibrecomics-extracted_manifest.csv` (S3-based IDs)
*   **Output:** `cosmo4v_embeddings.zarr` (15GB)
*   **Key Feature:** Used Suffix Matching to map S3 IDs to Local Files (`E:\...`).
*   **Result:** 1,215,620 pages processed.

## 2. Page Classification (PSS)
**Goal:** Classify pages (Story vs Cover vs Ad) to filter training data.

*   **Script:** `src/cosmo/classify_pages_zarr.py`
*   **Model:** `BookBERT v4` (Authentic architecture with 9 classes).
*   **Input:** `cosmo4v_embeddings.zarr`
*   **Output:** Updated `prediction` variable in Zarr.
*   **Key Stats:** ~973,000 Story pages (~80%).

## 3. Data Alignment (The "Bridge")
**Goal:** Create unified JSONs containing CNN Panel Boxes (Geometry) and VLM/OCR Text (Content).

*   **Script:** `src/tools/align_panels_and_text.py`
*   **Challenge:** Bridging "Master Manifest" (Local Detections) and "Calibre Manifest" (Cloud VLM).
*   **Solution:** **Suffix Bridging Strategy**.
    *   Index Master Manifest by filename (leaf).
    *   Match Calibre IDs by checking if any path suffix exists in the Master Index.
    *   This robustly handles `amazon/`, `CalibreComics/`, and `S3://` prefix mismatches.
*   **Result:** 
    *   **Success:** 1,215,620 JSONs generated.
    *   **Missing CNN:** 346 files (Confirmed as `__MACOSX` junk files).

## 4. Data Modalities & Text Hierachy
The aligned JSONs contain a hierarchy of text signals to support different learning objectives:

*   **Panel-Level Text (`panels[].text`):** 
    *   **Source:** PaddleOCR spatial intersection with CNN boxes.
    *   **Usage:** Primary signal for Stage 3 Contrastive Learning. It aligns local visual features (crops) with local dialogue/narration.
*   **Full-Page Text (`full_page_text`):**
    *   **Source:** VLM (Gemini/Zhipu) dense analysis and summaries.
    *   **Usage:** Preserved as a high-level semantic anchor. While not used for local panel alignment in Stage 3, it is the primary input for **Stage 4 (Sequence Modeling)** to understand the broader narrative context.

## 5. Integrity Verification
**Goal:** Prove that the "missing" files are indeed junk and not data loss.

*   **Script:** `src/tools/audit_macos_junk.py`
*   **Findings:**
    *   Calibre Manifest Total: **1,215,966**
    *   Master Manifest Total: **1,215,620**
    *   Difference: **346**
    *   MacOS Junk Count: **346**
*   **Conclusion:** Dataset is 100% complete. Every valid image has a corresponding JSON.

## 5. Training Setup (Stage 3)
**Goal:** Train the domain-adapted encoder.

*   **Script:** `src/version2/train_stage3.py`
*   **Dataset Loader:** `src/version2/stage3_dataset.py`
*   **Logic:**
    1.  Load **Master Manifest** to find local images.
    2.  Build **JSON Map** (Suffix Strategy) to find aligned JSONs.
    3.  Load **PSS Labels** (from Zarr export) to filter for "Story" pages.
    4.  Train on the intersection (Image + JSON + Story Label).

## Artifact Locations
*   **Zarr Store:** `cosmo4v_embeddings` (15GB)
*   **Aligned JSONs:** `E:/Comic_Analysis_Results_v2/stage3_json`
*   **Detections:** `E:/Comic_Analysis_Results_v2/detections`
*   **VLM Data:** `E:/vlm_recycling_staging`
*   **PSS Labels:** `pss_labels_v1.json`

---
*Verified correct as of Jan 2026.*
