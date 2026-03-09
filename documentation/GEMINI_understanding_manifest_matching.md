# Gemini Understanding: Manifest Matching & Path Resolution

This document details the critical logic required to bridge the two primary manifests used in the Comic Analysis pipeline. Understanding this relationship is mandatory for any script that needs to link **Canonical IDs** (Labels/JSONs) with **Local Image Paths** (Master Manifest).

## The Two Manifests

### 1. Master Manifest (Local Source of Truth)
*   **Path:** `manifests/master_manifest_20251229.csv`
*   **Purpose:** Maps canonical IDs to **Local File Paths** on the GPU machine (`E:\...`).
*   **Key Format:** Flat strings, often constructed from `Folder_Filename`.
    *   Example: `#Guardian 001_#Guardian 001 - p000.jpg`
*   **Path Format:** `E:\amazon\#Guardian 001_#Guardian 001 - p000.jpg.png` (Note the double extension).

### 2. Calibre Manifest (Cloud/Processing Source of Truth)
*   **Path:** `manifests/calibrecomics-extracted_manifest.csv`
*   **Purpose:** Maps canonical IDs to **S3 URIs** used for cloud processing (VLM/OCR).
*   **Key Format:** Nested paths representing the S3 structure.
    *   Example: `CalibreComics_extracted/amazon/#Guardian 001_#Guardian 001 - p000.jpg`
    *   Example: `CalibreComics_extracted/13thfloor vol1 - Unknown/JPG4CBZ/0001`
*   **Path Format:** `s3://calibrecomics-extracted/CalibreComics_extracted/amazon/...`

## The Problem: ID Mismatch

Scripts (like `train_stage3.py` or `align_panels_and_text.py`) often receive inputs derived from **different manifests**:
*   **PSS Labels / Stage 3 JSONs:** Generated using the **Calibre Manifest** keys.
*   **Image Access:** Requires the **Master Manifest** local paths.

Attempting to match them directly fails because:
`#Guardian 001...` != `CalibreComics_extracted/amazon/#Guardian 001...`

## The Solution: Suffix Bridging Strategy

Do **NOT** attempt to use "Smart Normalization" (stripping prefixes like `amazon/`) because the directory structures are inconsistent between the two datasets.

Use **Suffix Indexing** instead. This is the only proven, deterministic method.

### Algorithm

1.  **Index the Calibre Manifest (The Target Keys)**
    *   Iterate every `canonical_id` in the Calibre manifest.
    *   Split the ID by `/`.
    *   Store **EVERY POSSIBLE SUFFIX** in a lookup map.
    *   *Example:* For ID `A/B/C`, store:
        *   `"A/B/C" -> "A/B/C"`
        *   `"B/C" -> "A/B/C"`
        *   `"C" -> "A/B/C"` (Filename only)

2.  **Iterate the Master Manifest (The Source Keys)**
    *   For each row, extract the `filename` from the `absolute_image_path`.
        *   `E:\...\page_001.jpg` -> `page_001.jpg`
    *   Look up this filename in the **Suffix Map**.
    *   **IF MATCH FOUND:** You have successfully bridged the Master Image to the Calibre ID.

### Why This Works
*   It ignores all prefix differences (`s3://`, `E:\`, `CalibreComics_extracted/`, `amazon/`).
*   It anchors on the **Leaf Filename** (or the deepest folder structure available), which is almost always preserved across systems.
*   It handles the Amazon dataset (flat filenames) and the Calibre dataset (nested folders) with a single logic path.

## Code Implementation Reference

See `src/version2/train_stage3.py` (the `build_json_map` function) for the canonical implementation:

```python
def build_json_map(s3_manifest_path):
    suffix_map = {}
    with open(s3_manifest_path, 'r') as f:
        for row in csv.DictReader(f):
            cid = row['canonical_id']
            # Index every suffix
            parts = cid.split('/')
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in suffix_map:
                    suffix_map[suffix] = cid
            # Index filename
            filename = parts[-1]
            suffix_map[filename] = cid
    return suffix_map
```

## Critical Warnings

1.  **Do NOT rely on `stem` (stripping extensions) alone:** `0001.jpg` and `0001.png` might be different files in different folders.
2.  **Do NOT hardcode `amazon/` stripping:** The path structure varies. Suffix matching handles this automatically.
3.  **MacOS Junk:** Always filter out `__MACOSX` entries from the Calibre manifest, as they have no corresponding image in the Master manifest.

---
*Created: Jan 2026*
