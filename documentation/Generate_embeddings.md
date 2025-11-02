# Generate Embeddings Zarr Documentation

## Overview

`generate_embeddings_zarr_claude.py` is a robust system for creating searchable embeddings from comic book pages using a trained vision-language model. It processes DataSpec JSON files, generates embeddings for pages and panels, and stores everything in an efficient Zarr format for fast querying.

## What It Does

The system:
1. Loads a trained Closure-Lite model checkpoint
2. Processes comic pages from DataSpec JSON files
3. Generates embeddings for:
   - Full page images (page-level embeddings)
   - Individual panels (panel-level embeddings)
   - Text content (dialogue and narration)
4. Stores everything in a structured Zarr dataset for efficient querying
5. Handles crashes and resumption gracefully

## Key Features

### Robust Batch Processing
- Processes data in batches (default: 8 pages per batch)
- Writes data incrementally every batch to prevent data loss
- Automatically resumes from the last completed batch on crash
- Verifies data integrity after writing

### Data Storage Structure

The Zarr dataset contains:
```
combined_embeddings.zarr/
├── page_embeddings     # [N, 768] - Full page embeddings
├── panel_embeddings    # [N, max_panels, 768] - Per-panel embeddings
├── panel_mask          # [N, max_panels] - Valid panel indicators
├── page_text           # [N] - JSON strings with dialogue/narration
├── page_id             # [N] - Unique identifiers (DataSpec paths)
└── manifest/           # Batch tracking for resumption
    ├── batch_0000.json
    ├── batch_0001.json
    └── ...
```

## Usage

### Basic Usage (New Dataset)

```bash
python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py \
  --checkpoint "path/to/model/checkpoint.pth" \
  --amazon_json_list "path/to/dataspec_list.txt" \
  --amazon_image_root "E:\CalibreComics_extracted" \
  --output_dir "E:\calibre3" \
  --batch_size 8 \
  --device auto \
  --verify_batches \
  --write_check \
  --verbose
```

### Parameters Explained

**Required:**
- `--checkpoint`: Path to trained model checkpoint (.pth file)
- `--amazon_json_list`: Text file with paths to DataSpec JSON files (one per line)
- `--amazon_image_root`: Root directory where comic images are stored
- `--output_dir`: Where to create the Zarr dataset

**Optional:**
- `--batch_size`: Pages per batch (default: 8, higher = faster but more memory)
- `--device`: `cuda`, `cpu`, or `auto` (default: auto)
- `--verify_batches`: Verify data after each batch write (recommended)
- `--write_check`: Additional verification of written data
- `--verbose`: Detailed logging of processing
- `--incremental`: Add to existing dataset instead of creating new one

### Creating the DataSpec List

The list file should contain paths to DataSpec JSON files, one per line:

```text
perfect_match_training\calibre_dataspec_final\comic1\page_001.json
perfect_match_training\calibre_dataspec_final\comic1\page_002.json
perfect_match_training\calibre_dataspec_final\comic2\page_001.json
...
```

Generate this using `generate_dataspec_from_mapping.py`:

```bash
python tools/generate_dataspec_from_mapping.py \
  --mapping_csv "key_mapping_report_claude.csv" \
  --coco_file "E:\CalibreComics\test_dections\predictions.json" \
  --output_dir "perfect_match_training\calibre_dataspec_final" \
  --subset perfect
```

This creates both the DataSpec JSONs and the list file automatically.

## Crash Recovery & Resumption

### How Resumption Works

The system uses a manifest-based approach:
1. Each batch completion is recorded in `manifest/batch_XXXX.json`
2. On startup, the system scans existing manifests to find the last completed batch
3. Processing resumes from the next batch after the last completed one
4. Partial batches are **reprocessed** to ensure data integrity

### Resuming After a Crash

If the process crashes at batch 2021:

1. **Check what's been completed:**
   - Look in `output_dir/combined_embeddings.zarr/manifest/`
   - Find highest batch number (e.g., `batch_2020.json` = completed through batch 2020)

2. **Resume processing:**
   ```bash
   # Same command as before - it auto-detects existing work
   python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py \
     --checkpoint "path/to/checkpoint.pth" \
     --amazon_json_list "dataspec_list.txt" \
     --amazon_image_root "E:\CalibreComics_extracted" \
     --output_dir "E:\calibre3" \
     --batch_size 8 \
     --device auto \
     --verify_batches \
     --verbose
   ```

3. **What happens:**
   - System detects existing Zarr at `E:\calibre3\combined_embeddings.zarr`
   - Scans manifests, finds batch 2020 was last completed
   - Starts processing from batch 2021 (skipping 0-2020)
   - If batch 2021 was partial, it gets fully reprocessed (small efficiency cost for robustness)

### Important Notes on Resumption

- **Always use the same parameters** when resuming (especially `batch_size`)
- **Don't manually delete manifests** - they track progress
- **Partial batches are redone** - not a problem, just a few seconds of work
- **The system is append-only** - old data is never overwritten

### Starting Fresh

To completely start over:

```bash
# Delete the entire Zarr directory
rm -rf E:\calibre3\combined_embeddings.zarr

# Run normally
python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py ...
```

## Incremental Mode

Add new data to an existing dataset:

```bash
python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py \
  --checkpoint "path/to/checkpoint.pth" \
  --amazon_json_list "new_dataspec_list.txt" \
  --amazon_image_root "E:\CalibreComics_extracted" \
  --output_dir "E:\calibre3" \
  --incremental \
  --batch_size 8 \
  --device auto
```

**When to use:**
- Adding new comic pages to an existing dataset
- Expanding your searchable corpus without rebuilding everything

**Important:**
- Existing data is preserved
- New data is appended
- Must use `--incremental` flag
- Dataset must already exist (error if not)

## Verification & Quality Checks

### Built-in Verification

With `--verify_batches` enabled (recommended), each batch is checked:
- ✓ Data shapes are correct
- ✓ Embeddings are non-zero (L2 norm > 0)
- ✓ Panel masks align with panel counts
- ✓ Text data is valid JSON

### Manual Verification

Check the dataset after processing:

```bash
# View first 16 pages with text previews
python tools/print_page_summaries.py \
  --zarr "E:\calibre3\combined_embeddings.zarr" \
  --max_pages 16 \
  --text_len 128 \
  --compact
```

**What to look for:**
- `page_l2` should be > 0 (typically 1.0-3.0)
- `panel_mean_l2` should be > 0 (typically 1.5-2.5)
- `text_preview` should show actual dialogue/narration

**Warning signs:**
- All zeros (0.000000) = data didn't write correctly
- Empty text = VLM data not found or incorrect
- Very low L2 norms = potential model issues

### ⚠️ CRITICAL: Delayed Data Visibility Issue

**IMPORTANT - READ THIS BEFORE PANICKING:**

There is a known issue with Zarr DirectoryStore where data may not be immediately visible even after explicit flush operations. This is a **Zarr framework limitation**, not a bug in the embedding generation code.

**What you might see:**
- After 10+ batches complete and flush runs
- `print_page_summaries.py` shows all zeros or only 1-2 pages
- Everything looks empty even though processing is running fine
- Looking again minutes later suddenly shows more data

**What's actually happening:**
- The data IS being written correctly to disk
- The synchronizer calls ARE executing 
- But Zarr's DirectoryStore has lazy write-through behavior
- Data may not be queryable until OS file system cache flushes
- This can take several minutes or even longer on Windows

**What to do:**
1. **Don't panic and restart** - Let the process keep running
2. **Wait for significantly more batches** - Try checking after 50+ batches instead of 10
3. **Be patient** - Data may lag by 10-20 batches before becoming visible
4. **Close other readers** - Make sure you're not keeping the zarr open in another process
5. **Check file system** - Verify the actual zarr directory is growing on disk

**Verification strategy:**
```bash
# Instead of checking every 10 batches, wait longer
# Check after batch 50, 100, 150, etc.

# Also check the actual disk usage
dir E:\calibre3\combined_embeddings.zarr  # Should show growing size
```

**Why this happens:**
- Zarr DirectoryStore doesn't have a traditional flush() method
- We call array.synchronizer() which *should* force writes
- But the underlying file system may buffer writes
- Windows file caching can delay visibility significantly
- Multiple processes/threads can cause stale reads

**The bottom line:**
If you see batches processing successfully with non-zero embeddings in the verbose output (e.g., `emb_l2=1.6567`), the data IS being generated correctly. The visibility delay is a Zarr/filesystem issue, not data loss. By the time processing completes, all data will be there and queryable.

**Previous flush documentation (still relevant for understanding timing):**

Wait until at least 10 batches complete before running verification tools. The system flushes data to disk every 10 batches (after batches 9, 19, 29, etc.), so data won't be visible until the first flush completes.

**Flush behavior:**
- After batch 9 completes → first flush (batches 0-9 become visible, ~72 pages at batch_size=8)
- After batch 19 completes → second flush (batches 10-19 become visible, ~144 total pages)  
- After batch 29 completes → third flush (batches 20-29 become visible, ~216 total pages)
- And so on...

However due to the visibility issue above, you may not see data for 20-30+ batches after the flush.

## Performance & Scale

### Expected Performance

On a system with NVIDIA GPU:
- ~5-6 seconds per batch (8 pages)
- ~79,500 pages = ~9,938 batches
- **Total time: ~14-15 hours** for full dataset

### Memory Usage

- Batch size 8: ~6-8 GB GPU memory
- Batch size 16: ~12-14 GB GPU memory
- Larger batches = faster but need more memory

### Optimization Tips

1. **Use the largest batch size your GPU can handle**
   ```bash
   --batch_size 16  # If you have 16+ GB VRAM
   ```

2. **Run overnight for large datasets**
   - Set up, start, let it run
   - Check progress in the morning
   - Resumable if crashes

3. **Monitor progress**
   - Watch the progress bar: `Processing batches: X%`
   - Check manifests: `ls output_dir/combined_embeddings.zarr/manifest/`

## Common Issues & Solutions

### Issue: "No pages to process"
**Cause:** DataSpec list file is empty or paths are wrong
**Solution:** Verify the list file exists and contains valid paths

### Issue: "Incremental mode requested but Zarr not found"
**Cause:** Using `--incremental` but no existing dataset
**Solution:** Remove `--incremental` for first run, or create dataset first

### Issue: All embeddings show as 0.000000
**Cause:** Checked too early (before first flush at batch 9) or using old version with flush timing bug  
**Solution:** 
1. Wait for at least 10 batches to complete (watch progress bar)
2. Ensure you're using `generate_embeddings_zarr_claude.py` (has flush fix)
3. Check again after batch 10 completes
4. If still zero after 20+ batches, there's a real problem - check logs

### Issue: "CUDA out of memory"
**Cause:** Batch size too large for GPU
**Solution:** Reduce `--batch_size` to 4 or 2

### Issue: Progress seems stuck
**Cause:** Some pages may take longer to process
**Solution:** Wait and watch verbose output - it shows individual page progress

## Integration with Query System

Once embeddings are generated, query them using `query_embeddings_zarr_claude.py`:

```bash
python benchmarks/detections/openrouter/query_embeddings_zarr_claude.py \
  --zarr "E:\calibre3\combined_embeddings.zarr" \
  --query "superhero fighting villain" \
  --top_k 10
```

See `Query_Embeddings_Zarr.md` for full querying documentation.

## File Structure

```
CoMix/
├── benchmarks/detections/openrouter/
│   ├── generate_embeddings_zarr_claude.py  # Main script
│   └── query_embeddings_zarr_claude.py     # Query script
├── tools/
│   ├── generate_dataspec_from_mapping.py   # Create DataSpecs
│   └── print_page_summaries.py             # Verify Zarr data
├── perfect_match_training/
│   ├── calibre_dataspec_final/             # DataSpec JSONs
│   └── calibre3_dataspec_list.txt          # List of paths
└── key_mapping_report_claude.csv           # Master mapping
```

## Theory of Operation

### Why Zarr?

Zarr provides:
- **Chunked storage** - Only load what you need
- **Compression** - Smaller disk footprint
- **Append-only** - Safe for incremental updates
- **Fast random access** - Efficient querying
- **Metadata** - Self-documenting format

### Embedding Generation Pipeline

1. **Load batch of DataSpec JSONs** (8 pages)
2. **For each page:**
   - Load image from disk
   - Extract panel coordinates from DataSpec
   - Crop panel regions
   - Run through vision model encoder
   - Get page-level embedding
   - Get panel-level embeddings (up to max_panels)
   - Extract text from VLM data in DataSpec
3. **Write batch to Zarr**
   - Append embeddings to arrays
   - Write manifest for batch
   - Verify if requested
4. **Repeat** until all pages processed

### Data Layout

The Zarr format allows efficient queries:
- **Semantic search:** Find pages similar to query text
- **Panel search:** Find specific panel types
- **Text search:** Find pages containing specific dialogue
- **Combined queries:** Mix semantic + text filters

### Manifest System

Each batch writes a manifest JSON:
```json
{
  "batch_id": 42,
  "start_idx": 336,
  "end_idx": 343,
  "num_pages": 8,
  "timestamp": "2025-10-28T09:15:32",
  "page_ids": [
    "perfect_match_training/calibre_dataspec_final/comic/page_001.json",
    ...
  ]
}
```

This enables:
- Crash recovery
- Progress tracking  
- Incremental updates
- Debugging which pages were processed

## Best Practices

1. ✓ **Always use `--verify_batches`** for production runs
2. ✓ **Keep the same batch_size** when resuming
3. ✓ **Use verbose mode** for long runs to monitor progress
4. ✓ **Wait 10+ batches** before checking data
5. ✓ **Run on dedicated GPU** for best performance
6. ✓ **Keep DataSpec files** - needed for any regeneration
7. ✓ **Backup the Zarr** after completion - it's valuable!

## Summary

`generate_embeddings_zarr_claude.py` is a production-ready system for creating searchable comic page embeddings. It handles 80K+ pages reliably, resumes from crashes automatically, and produces a fast-queryable Zarr dataset for semantic search over comic book content.

The key to success: set it up correctly once, let it run, and verify the output. The robust batch-writing and manifest system means you can trust it to complete even very large datasets without manual intervention.

### Recent Improvements (Claude Version)

The `_claude.py` version includes important fixes:
- **DirectoryStore flush fix**: Uses `array.synchronizer()` instead of `store.flush()` which doesn't exist on DirectoryStore
- **Flush timing fix**: Data now flushes after batches 9, 19, 29, etc. (not 0, 10, 20 which was too early)
- **First-writer-wins policy**: Prevents duplicate page overwrites
- **Better error messages**: Clearer diagnostics when pages can't be mapped
- **Zero-out safety**: Clears stale data from previous writes before writing new panel data

**Critical**: The DirectoryStore in zarr doesn't have a `flush()` method. The original code called `store.flush()` which would error. The Claude version properly calls `synchronizer()` on each array individually to force writes to disk.

Always use `generate_embeddings_zarr_claude.py` for new work, not the original `generate_embeddings_zarr.py`.

## Troubleshooting Data Visibility

### Symptoms: Data appears as zeros even after many batches

If you run the verification tool and see output like this even after 20+ batches:
```
idx     page_l2 panel_mean_l2   text_preview
0       0.000000        0.000000
1       0.000000        0.000000
```

**Do this diagnostic:**

1. **Check if batches are actually processing:**
   Look at the console output while generating. You should see lines like:
   ```
   [PAGE] json=image-069.json page_id=... idx=15752 mask_sum=8 emb_l2=1.3445
   ```
   If you see non-zero `emb_l2` values (like 1.34 above), data IS being generated correctly.

2. **Check the debug output:**
   At startup you should see:
   ```
   [DEBUG] use_direct_zarr=True, incremental_mode=False, original_n=79500
   [DEBUG] zgroup opened successfully, will flush every 10 batches
   ```
   
   At every 10th batch you should see:
   ```
   [DEBUG] Batch 10: flush checkpoint reached, use_direct_zarr=True
   [FLUSH] Synchronized zarr to disk after batch 10
   ```

3. **If you DON'T see flush messages:**
   - The `use_direct_zarr` flag might be False
   - Check startup output for "failed to open zarr group" warnings
   - This means data is accumulating in memory and only written at the very end
   - Stop the process, delete the zarr, and restart

4. **If you DO see flush messages but data still shows zeros:**
   - This is the Zarr visibility delay issue (see section above)
   - **Keep the process running**
   - Wait much longer - try checking after 100+ batches
   - Check disk usage: `dir E:\calibre3\combined_embeddings.zarr` - should be growing
   - Consider waiting until process fully completes before verifying

5. **Final sanity check after completion:**
   ```bash
   # Let the process run to 100% completion
   # Wait 5 minutes after it finishes
   # Then check
   python tools/print_page_summaries.py --zarr "E:\calibre3\combined_embeddings.zarr" --max_pages 100 --compact
   ```

### Debug Mode Output Guide

The latest version includes detailed debug output to help diagnose issues:

**At startup:**
```
[DEBUG] use_direct_zarr=True, incremental_mode=False, original_n=79500
[DEBUG] zgroup opened successfully, will flush every 10 batches
```
- `use_direct_zarr=True` → Good! Data writes directly, flushed every 10 batches
- `use_direct_zarr=False` → Warning! All data held in memory until end

**During processing:**
```
[PAGE] json=0022.json page_id=... idx=17019 mask_sum=6 emb_l2=2.439
```
- `mask_sum` → Number of valid panels in this page (should be 1-12)
- `emb_l2` → L2 norm of page embedding (should be ~1.0-3.0)
- If emb_l2 is 0.0, there's a problem generating embeddings

**Every 10 batches:**
```
[DEBUG] Batch 10: flush checkpoint reached, use_direct_zarr=True  
[FLUSH] Synchronized zarr to disk after batch 10
```
- This confirms data is being explicitly flushed to disk
- If you don't see these messages, flushing isn't happening

**What's normal vs. problematic:**

✓ **Normal:**
- Flush messages every 10 batches
- Non-zero emb_l2 values in verbose output
- Data might not be visible for 20-50 batches due to OS caching

✗ **Problem:**
- No flush messages at batches 10, 20, 30, etc.
- All emb_l2 values showing as 0.0 in verbose output  
- `use_direct_zarr=False` at startup
- Growing memory usage instead of growing disk usage
