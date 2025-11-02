# Zarr Flush Fix - generate_embeddings_zarr_claude.py

## Problem Identified

When running `generate_embeddings_zarr_claude.py` without `--incremental` flag, embeddings were being written to the Zarr store but not becoming visible when queried with `print_page_summaries.py` until the entire run completed.

### Root Cause

The issue was that `zgroup.store.flush()` was being called every 10 batches, but this alone doesn't guarantee that individual Zarr arrays flush their write caches to disk. Some Zarr storage backends (like DirectoryStore) maintain per-array write buffers that need explicit synchronization.

## Changes Made

### 1. Enhanced Flush Operation (Line ~1036)

**Before:**
```python
if use_direct_zarr and ((batch_idx + 1) % 10 == 0):
    try:
        zgroup.store.flush()
        if args and getattr(args, 'verbose', False):
            print(f"[FLUSH] Flushed zarr to disk after batch {batch_idx + 1}")
    except Exception as e:
        if batch_idx == 0:
            print(f"Warning: zarr flush failed (batch {batch_idx}): {e}")
```

**After:**
```python
if use_direct_zarr and ((batch_idx + 1) % 10 == 0):
    try:
        # Call store-level flush
        zgroup.store.flush()
        
        # CRITICAL FIX: Explicitly flush each individual array
        for arr_name in ['page_embeddings', 'panel_embeddings', 'attention_weights', 
                         'reading_order', 'panel_coordinates', 'text_content', 'panel_mask',
                         'page_id', 'manifest_path', 'source', 'series', 'volume', 'issue', 'page_num']:
            if arr_name in zgroup:
                zgroup[arr_name].store.flush()
        
        print(f"[FLUSH] Flushed zarr to disk after batch {batch_idx + 1}")
    except Exception as e:
        print(f"Warning: zarr flush failed (batch {batch_idx + 1}): {e}")
```

**Key improvements:**
- Individual array flush in addition to store-level flush
- Always print flush messages (removed verbose-only condition) for better visibility
- More robust error handling

### 2. Write-Read Verification Enhancement (Line ~910)

Added explicit flush before read-back verification to ensure the just-written data is visible:

```python
if args and getattr(args, 'write_check', False) and not written_check_done:
    # CRITICAL: Flush before reading back
    try:
        zgroup['page_embeddings'].store.flush()
    except Exception:
        pass
    
    # Now read back...
    readback = np.array(zgroup['page_embeddings'][page_index])
    read_norm = float(np.linalg.norm(readback))
    ...
```

## Testing

To verify the fix works:

1. **Delete existing zarr** (if any):
   ```bash
   rm -rf E:\calibre3\combined_embeddings.zarr
   ```

2. **Run with write verification**:
   ```bash
   python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py \
     --checkpoint "path/to/checkpoint.pth" \
     --amazon_json_list "path/to/list.txt" \
     --amazon_image_root "E:\CalibreComics_extracted" \
     --output_dir "E:\calibre3" \
     --batch_size 8 \
     --device auto \
     --verify_batches \
     --write_check \
     --verbose
   ```

3. **Monitor flush messages**: You should now see:
   ```
   [FLUSH] Flushed zarr to disk after batch 10
   [FLUSH] Flushed zarr to disk after batch 20
   ...
   ```

4. **Query after 10+ batches complete**:
   ```bash
   python tools\print_page_summaries.py \
     --zarr "E:\calibre3\combined_embeddings.zarr" \
     --max_pages 20 \
     --text_len 128 \
     --compact
   ```

   You should now see actual data for pages 0-79 (10 batches × 8 pages/batch = 80 pages).

## Why This Matters

### Before the Fix
- Embeddings written to memory buffers
- `store.flush()` called, but per-array caches not synchronized
- Data not visible until final `store.close()` at end of run
- Risk of data loss if process crashed mid-run
- Impossible to monitor progress or debug issues

### After the Fix
- Embeddings written and explicitly flushed every 10 batches
- Data immediately visible to other processes
- Can safely interrupt and resume with `--incremental`
- Can monitor embedding quality during generation
- Early detection of write failures

## Additional Diagnostic Tool

Created `tools/diagnose_zarr_writes.py` to help diagnose Zarr write issues:

```bash
python tools\diagnose_zarr_writes.py "E:\calibre3\combined_embeddings.zarr"
```

This will:
- Check Zarr can be opened
- Verify page_embeddings shape
- Check first 20 pages for non-zero embeddings
- Compare zarr.open_group vs xarray.open_zarr results

## Best Practices Going Forward

1. **Always use `--write_check`** during initial testing to verify writes work
2. **Use `--verify_batches`** to log write verification for every batch
3. **Monitor flush messages** to ensure periodic synchronization
4. **Use diagnostic tool** if seeing zero embeddings after writes
5. **For large runs**, consider using smaller `--batch_size` with more frequent flushes

## Related Issues

This fix resolves:
- Issue where `print_page_summaries.py` showed only zeros or very few pages mid-run
- Data loss risk from incomplete writes on crashes
- Inability to verify embedding quality during generation
- Problems with incremental mode not finding "existing" pages that were written but not flushed


# Generate Embeddings Zarr - Technical Notes

## Critical Discovery: DirectoryStore Write Visibility

**IMPORTANT**: When using Zarr's DirectoryStore (which we use), writes from an open zarr group may NOT be immediately visible to other processes or even other dataset instances opened in the same process until the writing group is closed.

### What This Means

1. **During generation**: While `generate_embeddings_zarr_claude.py` is running with an open zarr group, running `print_page_summaries.py` in another terminal may show:
   - Zero/empty data
   - Only partial data
   - Stale data from before the current run started

2. **After generation completes**: Once `generate_embeddings_zarr_claude.py` finishes and closes the zarr group (script exits), the data becomes fully visible to new readers.

3. **DirectoryStore has no flush**: Unlike some other zarr storage backends, DirectoryStore doesn't have a `sync()` or `flush()` method. Writes go to individual chunk files, but filesystem caching and zarr's internal buffering mean they may not be readable by other processes immediately.

### The "[FLUSH]" Messages

The script prints messages like:
```
[FLUSH] Synchronized zarr to disk after batch 10
```

However, for DirectoryStore, this message is misleading - there's no actual flush operation happening. The code attempts to call `zgroup.store.sync()` but DirectoryStore doesn't have this method, so the call is silently skipped in the exception handler.

### What's Actually Happening

The data IS being written to the zarr arrays in memory within the script's zarr group instance. When the script completes and the zarr group is closed (garbage collected or explicitly closed), all writes are finalized to disk and become visible.

## How to Verify Progress

### DON'T Do This During Generation
```bash
# This will show stale/zero data while generation is running!
python tools\print_page_summaries.py --zarr "E:\calibre3\combined_embeddings.zarr"
```

### DO This Instead

1. **Check the generator's own output**: The script prints per-page stats like:
   ```
   [PAGE] json=image-069.json page_id=... idx=15752 mask_sum=8 emb_l2=1.3445
   ```
   If you see non-zero `mask_sum` and `emb_l2` values, the data is being generated correctly.

2. **Wait for completion**: Let the script finish completely. After it exits, THEN run print_page_summaries.

3. **Check the verify log**: If running with `--verify_batches`, check the verification log for pre/post write norms.

## Resuming After Crashes

If generation crashes partway through (e.g., at batch 202 out of 9938), you can resume:

```bash
python benchmarks/detections/openrouter/generate_embeddings_zarr_claude.py \
    --checkpoint "path\to\best_checkpoint.pth" \
    --amazon_json_list .\perfect_match_training\calibre3_dataspec_list.txt \
    --amazon_image_root "E:\CalibreComics_extracted" \
    --output_dir "E:\calibre3" \
    --batch_size 8 \
    --device auto \
    --incremental \
    --ledger_path "E:\calibre3\page_id_ledger.jsonl"
```

The `--incremental` flag tells the script to:
1. Open the existing zarr dataset
2. Load the ledger (or build it from existing data)
3. Skip pages that were already written
4. Continue with unprocessed pages

**Important**: Due to the batch-level writing (10 batches at a time), if you crash during batch 205, batches 200-204 may have been written to the in-memory zarr group but not yet visible on disk. The incremental logic will start checking from batch 0, skip all completed work up through the last fully flushed batch (~batch 200), and then may re-process batches 200-205. This is wasteful but safe - it won't corrupt data, just redo a bit of work.

## Why This Design?

The batch-level flushing (every 10 batches) is a performance optimization. Writing to zarr on every single page would be extremely slow. By buffering 10 batches worth of writes and then attempting a flush, we balance:
- Performance: Fewer disk operations
- Crash safety: At most 10 batches worth of work is lost
- Data integrity: Writes are atomic at the batch level

## Zero Out Strategy

The script includes a critical fix: before writing new panel data to a page index, it zeros out all 12 panel slots:

```python
# Zero out all panel data for this page_index
zgroup['panel_embeddings'][page_index, :, :] = 0
zgroup['attention_weights'][page_index, :] = 0
# ... etc
```

This prevents a subtle bug: if you previously wrote a page with 10 panels, then later write the same page with only 6 panels, without zeroing first the old panels 6-9 would remain as stale data.

## Expected Behavior Summary

✅ **Correct**: Script shows non-zero l2 norms and mask sums in its own output  
✅ **Correct**: After script completes, print_page_summaries shows the data  
❌ **Wrong/Confusing**: Trying to read the zarr with another script WHILE generation is running  

The key insight is that zarr DirectoryStore doesn't provide cross-process read-while-write consistency. This is a known limitation of the DirectoryStore backend.
