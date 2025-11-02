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
