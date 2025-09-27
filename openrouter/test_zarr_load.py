import xarray
import zarr
import os

ds = xarray.open_zarr("test_embeddings.zarr")

print(ds)

print(ds['panel_embeddings'].mean(dim=['page_id', 'panel_id']))