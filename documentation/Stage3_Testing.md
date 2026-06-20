
# STAGE 3 MODELLING
## TRAINING
```python
python -u src\version2\train_stage3_vlm.py --manifest manifests\master_manifest_20251229.csv --vlm_cache_dir  E:\vlm_cache --train_pss_labels train_pss.json --val_pss_labels val_pss.json --epochs 20 --batch_size 8  --freeze_backbones --checkpoint_dir checkpoints\stage3_vlm  --num_workers 8    
```

## TODO


 #### Step A: Generate the Zarr Embeddings

  Run generate_stage3_embeddings_vlm.py under the  caption3  environment:

    conda run -n caption3 python src/version2/generate_stage3_embeddings_vlm.py `
      --manifest manifests/master_manifest_20251229.csv `
      --vlm_cache_dir E:\vlm_cache `
      --pss_labels pss_labels_v1.json `
      --checkpoint checkpoints/stage3_vlm/best_model_vlm.pt `
      --output_zarr E:/Comic_Analysis_Results_v2/stage3_embeddings_vlm.zarr `
      --output_metadata E:/Comic_Analysis_Results_v2/stage3_metadata_vlm.json `
      --batch_size 16 `
      --num_workers 8

  #### Step B: Run the Sanity Check/Verification (Safe & Lightweight)

  The verification script verify_stage3_embeddings.py does not import PyTorch, so it is safe to run while keeping system
  utilization low:

    conda run -n caption3 python src/tools/verify_stage3_embeddings.py `
      --zarr E:/Comic_Analysis_Results_v2/stage3_embeddings_vlm.zarr `
      --metadata E:/Comic_Analysis_Results_v2/stage3_metadata_vlm.json
    ──────