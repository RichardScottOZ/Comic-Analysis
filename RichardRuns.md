 python benchmarks\captioning_2000ad\generate_captions.py --model florence2 --input-path data\datasets.unify\2000ad\images --output-path data\predicts.captions\2000ad --batch-size 64 --save-txt --save-csv                         

 python benchmarks\captioning_2000ad\generate_captions.py --model minicpm2.6 --input-path data\datasets.unify\2000ad\images --output-path data\predicts.captions\2000ad --batch-size 16 --save-txt --save-csv                      

 python benchmarks\detections_2000ad\yolov8.py --input-path data\datasets.unify\2000ad\images --output-path data\predicts.coco\2000ad\yolo-mix\predictions.json --weights-path benchmarks\weights\yolov8 --weights-name yolov8x-mix --batch-size 16 --save-vis 10   

  python benchmarks\detections_2000ad\magi.py --input-path data\datasets.unify\2000ad\images --output-path data\predicts.coco\2000ad\magi\predictions.json --batch-size 16 --save-vis 10
                                                     

 python benchmarks\detections_2000ad\groundingdino.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/grounding-dino/predictions.json --save-vis 10                                             

 python benchmarks\detections_2000ad\groundingdino.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/grounding-dino/predictions.json --batch-size 4 --save-vis 10 --box-threshold 0.3 --text-threshold 0.1                                                    

 python benchmarks\detections_2000ad\faster_rcnn.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json  --save-vis 10 --conf-threshold 0.5                        

python benchmarks\detections_2000ad\faster_rcnn.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json --weights-name faster_rcnn-c100-best --save-vis 10 --conf-threshold 0.5        

python benchmarks\detections_2000ad\faster_rcnn --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json --weights-name faster_rcnn-c100-best --save-vis 10 --conf-threshold 0.5 

python -m benchmarks.detections_2000ad.faster_rcnn --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json --weights-name faster_rcnn-c100-best --save-vis 10 --conf-threshold 0.5                                                       

 python benchmarks\detections_2000ad\dass.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/dass-m109/predictions.json --model-size xl --save-vis 10                                                                                                          

python -m benchmarks.detections_2000ad.dass --input-unify data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/dass-m109/predictions.json --model-size xl --save-vis 10                                     

python -m benchmarks.detections_2000ad.dass --input_unify data/datasets.unify/2000ad/images --output_coco data/predicts.coco/2000ad/dass-m109/predictions.json --model_size xl --save-vis 10                                                                                                         


python -m benchmarks.detections_2000ad.dass --input-unify data/datasets.unify/2000ad/images --output-coco data/predicts.coco/2000ad/dass-m109/predictions.json --model-size xl --save-vis 10                                    

 python -m benchmarks.detections_2000ad.dass --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/dass-m109/predictions.json --model-size xl --save-vis 10                                                                                                          


python -m comix.process.2000ad --input-path 2000AD --output-path data/datasets.unify/2000ad --override           

 python benchmarks/detections/dass.py -n popmanga -s val -pd m109

  python benchmarks/captioning/generate_captions.py --model florence2 --num_splits 4 --index 2 --batch_size 16 --save_txt --save_csv                                                                                                                    

  & "C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python39_64/python.exe" c:/Users/Richard/OneDrive/GIT/CoMix/benchmarks/captioning/generate_captions.py    


## Example batch command
- C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter>python batch_comic_analysis_multi.py --input-dir "C:\Users\Richard\OneDrive\GIT\CoMix\data\datasets.unify\2000ad\images" --max-images 20000  --output-dir gemma3122000adbigtest --model google/gemma-3-12b-it --api-key bananasplitsapikey

## Processing via VLM API
- This is output token variable - input tokens will be fairly consistent for a given resolution for a given model
Gemma 3 bonus is that it does a reduction - it appears that a 1000x1300 or 2000x3000 input images is about the same input tokens
So this makes the input part significantly cheaper using one of this family.
- 4B at openrouter is 0.02/0.04
- 12B at openrouter is 0.03/0.03  - and is the clear choice for cheap testing
- Mistral 3.1 smal 24B is 0.027/0.027 by uses more than double the input tokens

- here's some ballpark

| Model (via OpenRouter) | Avg. Input Tokens (per image) | Avg. Output Tokens (per image) | Image Input Cost | Text Input Cost | Text Output Cost | Total Projected Cost |
|---|---|---|---|---|---|---|
| Gemma 3 12B | 660 | 1,118 | $0.00 | $19.80 | $33.54 | ~$53.34 |
| Gemma 3 27B | 662 | 1,272 | $26.00 | $59.58 | $216.24 | ~$301.82 |
| Kimi VL A3B Thinking | 1,446 | 2,649 | $0.00 | $54.95 | $100.66 | ~$155.61 |
| Mistral Small 3.1 24B | 2,257 | 1,319 | $0.00 | $112.85 | $131.90 | ~$244.75 |
| Mistral Small 3.2 24B | 2,194 | 1,306 | $0.00 | $109.70 | $130.60 | ~$240.30 |
| Phi-4 Multimodal Instruct | 2,453 | 1,273 | $177.00 | $122.65 | $127.30 | ~$426.95 |

### Gemma 3 12B

-- first run
```python
=== Processing Summary ===
Total images: 13098
Successful: 12480
JSON parse errors: 488
Errors: 130
Skipped: 0
Total time: 24457.84 seconds
Average time per image: 1.87 seconds
```

-- second run
```python
=== Processing Summary ===
Total images: 618
Successful: 456
JSON parse errors: 140
Errors: 22
Skipped: 12480
Total time: 2185.99 seconds
Average time per image: 3.54 seconds
```

-- clearly getting harder as the error proportion is increasing - need to analyse the more difficult files it has problems with

-- third run
```python
=== Processing Summary ===
Total images: 162
Successful: 94
JSON parse errors: 62
Errors: 6
Skipped: 12936
Total time: 1239.72 seconds
Average time per image: 7.65 seconds
```

-- fourth run
```python
=== Processing Summary ===
Total images: 68
Successful: 31
JSON parse errors: 32
Errors: 5
Skipped: 13030
Total time: 872.34 seconds
Average time per image: 12.83 seconds
```

## Horizon Beta
- first run
- this model is fast [open ai rumour] 
    - but had twice the json error rate of Gemma 12B!
    - users 1463 input tokens per image as opposed to 660
```python
=== Processing Summary ===
Total images: 13098
Successful: 11929
JSON parse errors: 1168
Errors: 1
Skipped: 0
Total time: 8319.45 seconds
Average time per image: 0.64 seconds
```

- second run
```python
=== Processing Summary ===
Total images: 1169
Successful: 490
JSON parse errors: 679
Errors: 0
Skipped: 11929
Total time: 821.16 seconds
Average time per image: 0.70 seconds
```

- third run
```python
=== Processing Summary ===
Total images: 679
Successful: 175
JSON parse errors: 504
Errors: 0
Skipped: 12419
Total time: 569.34 seconds
Average time per image: 0.84 seconds
```

- fourth run
```python
=== Processing Summary ===
Total images: 504
Successful: 98
JSON parse errors: 406
Errors: 0
Skipped: 12594
Total time: 378.14 seconds
Average time per image: 0.75 seconds
```
- fifth run
```python
=== Processing Summary ===
Total images: 406
Successful: 79
JSON parse errors: 327
Errors: 0
Skipped: 12692
Total time: 318.43 seconds
Average time per image: 0.78 seconds
```

- sixth run
```python
=== Processing Summary ===
Total images: 327
Successful: 36
JSON parse errors: 291
Errors: 0
Skipped: 12771
Total time: 309.22 seconds
Average time per image: 0.95 seconds

```

- seventh run
```python
=== Processing Summary ===
Total images: 291
Successful: 29
JSON parse errors: 262
Errors: 0
Skipped: 12807
Total time: 271.45 seconds
Average time per image: 0.93 seconds
```

- eighth run
```python
=== Processing Summary ===
Total images: 262
Successful: 20
JSON parse errors: 242
Errors: 0
Skipped: 12836
Total time: 254.37 seconds
Average time per image: 0.97 seconds
```

- ninth run
```python
=== Processing Summary ===
Total images: 242
Successful: 13
JSON parse errors: 229
Errors: 0
Skipped: 12856
Total time: 234.17 seconds
Average time per image: 0.97 seconds
```

- tenth run
```python
=== Processing Summary ===
Total images: 229
Successful: 12
JSON parse errors: 217
Errors: 0
Skipped: 12869
Total time: 346.80 seconds
Average time per image: 1.51 seconds
```

- run 11
```python
=== Processing Summary ===
Total images: 217
Successful: 10
JSON parse errors: 207
Errors: 0
Skipped: 12881
Total time: 274.15 seconds
Average time per image: 1.26 seconds
```

- run 12
```python
=== Processing Summary ===
Total images: 207
Successful: 9
JSON parse errors: 198
Errors: 0
Skipped: 12891
Total time: 208.85 seconds
Average time per image: 1.01 seconds
```

- run 13
```python
=== Processing Summary ===
Total images: 198
Successful: 12
JSON parse errors: 186
Errors: 0
Skipped: 12900
Total time: 221.30 seconds
Average time per image: 1.12 seconds
```

- run 14
```python
=== Processing Summary ===
Total images: 186
Successful: 9
JSON parse errors: 177
Errors: 0
Skipped: 12912
Total time: 206.90 seconds
Average time per image: 1.11 seconds
```

- run 15
```python
=== Processing Summary ===
Total images: 177
Successful: 5
JSON parse errors: 172
Errors: 0
Skipped: 12921
Total time: 223.21 seconds
Average time per image: 1.26 seconds
```

- run 16
```python
=== Processing Summary ===
Total images: 172
Successful: 6
JSON parse errors: 166
Errors: 0
Skipped: 12926
Total time: 206.56 seconds
Average time per image: 1.20 seconds
```

- run 17
```python
=== Processing Summary ===
Total images: 166
Successful: 6
JSON parse errors: 160
Errors: 0
Skipped: 12932
Total time: 173.58 seconds
Average time per image: 1.05 seconds
```

- run 18
```python
=== Processing Summary ===
Total images: 160
Successful: 3
JSON parse errors: 157
Errors: 0
Skipped: 12938
Total time: 170.76 seconds
Average time per image: 1.07 seconds
```

- run 19
```python
=== Processing Summary ===
Total images: 157
Successful: 4
JSON parse errors: 153
Errors: 0
Skipped: 12941
Total time: 182.47 seconds
Average time per image: 1.16 seconds
```

- run 20
```python
=== Processing Summary ===
Total images: 153
Successful: 5
JSON parse errors: 148
Errors: 0
Skipped: 12945
Total time: 193.73 seconds
Average time per image: 1.27 seconds
```

- run 21
```python
=== Processing Summary ===
Total images: 148
Successful: 4
JSON parse errors: 144
Errors: 0
Skipped: 12950
Total time: 148.55 seconds
Average time per image: 1.00 seconds
```

- run 22
```python
=== Processing Summary ===
Total images: 144
Successful: 4
JSON parse errors: 140
Errors: 0
Skipped: 12954
Total time: 155.02 seconds
Average time per image: 1.08 seconds
```

- run 23
```python
=== Processing Summary ===
Total images: 140
Successful: 4
JSON parse errors: 136
Errors: 0
Skipped: 12958
Total time: 129.81 seconds
Average time per image: 0.93 seconds
```

- run 24
```python
=== Processing Summary ===
Total images: 136
Successful: 6
JSON parse errors: 130
Errors: 0
Skipped: 12962
Total time: 135.18 seconds
Average time per image: 0.99 seconds
```

- run 25
```python
=== Processing Summary ===
Total images: 130
Successful: 5
JSON parse errors: 125
Errors: 0
Skipped: 12968
Total time: 147.59 seconds
Average time per image: 1.14 seconds
```

- run 26
```python
=== Processing Summary ===
Total images: 125
Successful: 6
JSON parse errors: 119
Errors: 0
Skipped: 12973
Total time: 136.28 seconds
Average time per image: 1.09 seconds
```

- run 27
```python
=== Processing Summary ===
Total images: 125
Successful: 6
JSON parse errors: 119
Errors: 0
Skipped: 12973
Total time: 136.28 seconds
Average time per image: 1.09 seconds
```

- run 28
```python
=== Processing Summary ===
Total images: 119
Successful: 1
JSON parse errors: 118
Errors: 0
Skipped: 12979
Total time: 137.95 seconds
Average time per image: 1.16 seconds
```

- run 29
```python
=== Processing Summary ===
Total images: 119
Successful: 1
JSON parse errors: 118
Errors: 0
Skipped: 12979
Total time: 137.95 seconds
Average time per image: 1.16 seconds
```

- run 30
```python
=== Processing Summary ===
Total images: 118
Successful: 7
JSON parse errors: 111
Errors: 0
Skipped: 12980
Total time: 128.83 seconds
Average time per image: 1.09 seconds
```

- run 31
```python
=== Processing Summary ===
Total images: 111
Successful: 2
JSON parse errors: 109
Errors: 0
Skipped: 12987
Total time: 122.80 seconds
Average time per image: 1.11 seconds
```

- run 32
```python
=== Processing Summary ===
Total images: 109
Successful: 2
JSON parse errors: 107
Errors: 0
Skipped: 12989
Total time: 123.27 seconds
Average time per image: 1.13 seconds
```

- run 33
```python
=== Processing Summary ===
Total images: 107
Successful: 3
JSON parse errors: 104
Errors: 0
Skipped: 12991
Total time: 131.33 seconds
Average time per image: 1.23 seconds
```

- run 34
```python
=== Processing Summary ===
Total images: 104
Successful: 1
JSON parse errors: 103
Errors: 0
Skipped: 12994
Total time: 131.91 seconds
Average time per image: 1.27 seconds
```

- run 35
```python
=== Processing Summary ===
Total images: 103
Successful: 1
JSON parse errors: 102
Errors: 0
Skipped: 12995
Total time: 123.41 seconds
Average time per image: 1.20 seconds
```

- run 36
```python
=== Processing Summary ===
Total images: 102
Successful: 2
JSON parse errors: 100
Errors: 0
Skipped: 12996
Total time: 122.61 seconds
Average time per image: 1.20 seconds
```

- run 37
```python
=== Processing Summary ===
Total images: 100
Successful: 2
JSON parse errors: 98
Errors: 0
Skipped: 12998
Total time: 123.79 seconds
Average time per image: 1.24 seconds
```

- run 38
```python
=== Processing Summary ===
Total images: 98
Successful: 1
JSON parse errors: 97
Errors: 0
Skipped: 13000
Total time: 118.96 seconds
Average time per image: 1.21 seconds
```

- run 39
```python
=== Processing Summary ===
Total images: 98
Successful: 1
JSON parse errors: 97
Errors: 0
Skipped: 13000
Total time: 118.96 seconds
Average time per image: 1.21 seconds
```

- run 40
```python
=== Processing Summary ===
Total images: 97
Successful: 3
JSON parse errors: 94
Errors: 0
Skipped: 13001
Total time: 82.42 seconds
Average time per image: 0.85 seconds
```

- run 41
```python
=== Processing Summary ===
Total images: 94
Successful: 1
JSON parse errors: 93
Errors: 0
Skipped: 13004
Total time: 96.85 seconds
Average time per image: 1.03 seconds
```

- run 42
```python
=== Processing Summary ===
Total images: 94
Successful: 1
JSON parse errors: 93
Errors: 0
Skipped: 13004
Total time: 96.85 seconds
Average time per image: 1.03 seconds
```

- run 43
```python
=== Processing Summary ===
Total images: 93
Successful: 1
JSON parse errors: 92
Errors: 0
Skipped: 13005
Total time: 118.07 seconds
Average time per image: 1.27 seconds
```

- run 44
```python
=== Processing Summary ===
Total images: 92
Successful: 5
JSON parse errors: 87
Errors: 0
Skipped: 13006
Total time: 115.30 seconds
Average time per image: 1.25 seconds
```

- run 45
```python
=== Processing Summary ===
Total images: 87
Successful: 1
JSON parse errors: 86
Errors: 0
Skipped: 13011
Total time: 115.94 seconds
Average time per image: 1.33 seconds
```

- run 46
```python
=== Processing Summary ===
Total images: 86
Successful: 1
JSON parse errors: 85
Errors: 0
Skipped: 13012
Total time: 113.20 seconds
Average time per image: 1.32 seconds
```

- run 47
```python
=== Processing Summary ===
Total images: 85
Successful: 3
JSON parse errors: 82
Errors: 0
Skipped: 13013
Total time: 107.73 seconds
Average time per image: 1.27 seconds
```

- run 48
```python
=== Processing Summary ===
Total images: 82
Successful: 1
JSON parse errors: 81
Errors: 0
Skipped: 13016
Total time: 83.26 seconds
Average time per image: 1.02 seconds
```

- run 49
```python
=== Processing Summary ===
Total images: 81
Successful: 1
JSON parse errors: 80
Errors: 0
Skipped: 13017
Total time: 110.38 seconds
Average time per image: 1.36 seconds
```

- run 50
```python
=== Processing Summary ===
Total images: 80
Successful: 1
JSON parse errors: 79
Errors: 0
Skipped: 13018
Total time: 104.01 seconds
Average time per image: 1.30 seconds
```

- run 51
```python
=== Processing Summary ===
Total images: 79
Successful: 2
JSON parse errors: 77
Errors: 0
Skipped: 13019
Total time: 111.31 seconds
Average time per image: 1.41 seconds
```

- run 52
```python
=== Processing Summary ===
Total images: 77
Successful: 0
JSON parse errors: 77
Errors: 0
Skipped: 13021
Total time: 100.18 seconds
Average time per image: 1.30 seconds
```

- run 53
```python
=== Processing Summary ===
Total images: 77
Successful: 3
JSON parse errors: 74
Errors: 0
Skipped: 13021
Total time: 89.34 seconds
Average time per image: 1.16 seconds
```

- run 54
```python
=== Processing Summary ===
Total images: 74
Successful: 0
JSON parse errors: 74
Errors: 0
Skipped: 13024
Total time: 93.05 seconds
Average time per image: 1.26 seconds
```

- run 55
```python
=== Processing Summary ===
Total images: 74
Successful: 1
JSON parse errors: 73
Errors: 0
Skipped: 13024
Total time: 96.67 seconds
Average time per image: 1.31 seconds
```

- run 56
```python
=== Processing Summary ===
Total images: 73
Successful: 2
JSON parse errors: 71
Errors: 0
Skipped: 13025
Total time: 80.72 seconds
Average time per image: 1.11 seconds
```

- run 57
```python
=== Processing Summary ===
Total images: 71
Successful: 0
JSON parse errors: 71
Errors: 0
Skipped: 13027
Total time: 98.94 seconds
Average time per image: 1.39 seconds
```

- run 58
```python
=== Processing Summary ===
Total images: 71
Successful: 1
JSON parse errors: 70
Errors: 0
Skipped: 13027
Total time: 87.01 seconds
Average time per image: 1.23 seconds
```

- run 59
```python
=== Processing Summary ===
Total images: 70
Successful: 2
JSON parse errors: 68
Errors: 0
Skipped: 13028
Total time: 81.86 seconds
Average time per image: 1.17 seconds
```

- run 60
```python
=== Processing Summary ===
Total images: 68
Successful: 2
JSON parse errors: 66
Errors: 0
Skipped: 13030
Total time: 91.72 seconds
Average time per image: 1.35 seconds
```

- run 61
```python
=== Processing Summary ===
Total images: 66
Successful: 0
JSON parse errors: 66
Errors: 0
Skipped: 13032
Total time: 84.97 seconds
Average time per image: 1.29 seconds
```

- run 62
```python
=== Processing Summary ===
Total images: 66
Successful: 0
JSON parse errors: 66
Errors: 0
Skipped: 13032
Total time: 101.27 seconds
Average time per image: 1.53 seconds
```

- run 63
```python
=== Processing Summary ===
Total images: 66
Successful: 0
JSON parse errors: 66
Errors: 0
Skipped: 13032
Total time: 88.49 seconds
Average time per image: 1.34 seconds

```


python batch_comic_analysis_multi.py --input-dir "E:\amazon" --max-images 2000000 --output-dir "E:\amazon_analysis" --model openrouter/horizon-beta --api-key bananasplitsapikey



### Closure Lite

python train_closure_lite.py     --json_dir "E:\amazon_datacontract-test"     --image_root "E:\amazon"     --output_dir "./closure_lite_output"     --batch_size 4     --epochs 5     --wandb_project "closure-lite-amazon"     --resume "./closure_lite_output/best_checkpoint.pth"

python benchmarks/detections/openrouter/train_closure_lite.py \
    --json_dir "E:/amazon_datacontract-test" \
    --image_root "E:/amazon" \
    --output_dir "benchmarks/detections/openrouter/closure_lite_output" \
    --batch_size 4 \
    --epochs 5 \
    --lr 3e-4 \
    --resume "benchmarks/detections/openrouter/closure_lite_output/best_checkpoint.pth"


### Closure Lite Simple
- prevent attention collapse of other version

```python
 python benchmarks/detections/openrouter/train_closure_lite_simple.py --json_list_file "perfect_match_training/perfect_match_jsons.txt" --image_root "E:\amazon" --output_dir "closure_lite_simple_full_output" --batch_size 4 --epochs 8 --lr 1e-4 --wandb_project "closure-lite-simple-full-extended" --num_heads 4 --temperature 0.1 --resume "closure_lite_simple_full_output/best_checkpoint.pth"    


  python benchmarks/detections/openrouter/train_closure_lite_simple.py --json_list_file "perfect_match_training/perfect_match_jsons.txt" --image_root "E:\amazon" --output_dir "closure_lite_simple_full_output" --batch_size 4 --epochs 5 --lr 1e-4 --wandb_project "closure-lite-simple-full" --num_heads 4 --temperature 0.1
Using device: cuda
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
wandb: You chose "Don't visualize my results"
wandb: Tracking run with wandb version 0.19.9
wandb: W&B syncing is set to `offline` in this directory.
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
Creating dataloader...
Loading JSON list from perfect_match_training/perfect_match_jsons.txt
Using JSON list file: perfect_match_training/perfect_match_jsons.txt
Successfully loaded 212736 JSON paths using latin-1 encoding
Creating dataset with 212736 JSON files
Image root: /mnt/e/amazon
Dataset initialized with 212736 JSON files
Pages will be loaded on-demand during training
Created dataloader with 212736 samples
Creating model with simple framework (no sequence processing)...
Some weights of ViTModel were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Total parameters: 212,469,515
Trainable parameters: 212,469,515
Trainable percentage: 100.0%
/mnt/c/Users/Richard/OneDrive/GIT/CoMix/benchmarks/detections/openrouter/train_closure_lite_simple.py:297: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler()
Starting training with simple framework...
Epoch 1/5
Epoch 1:   0%|                                                                                | 0/53184 [00:00<?, ?it/s]/mnt/c/Users/Richard/OneDrive/GIT/CoMix/benchmarks/detections/openrouter/train_closure_lite_simple.py:130: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
Epoch 1:  19%|▏| 10000/53184 [1:46:36<7:23:38,  1.62it/s, Loss=0.1578, Avg=0.2619, MPM=0.0000, AvgMPM=0.0000, POP=0.0001Saved intermediate checkpoint at batch 10000
Epoch 1:  38%|▍| 20000/53184 [3:36:00<9:24:16,  1.02s/it, Loss=0.2790, Avg=0.2496, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 20000
Epoch 1:  56%|▌| 30000/53184 [5:22:49<4:15:12,  1.51it/s, Loss=0.3577, Avg=0.2361, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 30000
Epoch 1:  75%|▊| 40000/53184 [7:08:38<2:37:20,  1.40it/s, Loss=0.3126, Avg=0.2900, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 40000
Epoch 1:  94%|▉| 50000/53184 [8:54:03<47:53,  1.11it/s, Loss=0.3626, Avg=0.2935, MPM=0.0000, AvgMPM=0.0000, POP=0.0000, Saved intermediate checkpoint at batch 50000
Epoch 1: 100%|█| 53184/53184 [9:27:49<00:00,  1.56it/s, Loss=0.2027, Avg=0.2184, MPM=0.0000, AvgMPM=0.0000, POP=0.0000,
Epoch 1 - Avg Loss: 0.2697 MPM: 0.0000 POP: 0.0036 RPP: 0.5372
New best model saved! Loss: 0.2697
Epoch 2/5
Epoch 2:  19%|▏| 10000/53184 [1:37:23<4:15:55,  2.81it/s, Loss=0.1975, Avg=0.2808, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 10000
Epoch 2:  38%|▍| 20000/53184 [3:24:34<6:18:06,  1.46it/s, Loss=0.0835, Avg=0.2124, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 20000
Epoch 2:  56%|▌| 30000/53184 [5:12:38<2:56:46,  2.19it/s, Loss=0.4726, Avg=0.2604, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 30000
Epoch 2:  75%|▊| 40000/53184 [6:59:48<1:34:00,  2.34it/s, Loss=0.1832, Avg=0.1893, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 40000
Epoch 2:  94%|▉| 50000/53184 [8:45:44<33:24,  1.59it/s, Loss=0.2522, Avg=0.1887, MPM=0.0000, AvgMPM=0.0000, POP=0.0000, Saved intermediate checkpoint at batch 50000
Epoch 2: 100%|█| 53184/53184 [9:19:26<00:00,  1.58it/s, Loss=0.2673, Avg=0.2258, MPM=0.0000, AvgMPM=0.0000, POP=0.0000,
Epoch 2 - Avg Loss: 0.2312 MPM: 0.0000 POP: 0.0000 RPP: 0.4623
New best model saved! Loss: 0.2312
Epoch 3/5
Epoch 3:  19%|▏| 10000/53184 [1:37:44<6:29:14,  1.85it/s, Loss=0.1203, Avg=0.1730, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 10000
Epoch 3:  38%|▍| 20000/53184 [3:24:02<5:56:02,  1.55it/s, Loss=0.1321, Avg=0.1594, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 20000
Epoch 3:  56%|▌| 30000/53184 [5:12:56<6:15:44,  1.03it/s, Loss=0.0008, Avg=0.1174, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 30000
Epoch 3:  75%|▊| 40000/53184 [6:58:11<2:35:25,  1.41it/s, Loss=0.2314, Avg=0.2419, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 40000
Epoch 3:  94%|▉| 50000/53184 [8:42:22<54:19,  1.02s/it, Loss=0.4444, Avg=0.2080, MPM=0.0000, AvgMPM=0.0000, POP=0.0000, Saved intermediate checkpoint at batch 50000
Epoch 3: 100%|█| 53184/53184 [9:15:30<00:00,  1.60it/s, Loss=0.0470, Avg=0.1420, MPM=0.0000, AvgMPM=0.0000, POP=0.0000,
Epoch 3 - Avg Loss: 0.2077 MPM: 0.0000 POP: 0.0000 RPP: 0.4154
New best model saved! Loss: 0.2077
Epoch 4/5
Epoch 4:  19%|▏| 10000/53184 [1:36:11<10:17:40,  1.17it/s, Loss=0.1288, Avg=0.1664, MPM=0.0000, AvgMPM=0.0000, POP=0.000Saved intermediate checkpoint at batch 10000
Epoch 4:  38%|▍| 20000/53184 [3:22:53<4:49:19,  1.91it/s, Loss=0.1687, Avg=0.1819, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 20000
Epoch 4:  56%|▌| 30000/53184 [5:10:35<5:12:30,  1.24it/s, Loss=0.0663, Avg=0.1604, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 30000
Epoch 4:  75%|▊| 40000/53184 [7:01:01<1:48:35,  2.02it/s, Loss=0.1610, Avg=0.1960, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 40000
Epoch 4:  94%|▉| 50000/53184 [8:54:02<36:45,  1.44it/s, Loss=0.2190, Avg=0.2320, MPM=0.0000, AvgMPM=0.0000, POP=0.0000, Saved intermediate checkpoint at batch 50000
Epoch 4: 100%|█| 53184/53184 [9:29:27<00:00,  1.56it/s, Loss=0.2827, Avg=0.1880, MPM=0.0000, AvgMPM=0.0000, POP=0.0000,
Epoch 4 - Avg Loss: 0.1806 MPM: 0.0000 POP: 0.0001 RPP: 0.3611
New best model saved! Loss: 0.1806
Epoch 5/5
Epoch 5:  19%|▏| 10000/53184 [1:42:17<8:25:30,  1.42it/s, Loss=0.1029, Avg=0.1296, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 10000
Epoch 5:  38%|▍| 20000/53184 [3:27:18<4:44:56,  1.94it/s, Loss=0.1038, Avg=0.1356, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 20000
Epoch 5:  56%|▌| 30000/53184 [5:12:32<3:06:20,  2.07it/s, Loss=0.2238, Avg=0.1807, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 30000
Epoch 5:  75%|▊| 40000/53184 [6:57:24<1:52:58,  1.95it/s, Loss=0.2544, Avg=0.1718, MPM=0.0000, AvgMPM=0.0000, POP=0.0000Saved intermediate checkpoint at batch 40000
Epoch 5:  94%|▉| 50000/53184 [8:42:17<32:53,  1.61it/s, Loss=0.4480, Avg=0.2127, MPM=0.0000, AvgMPM=0.0000, POP=0.0000, Saved intermediate checkpoint at batch 50000
Epoch 5: 100%|█| 53184/53184 [9:15:23<00:00,  1.60it/s, Loss=0.2308, Avg=0.1743, MPM=0.0000, AvgMPM=0.0000, POP=0.0000,
Epoch 5 - Avg Loss: 0.1508 MPM: 0.0000 POP: 0.0000 RPP: 0.3015
New best model saved! Loss: 0.1508
```