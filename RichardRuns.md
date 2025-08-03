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
```
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
```
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
```
=== Processing Summary ===
Total images: 172
Successful: 6
JSON parse errors: 166
Errors: 0
Skipped: 12926
Total time: 206.56 seconds
Average time per image: 1.20 seconds
```

run 17
```

```

