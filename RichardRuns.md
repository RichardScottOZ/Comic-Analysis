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


## Processing via VLM API
- Gemma 3 12B

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