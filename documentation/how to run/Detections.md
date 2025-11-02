python benchmarks\detections_2000ad\faster_rcnn.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json  --save-vis 10 --conf-threshold 0.5         

python benchmarks\detections_2000ad\faster_rcnn.py --input-path "E:\CalibreComics_extracted\sex vol1 - Unknown" --output-path E:\CalibreComics\test_dections\sexpredictions.json  --save-vis 196 --conf- threshold 0.5         

3) C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv>python 02_comixs_prod_detect_HF.py --root "E:\CalibreComics_extracted\PRG1795 - Unknown" --out E:\CalibreComics\test_detections --viz

## coco dataset
(caption3) C:\Users\Richard\OneDrive\GIT\CoMix>python benchmarks\detections\openrouter\coco_to_dataspec.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test


python benchmarks\detections\openrouter\coco_to_dataspect_test.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10

# Test with 10 images, 8 workers
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10 --workers 8

# Full processing with all CPU cores
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --workers 32

# Debug mode (single-threaded)
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10 --workers 1

# Skip files that already exist (great for resuming interrupted runs)
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --workers 16 --skip-existing


### DO AMAZON detections - for text bosrs etc.


python benchmarks\detections_2000ad\faster_rcnn.py --input-path "E:\amazon_analysis" --output-path E:\amazon\test_dections\predictions.json  --save-vis 196 --conf-threshold 0.5         


python benchmarks\detections_2000ad\faster_rcnn.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json  --save-vis 10 --conf-threshold 0.5         

python benchmarks\detections_2000ad\faster_rcnn.py --input-path "E:\CalibreComics_extracted\sex vol1 - Unknown" --output-path E:\CalibreComics\test_dections\sexpredictions.json  --save-vis 196 --conf- threshold 0.5         

3) C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv>python 02_comixs_prod_detect_HF.py --root "E:\CalibreComics_extracted\PRG1795 - Unknown" --out E:\CalibreComics\test_detections --viz

## coco dataset
(caption3) C:\Users\Richard\OneDrive\GIT\CoMix>python benchmarks\detections\openrouter\coco_to_dataspec.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test


python benchmarks\detections\openrouter\coco_to_dataspect_test.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10

# Test with 10 images, 8 workers
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10 --workers 8

# Full processing with all CPU cores
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --workers 32

# Debug mode (single-threaded)
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10 --workers 1

# Skip files that already exist (great for resuming interrupted runs)
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --workers 16 --skip-existing


### DO AMAZON detections - for text bosrs etc.


python benchmarks\detections_2000ad\faster_rcnn.py --input-path "E:\amazon_analysis" --output-path E:\amazon\test_dections\predictions.json  --save-vis 196 --conf-threshold 0.5         



### current detections use calibre version
python benchmarks\detections_2000ad\faster_rcnn_calibre.py --input-path "E:\CalibreComics_extracted_test --output-path E:\CalibreComics\test_dections\sexpredictions.json  --save-vis 196 --conf- threshold 0.5         


### current detections use calibre version
python benchmarks\detections_2000ad\faster_rcnn_calibre.py --input-path "E:\amazon" --output-path E:\amazon\test_detections\amazon_predictions.json  --save-vis 196 --conf- threshold 0.5         


# amazon coco
# Skip files that already exist (great for resuming interrupted runs)
python benchmarks\detections\openrouter\coco_to_dataspect_test_multi.py --coco E:\amazon\test_detections\amazon_predictions.json --vlm_dir E:\amazon_analysis --out E:\amazon_datacontract-test --workers 16 --skip-existing
