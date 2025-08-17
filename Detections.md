python benchmarks\detections_2000ad\faster_rcnn.py --input-path data/datasets.unify/2000ad/images --output-path data/predicts.coco/2000ad/faster-rcnn/predictions.json  --save-vis 10 --conf-threshold 0.5         

python benchmarks\detections_2000ad\faster_rcnn.py --input-path "E:\CalibreComics_extracted\sex vol1 - Unknown" --output-path E:\CalibreComics\test_dections\sexpredictions.json  --save-vis 196 --conf- threshold 0.5         

3) C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv>python 02_comixs_prod_detect_HF.py --root "E:\CalibreComics_extracted\PRG1795 - Unknown" --out E:\CalibreComics\test_detections --viz

## coco dataset
(caption3) C:\Users\Richard\OneDrive\GIT\CoMix>python benchmarks\detections\openrouter\coco_to_dataspec.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test


python benchmarks\detections\openrouter\coco_to_dataspect_test.py --coco E:\CalibreComics\test_dections\predictions.json --vlm_dir E:\CalibreComics_analysis --out E:\CalibreComics_datacontract-test --limit 10