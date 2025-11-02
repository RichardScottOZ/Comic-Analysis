use this:

ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base",

(caption3) C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv>python 02_comixs_prod_detect_HF.py --root "E:\CalibreComics_extracted\13thfloor vol1 - Unknown\JPG4CBZ" --out E:\CalibreComics\test_detections
Books: 0book [00:00, ?book/C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv\02_comixs_prod_detect_HF.py:120: FutureWarning: `box_threshold` is deprecated and will be removed in version 4.51.0 for `GroundingDinoProcessor.post_process_grounded_object_detection`. Use `threshold` instead.
  processed = self.processor.post_process_grounded_object_detection(
dict_keys(['scores', 'boxes', 'text_labels', 'labels'])
Books: 0book [00:00, ?book/s]
Traceback (most recent call last):
  File "C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv\02_comixs_prod_detect_HF.py", line 246, in <module>
    main()
  File "C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv\02_comixs_prod_detect_HF.py", line 195, in main
    dets, (H, W) = detector.infer(im, args.box_thr, args.text_thr)
  File "C:\Users\Richard\.conda\envs\caption3\lib\site-packages\torch\utils\_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\alphaxiv\02_comixs_prod_detect_HF.py", line 128, in infer
    phrases = processed["phrases"]
  File "C:\Users\Richard\.conda\envs\caption3\lib\site-packages\transformers\models\grounding_dino\processing_grounding_dino.py", line 96, in __getitem__
    return super().__getitem__(key)
KeyError: 'phrases'