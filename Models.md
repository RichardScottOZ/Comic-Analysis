Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
- d-----         8/03/2025   5:04 PM                .locks
- d-----         8/03/2025   5:10 PM                datasets--ragavsachdeva--popmanga_test       
- d-----         9/03/2025   3:42 PM                models--IDEA-Research--grounding-dino-base   
- d-----        10/03/2025  11:58 AM                models--microsoft--Florence-2-large-ft       
- d-----         9/03/2025   8:11 PM                models--microsoft--trocr-base-printed        
- d-----        10/03/2025   9:23 AM                models--openbmb--MiniCPM-V-2_6
- d-----        10/03/2025   7:27 PM                models--Qwen--Qwen2-VL-7B-Instruct-GPTQ-Int4 
- d-----         9/03/2025   8:11 PM                models--ragavsachdeva--magi
- d-----         9/03/2025   8:12 PM                models--timm--resnet50.a1_in1k


https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ


## alphaxiv
Short answer: CoMix gets panel boxes with generic object detectors; panels are just one of the four detection classes. There’s no bespoke panel-segmentation algorithm or polygon masks—only rectangular boxes that detectors localize and that later drive reading-order.

Concretely:
- Zero‑shot open‑vocabulary detector. They run Grounding DINO and prompt its text encoder with panel phrases such as “comics panels,” “manga panels,” “frames,” and “windows” to produce panel boxes alongside characters, faces, and text; see https://arxiv.org/abs/2303.05499. This requires no fine‑tuning and is used as a style‑robust baseline.
- Supervised detectors fine‑tuned for four classes. They fine‑tune standard detectors (Faster R‑CNN with a ResNet‑50 backbone and YOLOv8) on the CoMix-style data to predict panel, character, face, and text boxes, evaluating with mAP@0.5 and R@100. Faster R‑CNN training details include a 5e‑3 learning rate with StepLR/Cosine scheduling; YOLOv8 uses Ultralytics defaults (https://github.com/ultralytics/ultralytics). The panel class is learned like any other object class.
- A manga‑trained transformer detector. They also evaluate Magi (a RelationFormer/DeTR-style model) that directly predicts panel, character, and text boxes (not faces); see https://arxiv.org/abs/2401.10224 and its backbone idea in Relationformer at https://arxiv.org/abs/2203.10202. Despite being trained on manga, it attains the strongest average mAP when the comparison is restricted to the classes it predicts.
- Note: DASS (https://arxiv.org/abs/2211.10641) is included only for faces/characters, not panels.

That’s it: panel “segmentation” in CoMix is box detection via these detectors; the paper does not report a per‑class panel mAP breakdown, only averages across classes.