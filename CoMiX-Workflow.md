

## google drive link from non-liknked FASTTRACK.md
https://drive.google.com/drive/folders/1i4c3ZXBEjGPAkQd2coS0_Ir2wz2q98oo

- Note this info in FASTTRACK.md not linked in Docs

# models
- https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to C:\Users\Richard/.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

# flash-attention-2
- on windows
- https://github.com/Dao-AILab/flash-attention/issues/1469

## quantisations
- for captioning could try some and see how they do
- CoMix uses:
## Qwen 2-VL 72B Instruct
    - Quantised to 12.6
    - There is a 7B version could be tried on
## Qwen
- to test on retail hardware - possible smaller quants - original is Qwen 2VL 72B-Intruct quant at 12.6
- Model ID: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4"
        - https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4

## MiniCPM
- Model ID: "openbmb/MiniCPM-V-2_6"
        - Needs Flash Attention 2 - likely won't run on Windows
        - https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

## Idefics2
- No quants made - would have to do

## Idefics3
- https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
        - https://huggingface.co/2dameneko/Idefics3-8B-Llama3-nf4
        - https://huggingface.co/leon-se/Idefics3-8B-Llama3-FP8-Dynamic

# downloads
- Have to fill in forms 
# EBDTheque
- one is open office doc
- https://ebdtheque.univ-lr.fr/registration/

## Mangal109
https://docs.google.com/forms/u/0/d/e/1FAIpQLSefUGHUlkDfYlnOKLZlBqRtqlhmZWmhL1_NBfZ24zHOeCoguA/formResponse?pli=1
## DCM
- https://gitlab.univ-lr.fr/crigau02/dcm-dataset
- Christophe has a websaite and is on linkedin
## PopManga mangaplus
- mangaplus dataset needs the datasets library
- so need huggingface_hub and a token to read datasets mthat are gated
- https://mangaplus.shueisha.co.jp/manga_list/ongoing
- https://huggingface.co/settings/tokens/new?canReadGatedRepos=true&tokenType=fineGrained
- Then stores in huggingface cache as per usual

## Comics
- https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz
        - This is 120 GB so will take a while

# Generate captions
- python benchmarks/captioning/generate_captions.py  --model MODEL_NAME  --num_splits N --index I      [options]
- need qwen-vl-utils [l not 1]
# Post-process results
python benchmarks/captioning/postprocess.py \
        --input_dir data/predicts.caps/MODEL_NAME-cap \
        --output_dir data/predicts.caps/MODEL_NAME-cap-processed
# Evaluate results
python comix/evaluators/captioning.py \
        -p MODEL_NAME \
        [--nlp_only | --arm_only]

magi needs transformers 4.45.2
had transformer 4.49        

- There is a quantized MiniCPM here
        - https://huggingface.co/openbmb/MiniCPM-V-2_6-int4