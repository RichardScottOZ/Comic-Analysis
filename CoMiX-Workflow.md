

## google drive link from non-liknked FASTTRACK.md
https://drive.google.com/drive/folders/1i4c3ZXBEjGPAkQd2coS0_Ir2wz2q98oo

- Note this info in FASTTRACK.md not linked in Docs

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