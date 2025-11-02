# Comic-Analysis
Work on looking at analysis of comics with data science
## Repo
Scattered work in progress being reorganised as version 1 wraps up.
### src
- comix
    - The CoMix repo original code and some additions I used for testing of capability and datasets
    - the 2000ad directory there is my versions with testing on some comics from the aforementioned due to handy quick downloads
- calibre
    - Updated work looking at modelling based on a 'Calibre' subset of comics - things from multiple not Amazon sources and the name references the great Digital library storage tool Calibre - also a smaller subset so good for testing
    - Amazon = Comixology but shorter to type and less likely to make me annoyed evoking the old name and how good it was.

# Closure Lite Framework
- A multimodal fusion model to look at images, text and panel reading order.
    - Aim - get multimodal fused embeddings that would be queryable for similarity
        - Embeddings stored in zarr works nicely for speed at size currently
            - 80K 'perfect match subset' test takes up around 500MB on disk.
        - See interface - with a flask ask to do that
    - Designed to be useable on a reasonable retail GPU - so 384 dim ebeddings etc.

## Data
- Comic pages, lots of them.
- VLM text extraction test - basically I wanted to try this ahead of general OCR - which is definitely not aimed at comics.

## Basic Process
# Basic Process

Find comics
- amazon
- dark horse
- calibre [humble bundle, drive thru etc.] 
- TODO; neon ichigan scraping? new humble bundle 
    - deal with duplicate comics - no point having two copies in embeddings

Convert to pages - from pdf/cbz etc.

Use VLMS to get panel text for each page

Use Fast-RCNN to get panel boxes and coords

Join together into DataSpec for modelling
- coco_to_dataspec

Make perfect match subset for training properly

Train model
- users closure_lite_dataset - closure_lite_simple_framework - train_closure_lite_simple_with_list

Embeddings
- Run model to make embeddings

Query
- By Code or Interface

## Perfect Match Notes

- Make sure text is as close to right panels as we can - same number from rcnn and vlm
    - Train with this dataset
- We got 25% alignment betwen fast-rcnn and various vlm runs - so need to work out what is best and affordable there
    - do we need to ocr fast rcnn




# Relevant Research
## Survey
- One missing piece in Vision and Language: A Survey on Comics Understanding https://arxiv.org/abs/2409.09502v1
- Investigating Neural Networks and Transformer Models for Enhanced Comic Decoding - https://dl.acm.org/doi/10.1007/978-3-031-70645-5_10

# Tasks of Interest
- Dialogue transcription
- Therefore, needs a pipeline above that can detect panels

# Pipeline
- See Comix repo
- the code in detections_2000ad are adaptations of the similarly named functions in CoMix/detections to just be inference and not worry about evaluations of models
- https://github.com/emanuelevivoli/CoMix/tree/main/benchmarks/detections
- refer there for details, but basically

- install pytorch as per pytorch site instructions
- using CUDA 11.8 so far

# Model Possibilities
## VLMS - current generation
- Some newer VLMs can basically zero shot this problem to some degree, including some cheap models

# NOTES
# Gemini 4B - can do basics
- fails on some perpeptually?
- yet to be understood why
- perhaps look at fourier analysis?

# Gemma 12B - can handle some failures of 4B

# Mistral 3.1
- character A and character B only generally but can handle gemini failures

# Qwen 2.5 VL Instruct failed on same that gemini 4B did

# Phi4 not very good

# Lllama 11B failed to process

# Mistral 3.1
- character A and character B only generally but can handle gemini failures

# Qwen 2.5 VL Instruct 
- failed on same that gemini 4B did

# Gemini Flash 1.5 
- can do missing - some null captions and characters

# Gemini Flash 2.5 flash lite
- can do missing 
- good, but output is 8 times more expensive than gemma 4B - which could run locally
- quite a few connection errors with google as the provider
- on the last hardest 660 had  1/3 errors and 1/5 json errors

# GPT Nano 4.1 says unsupported image type?
- so not as good as google which can handle

# Meta Llama 4 Scout
- much better, success on 300 out of 500 images left at the end 
- also 0.08/0.3 compared to 0.10/0.40 for Gemini Flash Lite 2.5 - so way better
- GMI Cloud provider big problems


# Future Research
- https://arxiv.org/abs/2503.08561
- https://www.researchgate.net/publication/389748978_ComicsPAP_understanding_comic_strips_by_picking_the_correct_panel?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19

- https://www.researchgate.net/publication/326137469_Digital_Comics_Image_Indexing_Based_on_Deep_Learning

- A Deep Learning Pipeline for the Synthesis of Graphic Novels - https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC_2021_paper_52.pdf


## Embeddings
### Embedding Generation Strategy:
1. What We Need to Generate:
Panel embeddings (P) - Raw panel representations
Page embeddings (E_page) - Aggregated page-level representations
Reading order embeddings - For sequence understanding
2. Dataset Coverage:
Amazon perfect matches: 212K pages
CalibreComics perfect matches: 80K
Combined dataset: All high-quality samples
3. Technical Approach:
Option A: Batch Processing Script

# Page Stream Segmentation
- Needed for next version of the above
- Feed page type markers into multimodal fusion

## CoSMo
- https://github.com/mserra0/CoSMo-ComicsPS
    - arXiv paper link there
- Model designed to do this
    - Uses Qwen 2.5 VL 32B for OCR 
    - Which reddit seems to like a lot
    - Is Gemma as good?  e.g. costs

- No actual model to test
- Need to train one