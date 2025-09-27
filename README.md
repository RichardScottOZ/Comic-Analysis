# Comic-Analysis
Work on looking at analysis of comics with data science

# Survey
- One missing piece in Vision and Language: A Survey on Comics Understanding https://arxiv.org/abs/2409.09502v1

Investigating Neural Networks and Transformer Models for Enhanced Comic Decoding - https://dl.acm.org/doi/10.1007/978-3-031-70645-5_10

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

# VLMS - current generation
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

# TODO
- Run model to make embeddings
- Make sure text is as close to right panels as we can
- We got 25% alignment betwen fast-rcnn and various vlm runs - so need to work out what is best and affordable there
    - do we need to ocr fast rcnn
- Consider cover identification by embedding
    - will it just work
    - do we need to cluster or supervise to detect - not all 1 panel pages will be covers - some will be ads