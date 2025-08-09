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
- the code in detections_2000ad are adaptations of the similarly named functions in CoMix/detections to just be inference and not worry about evalutions of models
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
