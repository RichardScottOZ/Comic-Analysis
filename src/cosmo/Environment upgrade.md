
(caption3) C:\Users\Richard\OneDrive\GIT\CoMix\CoSMo-ComicsPSS\CoSMo>pip install --upgrade transformers
Requirement already satisfied: transformers in c:\users\richard\.conda\envs\caption3\lib\site-packages (4.49.0)
Collecting transformers
  Downloading transformers-4.57.1-py3-none-any.whl.metadata (43 kB)
Requirement already satisfied: filelock in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (3.13.1)
Collecting huggingface-hub<1.0,>=0.34.0 (from transformers)
  Downloading huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: numpy>=1.17 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (2.1.1)
Requirement already satisfied: packaging>=20.0 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (24.2)
Requirement already satisfied: pyyaml>=5.1 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (2024.11.6)
Requirement already satisfied: requests in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (2.32.3)
Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers)
  Downloading tokenizers-0.22.1-cp39-abi3-win_amd64.whl.metadata (6.9 kB)
Requirement already satisfied: safetensors>=0.4.3 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (0.5.3)
Requirement already satisfied: tqdm>=4.27 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from transformers) (4.67.1)
Requirement already satisfied: fsspec>=2023.5.0 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (2024.6.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (4.12.2)
Requirement already satisfied: colorama in c:\users\richard\.conda\envs\caption3\lib\site-packages (from tqdm>=4.27->transformers) (0.4.6)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from requests->transformers) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from requests->transformers) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from requests->transformers) (2.3.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\richard\.conda\envs\caption3\lib\site-packages (from requests->transformers) (2025.1.31)
Downloading transformers-4.57.1-py3-none-any.whl (12.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 4.6 MB/s eta 0:00:00
Downloading huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.1/566.1 kB 4.8 MB/s eta 0:00:00
Downloading tokenizers-0.22.1-cp39-abi3-win_amd64.whl (2.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 5.9 MB/s eta 0:00:00
Installing collected packages: huggingface-hub, tokenizers, transformers
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.29.2
    Uninstalling huggingface-hub-0.29.2:
      Successfully uninstalled huggingface-hub-0.29.2
  Attempting uninstall: tokenizers
    Found existing installation: tokenizers 0.21.0
    Uninstalling tokenizers-0.21.0:
      Successfully uninstalled tokenizers-0.21.0
  Attempting uninstall: transformers
    Found existing installation: transformers 4.49.0
    Uninstalling transformers-4.49.0:
      Successfully uninstalled transformers-4.49.0
Successfully installed huggingface-hub-0.36.0 tokenizers-0.22.1 transformers-4.57.1