@echo off

echo Testing nvidia/nemotron-nano-12b-v2-vl:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "nvidia/nemotron-nano-12b-v2-vl:free" --output-dir "E:/openroutertests/nvidia_nemotron-nano-12b-v2-vl_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-small-3.1-24b-instruct:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-small-3.1-24b-instruct:free" --output-dir "E:/openroutertests/mistralai_mistral-small-3.1-24b-instruct_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-4b-it:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-4b-it:free" --output-dir "E:/openroutertests/google_gemma-3-4b-it_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-12b-it:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-12b-it:free" --output-dir "E:/openroutertests/google_gemma-3-12b-it_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-27b-it:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-27b-it:free" --output-dir "E:/openroutertests/google_gemma-3-27b-it_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.0-flash-exp:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.0-flash-exp:free" --output-dir "E:/openroutertests/google_gemini-2.0-flash-exp_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen-2.5-vl-7b-instruct:free...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen-2.5-vl-7b-instruct:free" --output-dir "E:/openroutertests/qwen_qwen-2.5-vl-7b-instruct_free"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing meta-llama/llama-3.2-11b-vision-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "meta-llama/llama-3.2-11b-vision-instruct" --output-dir "E:/openroutertests/meta-llama_llama-3.2-11b-vision-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-4b-it...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-4b-it" --output-dir "E:/openroutertests/google_gemma-3-4b-it"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/ministral-3b-2512...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/ministral-3b-2512" --output-dir "E:/openroutertests/mistralai_ministral-3b-2512"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-12b-it...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-12b-it" --output-dir "E:/openroutertests/google_gemma-3-12b-it"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing microsoft/phi-4-multimodal-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "microsoft/phi-4-multimodal-instruct" --output-dir "E:/openroutertests/microsoft_phi-4-multimodal-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/pixtral-12b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/pixtral-12b" --output-dir "E:/openroutertests/mistralai_pixtral-12b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-small-3.1-24b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-small-3.1-24b-instruct" --output-dir "E:/openroutertests/mistralai_mistral-small-3.1-24b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing thudm/glm-4.1v-9b-thinking...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "thudm/glm-4.1v-9b-thinking" --output-dir "E:/openroutertests/thudm_glm-4.1v-9b-thinking"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/ministral-8b-2512...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/ministral-8b-2512" --output-dir "E:/openroutertests/mistralai_ministral-8b-2512"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemma-3-27b-it...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemma-3-27b-it" --output-dir "E:/openroutertests/google_gemma-3-27b-it"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-small-3.2-24b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-small-3.2-24b-instruct" --output-dir "E:/openroutertests/mistralai_mistral-small-3.2-24b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing arcee-ai/spotlight...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "arcee-ai/spotlight" --output-dir "E:/openroutertests/arcee-ai_spotlight"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing meta-llama/llama-guard-4-12b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "meta-llama/llama-guard-4-12b" --output-dir "E:/openroutertests/meta-llama_llama-guard-4-12b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/ministral-14b-2512...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/ministral-14b-2512" --output-dir "E:/openroutertests/mistralai_ministral-14b-2512"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing bytedance/ui-tars-1.5-7b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "bytedance/ui-tars-1.5-7b" --output-dir "E:/openroutertests/bytedance_ui-tars-1.5-7b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen-2.5-vl-7b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen-2.5-vl-7b-instruct" --output-dir "E:/openroutertests/qwen_qwen-2.5-vl-7b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen2.5-vl-32b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen2.5-vl-32b-instruct" --output-dir "E:/openroutertests/qwen_qwen2.5-vl-32b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing amazon/nova-lite-v1...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "amazon/nova-lite-v1" --output-dir "E:/openroutertests/amazon_nova-lite-v1"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen2.5-vl-72b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen2.5-vl-72b-instruct" --output-dir "E:/openroutertests/qwen_qwen2.5-vl-72b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing bytedance-seed/seed-1.6-flash...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "bytedance-seed/seed-1.6-flash" --output-dir "E:/openroutertests/bytedance-seed_seed-1.6-flash"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing meta-llama/llama-4-scout...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "meta-llama/llama-4-scout" --output-dir "E:/openroutertests/meta-llama_llama-4-scout"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.0-flash-lite-001...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.0-flash-lite-001" --output-dir "E:/openroutertests/google_gemini-2.0-flash-lite-001"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing opengvlab/internvl3-78b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "opengvlab/internvl3-78b" --output-dir "E:/openroutertests/opengvlab_internvl3-78b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-8b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-8b-instruct" --output-dir "E:/openroutertests/qwen_qwen3-vl-8b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash-lite-preview-09-2025...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash-lite-preview-09-2025" --output-dir "E:/openroutertests/google_gemini-2.5-flash-lite-preview-09-2025"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-5-nano...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-5-nano" --output-dir "E:/openroutertests/openai_gpt-5-nano"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash-lite...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash-lite" --output-dir "E:/openroutertests/google_gemini-2.5-flash-lite"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-4.1-nano...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-4.1-nano" --output-dir "E:/openroutertests/openai_gpt-4.1-nano"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.0-flash-001...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.0-flash-001" --output-dir "E:/openroutertests/google_gemini-2.0-flash-001"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing meta-llama/llama-3.2-90b-vision-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "meta-llama/llama-3.2-90b-vision-instruct" --output-dir "E:/openroutertests/meta-llama_llama-3.2-90b-vision-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing baidu/ernie-4.5-vl-28b-a3b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "baidu/ernie-4.5-vl-28b-a3b" --output-dir "E:/openroutertests/baidu_ernie-4.5-vl-28b-a3b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing x-ai/grok-4.1-fast...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "x-ai/grok-4.1-fast" --output-dir "E:/openroutertests/x-ai_grok-4.1-fast"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing x-ai/grok-4-fast...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "x-ai/grok-4-fast" --output-dir "E:/openroutertests/x-ai_grok-4-fast"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing deepcogito/cogito-v2-preview-llama-109b-moe...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "deepcogito/cogito-v2-preview-llama-109b-moe" --output-dir "E:/openroutertests/deepcogito_cogito-v2-preview-llama-109b-moe"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing nvidia/nemotron-nano-12b-v2-vl...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "nvidia/nemotron-nano-12b-v2-vl" --output-dir "E:/openroutertests/nvidia_nemotron-nano-12b-v2-vl"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-30b-a3b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-30b-a3b-instruct" --output-dir "E:/openroutertests/qwen_qwen3-vl-30b-a3b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing meta-llama/llama-4-maverick...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "meta-llama/llama-4-maverick" --output-dir "E:/openroutertests/meta-llama_llama-4-maverick"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-4o-mini-2024-07-18...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-4o-mini-2024-07-18" --output-dir "E:/openroutertests/openai_gpt-4o-mini-2024-07-18"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-4o-mini...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-4o-mini" --output-dir "E:/openroutertests/openai_gpt-4o-mini"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen-vl-plus...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen-vl-plus" --output-dir "E:/openroutertests/qwen_qwen-vl-plus"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-30b-a3b-thinking...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-30b-a3b-thinking" --output-dir "E:/openroutertests/qwen_qwen3-vl-30b-a3b-thinking"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing z-ai/glm-4.6v...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "z-ai/glm-4.6v" --output-dir "E:/openroutertests/z-ai_glm-4.6v"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing baidu/ernie-4.5-vl-424b-a47b...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "baidu/ernie-4.5-vl-424b-a47b" --output-dir "E:/openroutertests/baidu_ernie-4.5-vl-424b-a47b"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing perplexity/sonar...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "perplexity/sonar" --output-dir "E:/openroutertests/perplexity_sonar"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing minimax/minimax-01...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "minimax/minimax-01" --output-dir "E:/openroutertests/minimax_minimax-01"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-235b-a22b-thinking...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-235b-a22b-thinking" --output-dir "E:/openroutertests/qwen_qwen3-vl-235b-a22b-thinking"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-235b-a22b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-235b-a22b-instruct" --output-dir "E:/openroutertests/qwen_qwen3-vl-235b-a22b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing anthropic/claude-3-haiku...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "anthropic/claude-3-haiku" --output-dir "E:/openroutertests/anthropic_claude-3-haiku"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing stepfun-ai/step3...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "stepfun-ai/step3" --output-dir "E:/openroutertests/stepfun-ai_step3"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing z-ai/glm-4.5v...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "z-ai/glm-4.5v" --output-dir "E:/openroutertests/z-ai_glm-4.5v"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-large-2512...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-large-2512" --output-dir "E:/openroutertests/mistralai_mistral-large-2512"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-32b-instruct...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-32b-instruct" --output-dir "E:/openroutertests/qwen_qwen3-vl-32b-instruct"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-4.1-mini...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-4.1-mini" --output-dir "E:/openroutertests/openai_gpt-4.1-mini"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing bytedance-seed/seed-1.6...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "bytedance-seed/seed-1.6" --output-dir "E:/openroutertests/bytedance-seed_seed-1.6"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-5.1-codex-mini...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-5.1-codex-mini" --output-dir "E:/openroutertests/openai_gpt-5.1-codex-mini"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-5-image-mini...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-5-image-mini" --output-dir "E:/openroutertests/openai_gpt-5-image-mini"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-medium-3.1...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-medium-3.1" --output-dir "E:/openroutertests/mistralai_mistral-medium-3.1"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing openai/gpt-5-mini...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "openai/gpt-5-mini" --output-dir "E:/openroutertests/openai_gpt-5-mini"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing mistralai/mistral-medium-3...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "mistralai/mistral-medium-3" --output-dir "E:/openroutertests/mistralai_mistral-medium-3"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing qwen/qwen3-vl-8b-thinking...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "qwen/qwen3-vl-8b-thinking" --output-dir "E:/openroutertests/qwen_qwen3-vl-8b-thinking"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing amazon/nova-2-lite-v1...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "amazon/nova-2-lite-v1" --output-dir "E:/openroutertests/amazon_nova-2-lite-v1"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash-image...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash-image" --output-dir "E:/openroutertests/google_gemini-2.5-flash-image"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash-preview-09-2025...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash-preview-09-2025" --output-dir "E:/openroutertests/google_gemini-2.5-flash-preview-09-2025"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash-image-preview...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash-image-preview" --output-dir "E:/openroutertests/google_gemini-2.5-flash-image-preview"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-2.5-flash...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-2.5-flash" --output-dir "E:/openroutertests/google_gemini-2.5-flash"
if %errorlevel% neq 0 echo Failed %errorlevel%

echo Testing google/gemini-3-flash-preview...
python src/version2/batch_comic_analysis_vlm.py --manifest manifests/master_manifest_20251229.csv --limit 25 --workers 8 --timeout 180  --model "google/gemini-3-flash-preview" --output-dir "E:/openroutertests/google_gemini-3-flash-preview"
if %errorlevel% neq 0 echo Failed %errorlevel%

