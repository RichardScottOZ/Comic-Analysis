import json
import os
import subprocess

def get_multimodal_models(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    candidates = []
    
    for model in data['data']:
        # check pricing (per 1M tokens, so raw value * 1000000)
        # API usually returns price per token or per 1M. Let's check raw values.
        # OpenRouter API returns strict per-token pricing (e.g. 0.000001)
        
        prompt_price = float(model['pricing'].get('prompt', 0)) * 1_000_000
        completion_price = float(model['pricing'].get('completion', 0)) * 1_000_000
        
        if completion_price > 3.00:
            continue
            
        # check modality
        # architecture.modality is often 'text+image->text' for VLMs
        arch = model.get('architecture', {})
        modality = arch.get('modality', '')
        
        # Some models don't have this field populated well, check ID or simple heuristic
        model_id = model['id'].lower()
        is_vision = 'vision' in model_id or 'vl' in model_id or 'multimodal' in modality or 'image' in modality
        
        # Explicitly exclude known non-vision models if heuristic is too broad, 
        # but 'vision'/'vl' usually safe.
        # Also check 'context_length' to avoid tiny context models? No, let's keep them.
        
        if is_vision:
            candidates.append({
                'id': model['id'],
                'input_price': prompt_price,
                'output_price': completion_price
            })
            
    # Sort by price (cheapest first)
    candidates.sort(key=lambda x: x['output_price'])
    return candidates

def generate_test_commands(models):
    # Base command structure
    # python src/version2/batch_comic_analysis_vlm.py --manifest ... --limit 25 --workers 4 ...
    
    print(f"Found {len(models)} candidate models.")
    
    base_cmd = (
        "python src/version2/batch_comic_analysis_vlm.py "
        "--manifest manifests/master_manifest_20251229.csv "
        "--limit 25 "
        "--workers 8 "
        "--timeout 180 "  # slightly generous timeout for 25 pages
    )
    
    # Create a batch file content
    batch_file_content = "@echo off\n\n"
    
    for m in models:
        model_id = m['id']
        # Create safe folder name
        safe_name = model_id.replace('/', '_').replace(':', '_')
        output_dir = f"E:/openroutertests/{safe_name}"
        
        cmd = f'{base_cmd} --model "{model_id}" --output-dir "{output_dir}"'
        
        print(f"Model: {model_id} | Input: ${m['input_price']:.4f} | Output: ${m['output_price']:.4f}")
        
        batch_file_content += f"echo Testing {model_id}...\n"
        batch_file_content += f"{cmd}\n"
        batch_file_content += "if %errorlevel% neq 0 echo Failed %errorlevel%\n\n"

    with open("run_vlm_benchmark.bat", "w") as f:
        f.write(batch_file_content)
    
    print("\nGenerated 'run_vlm_benchmark.bat'")

if __name__ == "__main__":
    models = get_multimodal_models('openrouter_models.json')
    generate_test_commands(models)
