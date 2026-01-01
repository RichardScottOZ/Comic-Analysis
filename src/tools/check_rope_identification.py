import json
import os

INPUT_FILE = 'model_comparisons_p002_p003_FULL.json'
TARGET_PAGE = '#Guardian 001_#Guardian 001 - p003.jpg'

def analyze_rope():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rope_models = []
    other_models = {}
    
    common_mistakes = [
        'shovel', 'spade', 'pickaxe', 'axe', 'weapon', 
        'sword', 'blade', 'staff', 'stick', 'crowbar', 
        'tool', 'lever', 'scythe'
    ]

    print(f"Analyzing {len(data)} models for 'rope' detection in {TARGET_PAGE}...\n")

    for model, pages in data.items():
        if TARGET_PAGE not in pages: continue
        page_data = pages[TARGET_PAGE]
        if 'error' in page_data or 'status' in page_data: continue
            
        if 'panels' in page_data and isinstance(page_data['panels'], list) and page_data['panels']:
            last_panel = page_data['panels'][-1]
            desc = str(last_panel.get('description', '')).lower()
            elements = str(last_panel.get('key_elements', [])).lower()
            combined_text = desc + " " + elements
            
            if 'rope' in combined_text:
                rope_models.append(model)
            else:
                found_mistakes = [k for k in common_mistakes if k in combined_text]
                other_models[model] = found_mistakes if found_mistakes else ["unspecified object"]

    print(f"✅ Models that correctly identified the ROPE ({len(rope_models)}):")
    for m in sorted(rope_models): print(f"  - {m}")

    print(f"\n❌ Models that hallucinated something else ({len(other_models)}):")
    for m, items in sorted(other_models.items()):
        item_str = ", ".join(items)
        print(f"  - {m}: {item_str}")

if __name__ == "__main__":
    analyze_rope()