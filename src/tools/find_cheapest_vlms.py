import json
import pandas as pd

def main():
    try:
        df = pd.read_csv('benchmarks/detections/openrouter/models_dataset/openrouter_vision_models_20260301_174734.csv')
        
        # Calculate estimated cost for 1.2M pages 
        # Assume: 1 image + text = ~1000 input tokens total.
        # Output: ~400 tokens of JSON (descriptions + bounding boxes)
        df['est_cost_1M'] = ((df['prompt_price'] * 1000) + (df['completion_price'] * 400)) * 1_200_000
        
        # Filter out completely free and auto models
        df = df[~df['id'].str.contains(':free|openrouter/auto|openrouter/free', case=False, na=False)]
        
        # Sort by cheapest
        df = df.sort_values('est_cost_1M')
        
        print('\nTop 20 Cheapest Paid Vision Models for 1.2M Pages (Est. 1000 in, 400 out tokens):')
        print(df[['id', 'prompt_price', 'completion_price', 'est_cost_1M']].head(20).to_string(index=False))
        
        print('\nQwen Vision Models:')
        qwen_df = df[df['id'].str.contains('qwen', case=False)]
        print(qwen_df[['id', 'prompt_price', 'completion_price', 'est_cost_1M']].to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
