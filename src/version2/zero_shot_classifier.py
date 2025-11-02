import argparse
import pandas as pd
import json
from transformers import pipeline
from tqdm import tqdm
import numpy as np

def create_text_sequence(vlm_json_content, strategy: str = "all_text"):
    """
    Extracts and concatenates relevant text from the VLM JSON content based on strategy.
    """
    if not vlm_json_content or pd.isna(vlm_json_content):
        return ""
    
    try:
        vlm_data = json.loads(vlm_json_content)
    except json.JSONDecodeError:
        return "" # Return empty if JSON is invalid

    if strategy == "overall_summary":
        return vlm_data.get('overall_summary', "").strip()
    elif strategy == "summary_plot":
        return vlm_data.get('summary', {}).get('plot', "").strip()
    elif strategy == "all_text":
        text_parts = []

        # 1. Overall Summary
        if 'overall_summary' in vlm_data and isinstance(vlm_data['overall_summary'], str):
            text_parts.append(vlm_data['overall_summary'])
        
        # 2. Plot Summary
        if 'summary' in vlm_data and isinstance(vlm_data['summary'], dict):
            if 'plot' in vlm_data['summary'] and isinstance(vlm_data['summary']['plot'], str):
                text_parts.append(vlm_data['summary']['plot'])
        
        # 3. Panel-level text (captions, dialogue, narration)
        if 'panels' in vlm_data and isinstance(vlm_data['panels'], list):
            for panel in vlm_data['panels']:
                if isinstance(panel, dict):
                    if 'caption' in panel and isinstance(panel['caption'], str):
                        text_parts.append(panel['caption'])
                    if 'description' in panel and isinstance(panel['description'], str):
                        text_parts.append(panel['description'])
                    if 'speakers' in panel and isinstance(panel['speakers'], list):
                        for speaker in panel['speakers']:
                            if isinstance(speaker, dict) and 'dialogue' in speaker and isinstance(speaker['dialogue'], str):
                                text_parts.append(speaker['dialogue'])
                    # Also check for direct dialogue/narration/text fields at panel level
                    if 'dialogue' in panel and isinstance(panel['dialogue'], str):
                        text_parts.append(panel['dialogue'])
                    if 'narration' in panel and isinstance(panel['narration'], str):
                        text_parts.append(panel['narration'])
                    if 'text' in panel and isinstance(panel['text'], str):
                        text_parts.append(panel['text'])

        return " ".join(text_parts).strip()
    else:
        print(f"Warning: Unknown text input strategy: {strategy}. Using all_text.")
        return create_text_sequence(vlm_json_content, strategy="all_text") # Fallback

def main():
    parser = argparse.ArgumentParser(description="Perform zero-shot text classification on VLM JSON content.")
    parser.add_argument('--input_csv', required=True, help='Path to the combined CSV with vlm_json_content column.')
    parser.add_argument('--output_csv', help='Path for the output CSV with zero-shot predictions. Defaults to input_csv_zero_shot.csv')
    parser.add_argument('--model_name', default="facebook/bart-large-mnli", help='Hugging Face model name for zero-shot classification.')
    parser.add_argument('--sample_rows', type=int, default=None, help='Process only the first N rows for testing/estimation.')
    parser.add_argument('--text_input_strategy', default="all_text", choices=["all_text", "overall_summary", "summary_plot"], help='Strategy for creating text sequence for classification.')
    parser.add_argument('--pipeline_batch_size', type=int, default=32, help='Batch size for the Hugging Face pipeline inference.')
    args = parser.parse_args()

    print(f"args.sample_rows: {args.sample_rows}") # Debug print

    if args.output_csv is None:
        args.output_csv = args.input_csv.replace('.csv', f'_zero_shot_{args.text_input_strategy}.csv')

    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    if args.sample_rows is not None:
        print(f"Processing only the first {args.sample_rows} rows as requested...")
        df = df.head(args.sample_rows)
        print(f"DataFrame length after sampling: {len(df)}") # Debug print

    if 'vlm_json_content' not in df.columns:
        print("Error: 'vlm_json_content' column not found in the input CSV. Please ensure you ran annotate_page_types.py with --return_vlm_json.")
        return

    # Define your original target page type classes
    original_labels = [
        "cover_front",
        "narrative",
        "advertisement",
        "credits_indicia",
        "back_matter_text",
        "back_matter_art",
        "preview",
        "cover_internal",
        "other"
    ]

    # Natural language versions for the zero-shot model
    natural_language_labels = [
        "front cover",
        "narrative page",
        "advertisement",
        "credits and indicia page",
        "back matter text page",
        "back matter art page",
        "preview page",
        "internal cover page",
        "other page type"
    ]

    # Create a mapping from natural language back to original underscore format
    label_mapping = dict(zip(natural_language_labels, original_labels))

    print(f"Loading zero-shot classification model: {args.model_name}...")
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")
    classifier = pipeline(
        "zero-shot-classification", 
        model=args.model_name,
        device=device,
        batch_size=args.pipeline_batch_size
    )

    sequences_to_process = []
    original_indices = []

    print("Preparing text sequences for batch classification...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text"):
        text_sequence = create_text_sequence(row['vlm_json_content'], strategy=args.text_input_strategy)
        if text_sequence:
            sequences_to_process.append(text_sequence)
            original_indices.append(idx)

    if not sequences_to_process:
        print("No valid text sequences found to classify. Exiting.")
        df['zero_shot_prediction'] = None
        df['zero_shot_confidence'] = None
        df.to_csv(args.output_csv, index=False)
        return

    print(f"Performing zero-shot classification on {len(sequences_to_process)} sequences...")
    print(f"Pipeline will process in batches of {args.pipeline_batch_size}")
    
    all_predictions = []
    all_confidences = []
    all_probabilities = [] # New list to store all class probabilities

    # Explicitly batch sequences for tqdm progress
    # We still iterate explicitly to show progress
    num_batches = (len(sequences_to_process) + args.pipeline_batch_size - 1) // args.pipeline_batch_size
    
    for i in tqdm(range(0, len(sequences_to_process), args.pipeline_batch_size), 
                  total=num_batches, desc="Classifying batches"):
        batch_sequences = sequences_to_process[i:i + args.pipeline_batch_size]
        batch_original_indices = original_indices[i:i + args.pipeline_batch_size]

        # Pipeline internally handles batching with the batch_size set at initialization
        batch_results = classifier(batch_sequences, natural_language_labels, multi_label=False)

        # Handle both single result dict and list of dicts
        if not isinstance(batch_results, list):
            batch_results = [batch_results]

        for j, result in enumerate(batch_results):
            predicted_label_natural = result['labels'][0]
            predicted_label_original = label_mapping.get(predicted_label_natural, predicted_label_natural)
            
            all_predictions.append((batch_original_indices[j], predicted_label_original))
            all_confidences.append((batch_original_indices[j], result['scores'][0]))
            # Store all labels and their scores as a dictionary
            all_probabilities.append((batch_original_indices[j], dict(zip(result['labels'], result['scores']))))

    # Initialize new columns with None
    prediction_col_name = f'zero_shot_prediction_{args.text_input_strategy}'
    confidence_col_name = f'zero_shot_confidence_{args.text_input_strategy}'
    df[prediction_col_name] = None
    df[confidence_col_name] = None

    # Map results back to the original DataFrame
    for idx, pred in all_predictions:
        df.loc[idx, prediction_col_name] = pred
    for idx, conf in all_confidences:
        df.loc[idx, confidence_col_name] = conf

    # Add columns for all class probabilities
    for idx, probs_dict in all_probabilities:
        for natural_label, score in probs_dict.items():
            original_label = label_mapping.get(natural_label, natural_label) # Map back to original format
            prob_col_name = f'zero_shot_prob_{original_label}_{args.text_input_strategy}'
            df.loc[idx, prob_col_name] = round(score, 3)

    print(f"Saving results to {args.output_csv}...")
    try:
        df.to_csv(args.output_csv, index=False)
        print("Zero-shot classification complete and results saved!")
    except Exception as e:
        print(f"Error saving results to {args.output_csv}: {e}")

if __name__ == "__main__":
    main()
