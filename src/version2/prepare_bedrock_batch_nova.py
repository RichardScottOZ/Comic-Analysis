#!/usr/bin/env python3
"""
Prepare AWS Bedrock Batch Inference (Nova Lite)

This script generates the input JSONL file required for an AWS Bedrock Batch Inference job.
It uses the 'Amazon Nova Lite' model which is cost-effective ($0.06/1k images on-demand, $0.03/1k batch).

Usage:
    python src/version2/prepare_bedrock_batch_nova.py \
        --manifest manifests/master_manifest_20251229.csv \
        --output-jsonl batch_input_nova_lite_100.jsonl \
        --limit 100
"""

import argparse
import csv
import json
import os

def create_structured_prompt():
    return """Analyze this comic page and provide a detailed structured analysis in JSON format. Focus on:

1. **Panel Analysis**: Identify and describe each panel
2. **Character Identification**: Note characters, their actions, and dialogue
3. **Story Elements**: Plot points, setting, mood
4. **Visual Elements**: Art style, colors, composition
5. **Text Elements**: Speech bubbles, captions, sound effects

Return ONLY valid JSON with this structure:
{
  "overall_summary": "Brief description of the page",
  "panels": [
    {
      "panel_number": 1,
      "caption": "Panel title/description",
      "description": "Detailed panel description",
      "speakers": [
        {
          "character": "Character name",
          "dialogue": "What they say",
          "speech_type": "dialogue|thought|narration"
        }
      ],
      "key_elements": ["element1", "element2"],
      "actions": ["action1", "action2"]
    }
  ],
  "summary": {
    "characters": ["Character1", "Character2"],
    "setting": "Setting description",
    "plot": "Plot summary",
    "dialogue": ["Line1", "Line2"]
  }
"""

def main():
    parser = argparse.ArgumentParser(description='Prepare Bedrock Batch JSONL for Nova Lite')
    parser.add_argument('--manifest', required=True, help='Path to master manifest CSV')
    parser.add_argument('--output-jsonl', required=True, help='Output path for .jsonl file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records (for testing)')
    args = parser.parse_args()

    print(f"Reading manifest: {args.manifest}")
    records = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    if args.limit:
        records = records[:args.limit]
        print(f"Limiting to first {args.limit} records for testing.")

    print(f"Generating {len(records)} batch inputs...")
    
    with open(args.output_jsonl, 'w', encoding='utf-8') as f_out:
        for rec in records:
            canonical_id = rec['canonical_id']
            s3_uri = rec['absolute_image_path']
            
            # Ensure uri is valid s3://
            if not s3_uri.startswith('s3://'):
                print(f"Skipping non-S3 path: {s3_uri}")
                continue

            # Amazon Nova Input Format (Converse API)
            # Note: For Batch, we provide the 'modelInput' that matches the runtime call
            batch_item = {
                "recordId": canonical_id, # Bedrock uses this to map output back to input
                "modelInput": {
                    "inferenceConfig": {
                        "max_new_tokens": 4096,
                        "temperature": 0.1
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": create_structured_prompt()
                                },
                                {
                                    "image": {
                                        "format": os.path.splitext(s3_uri)[1].lstrip('.').lower().replace('jpg','jpeg') or 'jpeg',
                                        "source": {
                                            "bytes": None # For batch with S3 images, logic is slightly different, checking...
                                        } 
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            
            # CORRECTION: Bedrock Batch currently supports S3 objects DIRECTLY in the request
            # ONLY if using the specific "S3 Data Source" feature which is newer.
            # Standard Bedrock Batch expects 'bytes' (base64) in JSONL which defeats the purpose (huge file).
            
            # HOWEVER: Nova models specifically support "s3Location" in the image block for Batch inference 
            # to avoid base64 encoding 1.2M images into a JSON file.
            
            batch_item["modelInput"]["messages"][0]["content"][1]["image"] = {
                "format": "jpeg", # Nova is tolerant, but generally expects 'jpeg', 'png', 'gif', 'webp'
                "source": {
                    "s3Location": {
                        "uri": s3_uri
                    }
                }
            }

            f_out.write(json.dumps(batch_item) + '\n')

    print(f"Done. Saved to {args.output_jsonl}")
    print("Next steps:")
    print(f"1. Upload this file to S3: aws s3 cp {args.output_jsonl} s3://calibrecomics-extracted/batch-inputs/")
    print("2. Create Batch Job in AWS Console (Bedrock > Batch Inference)")
    print("   - Select Model: Amazon Nova Lite")
    print(f"   - Input Data: s3://calibrecomics-extracted/batch-inputs/{os.path.basename(args.output_jsonl)}")
    print("   - Output Data: s3://calibrecomics-extracted/batch-outputs/")

if __name__ == "__main__":
    main()
