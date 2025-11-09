import pandas as pd
import os
import shutil
import json
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = r'C:\Users\Richard\OneDrive\GIT\CoMix\data_analysis\pss_training.csv'
NEW_ROOT_DIR = r'E:\PSS_Training'
IMAGE_PATH_COLUMN = 'image_path' # Corrected column name for the full image path
BOOK_FOLDER_COLUMN = 'book_folder'
PAGE_NUM_COLUMN = 'page_num'
TEXT_COLUMN = 'vlm_json_content'
# -------------------

def extract_vlm_content_for_ocr(json_string):
    """
    Extracts the content from the vlm_json_content string,
    which is expected to be a JSON string representing a dictionary.
    Returns the parsed dictionary, or an empty dictionary if parsing fails.
    """
    if not isinstance(json_string, str) or not json_string.strip():
        return {}
    try:
        # The user states it's already a dictionary, but it's a string in the CSV
        # so we still need to load it from string to dict.
        data = json.loads(json_string)
        if isinstance(data, dict):
            return data
        else:
            # If it's not a dict, return empty or handle as appropriate
            print(f"Warning: vlm_json_content was not a dictionary after parsing: {json_string}")
            return {}
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse vlm_json_content as JSON: {json_string}")
        return {}

def main():
    """
    Creates a new training directory by copying images and creating OCR JSON files.
    """
    print(f"Creating new training directory at: {NEW_ROOT_DIR}")
    os.makedirs(NEW_ROOT_DIR, exist_ok=True)

    print(f"Reading CSV: {CSV_PATH}")
    
    # Use chunking for large CSV
    chunksize = 10000
    
    # Get total number of rows for progress bar
    # This might be slow for very large files, but provides accurate progress.
    # Alternatively, you can estimate or remove the total for tqdm.
    total_rows = sum(1 for row in open(CSV_PATH, 'r', encoding='utf-8')) -1


    with tqdm(total=total_rows, desc="Processing pages") as pbar:
        for chunk in pd.read_csv(CSV_PATH, chunksize=chunksize, usecols=[IMAGE_PATH_COLUMN, BOOK_FOLDER_COLUMN, PAGE_NUM_COLUMN, TEXT_COLUMN], low_memory=False):
            for _, row in chunk.iterrows():
                source_image_path = row[IMAGE_PATH_COLUMN]
                book_folder = row[BOOK_FOLDER_COLUMN]
                page_num = row[PAGE_NUM_COLUMN]
                vlm_content = row[TEXT_COLUMN]

                if not all([source_image_path, book_folder, str(page_num)]): # page_num can be 0, so check as string
                    pbar.update(1)
                    continue

                # --- Create destination paths ---
                dest_book_dir = os.path.join(NEW_ROOT_DIR, book_folder)
                os.makedirs(dest_book_dir, exist_ok=True)

                # Standardize file names
                try:
                    page_num_int = int(float(page_num)) # Handle potential float representation from CSV
                    _, extension = os.path.splitext(source_image_path)
                    if not extension: extension = '.jpg' # Default extension if none found
                    
                    new_file_basename = f"page_{page_num_int:04d}" # e.g., page_0001
                    dest_image_path = os.path.join(dest_book_dir, f"{new_file_basename}{extension}")
                    dest_json_path = os.path.join(dest_book_dir, f"{new_file_basename}.json")
                except (ValueError, TypeError):
                    print(f"Skipping row due to invalid page_num: {page_num}")
                    pbar.update(1)
                    continue

                # --- 1. Copy Image ---
                # Only copy if source exists and destination doesn't, to save time on reruns
                if os.path.exists(source_image_path) and not os.path.exists(dest_image_path):
                    try:
                        shutil.copy2(source_image_path, dest_image_path)
                    except Exception as e:
                        print(f"Warning: Could not copy {source_image_path} to {dest_image_path}. Error: {e}")
                elif not os.path.exists(source_image_path):
                    print(f"Warning: Source image not found: {source_image_path}")


                # --- 2. Create OCR JSON ---
                # Only create if destination doesn't exist, to save time on reruns
                if not os.path.exists(dest_json_path):
                    vlm_dict_content = extract_vlm_content_for_ocr(vlm_content)
                    ocr_data = {"OCRResult": vlm_dict_content} # Store the entire dictionary
                    try:
                        with open(dest_json_path, 'w', encoding='utf-8') as f:
                            json.dump(ocr_data, f, indent=4)
                    except Exception as e:
                        print(f"Warning: Could not write JSON for {dest_json_path}. Error: {e}")
                
                pbar.update(1)

    print("\nProcessing complete.")
    print(f"New training directory created at {NEW_ROOT_DIR}")
    print("Next steps:")
    # The root_dir in pss_multimodal.py has already been updated by the assistant.
    # This message is for informational purposes.
    escaped_path = NEW_ROOT_DIR.replace('\\', '\\\\')
    print(f"1. The 'root_dir' in 'pss_multimodal.py' has been updated to '{escaped_path}'")
    print("2. Run the training command.")

if __name__ == '__main__':
    main()
