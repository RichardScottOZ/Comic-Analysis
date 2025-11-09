import pandas as pd
import json
import os
from collections import defaultdict
import random

def get_category_from_rpage(rpage):
    """Maps Rpage value to a category key for the JSON output."""
    mapping = {
        'cover': 'cover',
        'credits': 'credits',
        'story': 'story',
        'advertisement': 'advertisement',
        'art': 'art',
        'text': 'text',
        'back_cover': 'back_cover'
    }
    return mapping.get(rpage, None)

def process_chunk(chunk, books_data):
    """Processes a chunk of the CSV data."""
    for _, row in chunk.iterrows():
        book_folder = row['book_folder']
        try:
            page_num = int(row['page_num'])
        except ValueError:
            continue
        rpage = row['Rpage']
        
        category = get_category_from_rpage(rpage)
        if category:
            books_data[book_folder].append({'page_num': page_num, 'category': category})

def create_json_structure(books_data):
    """Creates the final JSON structure from the processed data."""
    output_data = []
    for book_folder, pages in books_data.items():
        if not pages:
            continue

        # Sort pages by page number
        pages.sort(key=lambda x: x['page_num'])
        
        book_dict = {'hash_code': book_folder}
        
        # Group pages by category
        pages_by_cat = defaultdict(list)
        for page in pages:
            pages_by_cat[page['category']].append(page)

        for category, cat_pages in pages_by_cat.items():
            if not cat_pages:
                continue

            book_dict[category] = []
            
            # Find contiguous page blocks
            start_page = cat_pages[0]['page_num']
            end_page = cat_pages[0]['page_num']
            
            for i in range(1, len(cat_pages)):
                if cat_pages[i]['page_num'] == end_page + 1:
                    end_page = cat_pages[i]['page_num']
                else:
                    item = {'page_start': start_page, 'page_end': end_page}
                    book_dict[category].append(item)
                    start_page = cat_pages[i]['page_num']
                    end_page = cat_pages[i]['page_num']
            
            # Add the last block
            item = {'page_start': start_page, 'page_end': end_page}
            book_dict[category].append(item)

        output_data.append(book_dict)
        
    return output_data

def main():
    csv_path = r'C:\Users\Richard\OneDrive\GIT\CoMix\data_analysis\pss_training.csv'
    output_dir = r'C:\Users\Richard\OneDrive\GIT\CoMix\data'
    
    os.makedirs(output_dir, exist_ok=True)

    books_data = defaultdict(list)
    
    use_cols = ['book_folder', 'page_num', 'Rpage']
    dtype_spec = {col: str for col in use_cols}
    
    chunksize = 100000
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=use_cols, dtype=dtype_spec, low_memory=False):
        chunk['page_num'] = pd.to_numeric(chunk['page_num'], errors='coerce')
        chunk.dropna(subset=['page_num'], inplace=True)
        process_chunk(chunk, books_data)
        
    print(f"Processed {len(books_data)} books for segmentation data.")

    json_data = create_json_structure(books_data)
    
    random.seed(42)
    random.shuffle(json_data)
    
    train_size = int(0.8 * len(json_data))
    val_size = int(0.1 * len(json_data))
    
    train_data = json_data[:train_size]
    val_data = json_data[train_size:train_size + val_size]
    test_data = json_data[train_size + val_size:]
    
    with open(os.path.join(output_dir, 'comics_train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(output_dir, 'comics_val.json'), 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(output_dir, 'comics_test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)
        
    print(f"Successfully created segmentation files (comics_train.json, etc.) in {output_dir}")

if __name__ == '__main__':
    main()