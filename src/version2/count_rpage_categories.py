import pandas as pd
import argparse
import os

def count_rpage_categories(file_path):
    """
    Counts the occurrences of each category in the 'Rpage' column of a CSV file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading CSV file {file_path}: {e}"
    
    if 'Rpage' not in df.columns:
        return f"Error: 'Rpage' column not found in {file_path}"
    
    counts = df['Rpage'].value_counts()
    total = counts.sum()
    percentages = (counts / total * 100).round(2)
    
    result_df = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages.astype(str) + '%'
    })
    
    return result_df.to_string() + f"\nTotal pages: {total}"

def main():
    parser = argparse.ArgumentParser(description="Count 'Rpage' categories in CSV files.")
    parser.add_argument('csv_files', metavar='FILE', type=str, nargs='+',
                        help='One or more CSV files to process.')
    
    args = parser.parse_args()
    
    for file_path in args.csv_files:
        print(f"\n--- Processing: {file_path} ---")
        counts = count_rpage_categories(file_path)
        print(counts)
        print("----------------------------------")

if __name__ == "__main__":
    main()

