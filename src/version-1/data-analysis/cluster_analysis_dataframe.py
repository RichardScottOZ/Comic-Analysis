import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Inspect clustering and text analysis CSV and print grouped statistics.')
    parser.add_argument('--csv', '-c', help='Path to the combo_analysis_report.csv file',
                        default=r'C:\Users\Richard\OneDrive\GIT\CoMix\combined_analysis_report.csv')
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}\nPlease generate `combo_analysis_report.csv` or pass --csv <path>")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print('\nDataframe info:')
    print(df.info())

    print('\nDataframe description (numeric):')
    print(df.describe())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print('\nNo numeric columns found to aggregate.')
        return

    # --- Group by cluster_id ---
    if 'cluster_id' in df.columns:
        print('\n--- Analysis by Cluster ID (averaging all panels in each cluster) ---')
        gb_cluster = df.groupby('cluster_id')[numeric_cols].agg(['mean', 'std', 'count'])
        print(gb_cluster)
    else:
        print('\nNo `cluster_id` column found.')

    # --- Group by text_category ---
    if 'text_category' in df.columns:
        print('\n--- Analysis by Text Category (averaging all panels in each category) ---')
        # Also show the count of panels in each category
        gb_text = df.groupby('text_category')[numeric_cols].agg(['mean', 'std', 'count'])
        print(gb_text)
    else:
        print('\nNo `text_category` column found.')

    # --- Group by cluster_id and text_category ---
    if 'cluster_id' in df.columns and 'text_category' in df.columns:
        print('\n--- Combined Analysis by Cluster ID and Text Category ---')
        # Also show the count here for context
        gb_combo = df.groupby(['cluster_id', 'text_category'])['page_index'].count().unstack(fill_value=0)
        print("Panel counts per text category within each cluster:")
        print(gb_combo)

        print("\nMean panel_embedding_mean per text category within each cluster:")
        gb_combo_emb = df.groupby(['cluster_id', 'text_category'])['panel_embedding_mean'].mean().unstack(fill_value=0)
        print(gb_combo_emb)

if __name__ == '__main__':
    main()


