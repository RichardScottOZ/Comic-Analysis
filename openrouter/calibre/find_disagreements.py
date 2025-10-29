import argparse
import pandas as pd
import os

def find_disagreements(file_path):
    """
    Reads a comic analysis CSV file, finds rows where rcnn_panels and vlm_panels disagree,
    and saves them to a new compressed CSV file.
    """
    try:
        print(f"\n--- Processing file: {file_path} ---")
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ['rcnn_panels', 'vlm_panels']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Input file must contain the columns: {required_cols}")
            return

        # Find disagreements
        disagreements_df = df[df['rcnn_panels'] != df['vlm_panels']].copy()

        total_rows = len(df)
        disagreement_count = len(disagreements_df)
        agreement_count = total_rows - disagreement_count
        agreement_percentage = (agreement_count / total_rows) * 100 if total_rows > 0 else 0

        print(f"Total rows processed: {total_rows}")
        print(f"Rows with panel count disagreement: {disagreement_count}")
        print(f"Rows with panel count agreement: {agreement_count}")
        print(f"Agreement Percentage: {agreement_percentage:.2f}%")

        if disagreement_count > 0:
            # Save the disagreements to a new compressed file
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_filename = f"{file_name_without_ext}_disagreements.csv.gz"
            
            output_dir = os.path.dirname(file_path)
            output_path = os.path.join(output_dir, output_filename)
            
            disagreements_df.to_csv(output_path, index=False, compression='gzip')
            print(f"Saved {disagreement_count} disagreement rows to: {output_path}")
        else:
            print("No disagreements found.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find disagreements in panel counts between R-CNN and VLM.")
    parser.add_argument("file_paths", nargs='+', help="One or more absolute paths to the input CSV files.")
    args = parser.parse_args()

    for path in args.file_paths:
        find_disagreements(path)