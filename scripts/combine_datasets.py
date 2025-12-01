# scripts/combine_datasets.py
import pandas as pd
import sys
import os

def combine_datasets(base_path, incremental_path, output_path):
    """Combines a base labeled dataset with an incremental dataset."""
    try:
        base_df = pd.read_csv(base_path)
        print(f"Loaded base dataset with {len(base_df)} prompts.")
    except FileNotFoundError:
        print(f"Warning: Base dataset '{base_path}' not found. Starting with incremental data.")
        base_df = pd.DataFrame()

    try:
        incremental_df = pd.read_csv(incremental_path)
        print(f"Loaded incremental dataset with {len(incremental_df)} prompts.")
    except FileNotFoundError:
        print(f"Error: Incremental dataset '{incremental_path}' not found. Cannot combine.")
        sys.exit(1)
    
    # Drop the 'source' column if it exists in incremental data, as it's not a label
    if 'source' in incremental_df.columns:
        incremental_df = incremental_df.drop(columns=['source'])

    # Ensure all label columns exist in both DataFrames before concatenating
    label_cols = [
        'harm_prevention', 'fairness_discrimination', 'privacy_violation',
        'transparency_deception', 'accountability_misuse', 'safe_compliant'
    ]
    for col in label_cols:
        if col not in base_df.columns:
            base_df[col] = 0 # Default to 0
        if col not in incremental_df.columns:
            incremental_df[col] = 0 # Default to 0

    combined_df = pd.concat([base_df, incremental_df], ignore_index=True)
    
    # Remove duplicate prompts to ensure unique entries
    combined_df.drop_duplicates(subset=['prompt'], inplace=True)
    
    # Ensure all required label columns are present before saving
    final_label_cols = ['harm_prevention', 'fairness_discrimination', 'privacy_violation',
                        'transparency_deception', 'accountability_misuse']
    if 'safe_compliant' in combined_df.columns:
        final_label_cols.append('safe_compliant')
    
    # Ensure all final_label_cols are integers
    for col in final_label_cols:
        combined_df[col] = combined_df[col].astype(int)

    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to '{output_path}' with {len(combined_df)} unique prompts.")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python combine_datasets.py <base_csv_path> <incremental_csv_path> <output_csv_path>")
        sys.exit(1)
    combine_datasets(sys.argv[1], sys.argv[2], sys.argv[3])
