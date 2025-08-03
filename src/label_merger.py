"""
Utility to merge gold labels into CSV files
"""
import pandas as pd
import ast

def merge_labels_into_csv(csv_path, labels_dict_str, output_path=None):
    """
    Merge gold labels into the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        labels_dict_str: String representation of the labels dictionary
        output_path: Where to save the labeled CSV (default: adds '_labeled' suffix)
    
    Returns:
        DataFrame with labels merged
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Parse the labels dictionary
    labels_dict = ast.literal_eval(labels_dict_str)
    print(f"Loaded {len(labels_dict)} labels")
    
    # Create index column if it doesn't exist (1-based indexing)
    if 'Index' not in df.columns:
        df['Index'] = range(1, len(df) + 1)
    
    # Map the labels
    df['Label'] = df['Index'].map(labels_dict)
    
    # Check for missing labels
    missing = df['Label'].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have no labels!")
    
    # Ensure labels are integers
    df['Label'] = df['Label'].astype('Int64')
    
    # Save the result
    if output_path is None:
        output_path = csv_path.replace('.csv', '_labeled.csv')
    
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\nLabel distribution:")
    print(df['Label'].value_counts().sort_index())
    print(f"\nSaved to: {output_path}")
    
    return df
