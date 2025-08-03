"""
Feature engineering utilities for combining title and description
"""

def combine_text(row, title_weight=3):
    """
    Combine title and description with title having more weight
    
    Args:
        row: DataFrame row with 'Job_Title_Clean' and 'Job_Description_Clean'
        title_weight: How many times to repeat the title (default: 3)
    
    Returns:
        Combined text string
    """
    title = row.get('Job_Title_Clean', '')
    description = row.get('Job_Description_Clean', '')
    
    # Title appears 3 times to give it more weight
    return(title + " ") * title_weight + description

def prepare_features(df, title_weight=3):
    """
    Prepare text features for model training
    
    Args:
        df: DataFrame with cleaned text columns
        title_weight: Weight for title in combined text
    
    Returns:
        Series of combined text features
    """
    return df.apply(lambda row: combine_text(row, title_weight), axis=1)
