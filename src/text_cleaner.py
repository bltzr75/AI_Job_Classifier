"""
Text cleaning utilities for job description preprocessing
"""
import re
import pandas as pd

def clean_text_for_classification(text, is_title=False):
    """
    Cleaning preserving keywords and removing noise.
    
    Args:
        text: Input text to clean
        is_title: Boolean indicating if this is a job title (less aggressive cleaning)
    
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return("")
    
    text = str(text)
    
    # Remove common prefixes
    if not is_title:
        text = re.sub(r'^About the job\s*', '', text, flags=re.IGNORECASE) # \s* is 0 or + whitespaces (tabs, spaces, newlines)
        text = re.sub(r'^Job Description\s*', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace (including spaces, newlines, tabs)
    text = re.sub(r'\s+', ' ', text) # replace 1 or more spaces with just ' '
    
    # Remove excessive punctuation but keeping structure
    text = re.sub(r'([.!?¿])\1+', r'\1', text)
    text = re.sub(r'[-=_]{3,}', ' ', text)
    
    # Remove bullet points but keep the text
    text = re.sub(r'^\s*[•·▪▫◆◇○●\-\*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def preprocess_dataframe(df):
    """
    Apply text cleaning to a dataframe with job data
    
    Args:
        df: DataFrame with 'Job_Title' and 'Job_Description' columns
    
    Returns:
        DataFrame with added 'Job_Title_Clean' and 'Job_Description_Clean' columns
    """
    df = df.copy()
    
    # Clean titles and descriptions
    df['Job_Title_Clean'] = df['Job_Title'].apply(lambda x: clean_text_for_classification(x, is_title=True))
    df['Job_Description_Clean'] = df['Job_Description'].apply(lambda x: clean_text_for_classification(x, is_title=False))
    
    # Handle missing values
    df['Job_Description_Clean'] = df['Job_Description_Clean'].fillna('')
    
    # Calculate text lengths
    df['title_len'] = df['Job_Title_Clean'].str.len()
    df['desc_len'] = df['Job_Description_Clean'].str.len()
    
    return df
