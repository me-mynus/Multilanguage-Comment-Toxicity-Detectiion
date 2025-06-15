from bs4 import BeautifulSoup
import re
import pandas as pd

def clean_text(text):
    """
    Clean the text by removing HTML tags, special characters, and extra whitespace
    Args:
        text (str): Input text to clean
    Returns:
        str: Cleaned text
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove special characters and numbers (keep letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s\.,!?\'"]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def prepare_data(df):
    """
    Prepare the dataframe by cleaning text and handling missing values
    Args:
        df (pandas.DataFrame): Input dataframe with 'comment_text' and 'toxic' columns
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    # Drop rows with missing values
    df = df.dropna(subset=['comment_text', 'toxic'])
    
    # Clean the text
    df['comment_text'] = df['comment_text'].apply(clean_text)
    
    # Remove empty strings after cleaning
    df = df[df['comment_text'].str.strip() != '']
    
    return df
