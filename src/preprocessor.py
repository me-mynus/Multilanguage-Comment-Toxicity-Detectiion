from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import re
import pandas as pd
from tqdm import tqdm
import warnings
import html
from data_adjust import df_train, df_test, df_validation

# Suppress BeautifulSoup warnings about markup resembling a URL
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

def clean_text(text):
    """
    Clean the text by removing HTML tags, special characters, and extra whitespace
    Args:
        text (str): Input text to clean
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # First, unescape HTML entities
    text = html.unescape(text)
    
    try:
        # Remove HTML tags safely
        soup = BeautifulSoup(text, "html.parser", from_encoding='utf-8')
        text = soup.get_text(separator=' ')
    except Exception as e:
        # If BeautifulSoup fails, use regex as fallback
        text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers (keep letters and basic punctuation)
    text = re.sub(r'[^a-zA-Z\s\.,!?\'"]+', ' ', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def preprocess_data(df, dataset_name=""):
    """
    Preprocess the dataframe
    Args:
        df (pandas.DataFrame): Input dataframe with comment_text and toxic columns
        dataset_name (str): Name of the dataset for progress bar
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    print(f"\nPreprocessing {dataset_name} data...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Drop rows with missing values
    df = df.dropna(subset=['comment_text'])
    
    # Clean the text
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['comment_text'].progress_apply(clean_text)
    
    # Remove empty strings after cleaning
    df = df[df['clean_text'].str.strip() != '']
    
    # Keep only required columns
    return df[['clean_text', 'toxic']]

def get_preprocessed_data():
    """
    Get preprocessed versions of all datasets
    Returns:
        tuple: (train_processed, test_processed, validation_processed)
    """
    
    
    # Process all datasets
    train_processed = preprocess_data(df_train, "training")
    test_processed = preprocess_data(df_test, "test")
    validation_processed = preprocess_data(df_validation, "validation")
    
    # Print preprocessing results
    print("\nPreprocessing Results:")
    print(f"Training data: {len(df_train)} -> {len(train_processed)} samples")
    print(f"Test data: {len(df_test)} -> {len(test_processed)} samples")
    print(f"Validation data: {len(df_validation)} -> {len(validation_processed)} samples")
    
    return train_processed, test_processed, validation_processed

if __name__ == "__main__":
    # If run directly, preprocess and show sample results
    train_processed, test_processed, validation_processed = get_preprocessed_data()
    
    print("\nSample processed data:")
    print("\nTraining sample:")
    print(train_processed.head())
    print("\nTest sample:")
    print(test_processed.head())
    print("\nValidation sample:")
    print(validation_processed.head())
