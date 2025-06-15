from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

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
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def detect_language(text):
    """
    Detect the language of the text
    Args:
        text (str): Input text
    Returns:
        str: Language code
    """
    try:
        return detect(text)
    except:
        return 'unknown'

def prepare_train_data(df):
    """
    Prepare the training dataframe
    Args:
        df (pandas.DataFrame): Input dataframe with toxic columns
    Returns:
        pandas.DataFrame: Cleaned dataframe with binary labels
    """
    # Drop rows with missing values
    df = df.dropna(subset=['comment_text'])
    
    # Clean the text
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['comment_text'].progress_apply(clean_text)
    
    # Create binary toxicity label (1 if any toxic label is 1)
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['toxic'] = df[toxic_columns].max(axis=1)
    
    # Detect language
    tqdm.pandas(desc="Detecting language")
    df['lang'] = df['clean_text'].progress_apply(detect_language)
    
    # Remove empty strings after cleaning
    df = df[df['clean_text'].str.strip() != '']
    
    return df[['clean_text', 'lang', 'toxic']]

def prepare_test_validation_data(df):
    """
    Prepare the test/validation dataframe
    Args:
        df (pandas.DataFrame): Input dataframe with comment_text/content column
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    # Handle different column names
    text_col = 'comment_text' if 'comment_text' in df.columns else 'content'
    df = df.rename(columns={text_col: 'clean_text'})
    
    # Drop rows with missing values
    df = df.dropna(subset=['clean_text'])
    
    # Clean the text
    tqdm.pandas(desc="Cleaning text")
    df['clean_text'] = df['clean_text'].progress_apply(clean_text)
    
    # Add language if not present
    if 'lang' not in df.columns:
        tqdm.pandas(desc="Detecting language")
        df['lang'] = df['clean_text'].progress_apply(detect_language)
    
    # Remove empty strings after cleaning
    df = df[df['clean_text'].str.strip() != '']
    
    return df[['clean_text', 'lang', 'toxic'] if 'toxic' in df.columns else ['clean_text', 'lang']]
