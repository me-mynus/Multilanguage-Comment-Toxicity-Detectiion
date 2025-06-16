import torch
import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, AutoTokenizer
from config import train_data_path, test_data_path, test_label_path, validation_path
from preprocessor import get_preprocessed_data
from tqdm import tqdm
import numpy as np

class ToxicityTokenizer:
    def __init__(self, model_name="xlm-roberta-base", max_length=128):
        """
        Initialize the tokenizer
        Args:
            model_name (str): Name of the pretrained model to use
            max_length (int): Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_batch(self, examples):
        """
        Tokenize a batch of examples
        Args:
            examples: Batch of examples from Dataset
        Returns:
            dict: Tokenized batch
        """
        return self.tokenizer(
            examples['clean_text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Return lists instead of tensors
        )

    def prepare_dataset(self, df, dataset_name=""):
        """
        Convert dataframe to tokenized Dataset
        Args:
            df (pandas.DataFrame): Input dataframe
            dataset_name (str): Name of the dataset for progress info
        Returns:
            datasets.Dataset: Tokenized dataset
        """
        print(f"\nPreparing {dataset_name} dataset...")
        
        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {dataset_name} data"
        )
        
        # Convert labels to float values
        labels = df['toxic'].astype(float).values
        tokenized_dataset = tokenized_dataset.add_column("labels", labels)
        
        return tokenized_dataset

def load_and_preprocess_data():
    """
    Load and preprocess all datasets
    Returns:
        tuple: Preprocessed train, test, and validation datasets
    """
    print("Loading and preprocessing training data...")
    df_train = pd.read_csv(train_data_path)
    df_train = prepare_train_data(df_train)

    print("Loading and preprocessing test data...")
    df_test = pd.read_csv(test_data_path)
    df_test_labels = pd.read_csv(test_label_path)
    df_test = pd.merge(df_test, df_test_labels[['id', 'toxic']], on='id', how='left')
    df_test = prepare_test_validation_data(df_test)

    print("Loading and preprocessing validation data...")
    df_validation = pd.read_csv(validation_path)
    df_validation = prepare_test_validation_data(df_validation)

    return df_train, df_test, df_validation

def get_tokenized_data():
    """
    Get tokenized versions of all datasets
    Returns:
        tuple: (train_tokenized, test_tokenized, validation_tokenized)
    """
    # Get preprocessed data
    train_processed, test_processed, validation_processed = get_preprocessed_data()
    
    # Initialize tokenizer
    tokenizer = ToxicityTokenizer()
    
    # Tokenize all datasets
    train_tokenized = tokenizer.prepare_dataset(train_processed, "training")
    test_tokenized = tokenizer.prepare_dataset(test_processed, "test")
    validation_tokenized = tokenizer.prepare_dataset(validation_processed, "validation")
    
    # Print tokenization results
    print("\nTokenization Results:")
    print(f"Training data: {len(train_tokenized)} sequences")
    print(f"Test data: {len(test_tokenized)} sequences")
    print(f"Validation data: {len(validation_tokenized)} sequences")
    
    # Calculate sequence length statistics
    train_lengths = [len(x) for x in train_tokenized['input_ids']]
    print(f"\nSequence length statistics:")
    print(f"Mean: {np.mean(train_lengths):.1f}")
    print(f"Median: {np.median(train_lengths):.1f}")
    print(f"95th percentile: {np.percentile(train_lengths, 95):.1f}")
    print(f"Max: {np.max(train_lengths)}")
    
    return train_tokenized, test_tokenized, validation_tokenized

def main():
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load and preprocess data
    df_train, df_test, df_validation = load_and_preprocess_data()

    # Initialize tokenizer
    tokenizer = ToxicityTokenizer()

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = tokenizer.prepare_dataset(df_train)
    test_dataset = tokenizer.prepare_dataset(df_test)
    validation_dataset = tokenizer.prepare_dataset(df_validation)

    print("Dataset statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    return train_dataset, test_dataset, validation_dataset

if __name__ == "__main__":
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Get tokenized datasets
    train_tokenized, test_tokenized, validation_tokenized = get_tokenized_data()
    
    print("\nSample tokenized data:")
    print("\nTraining sample:")
    print(train_tokenized[0])
    print("\nTest sample:")
    print(test_tokenized[0])
    print("\nValidation sample:")
    print(validation_tokenized[0])
