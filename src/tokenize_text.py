import torch
import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, AutoTokenizer
from config import train_data_path, test_data_path, test_label_path, validation_path
from preprocessor import prepare_train_data, prepare_test_validation_data
from tqdm import tqdm

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

class ToxicityDataTokenizer:
    def __init__(self, model_name="xlm-roberta-base", max_length=128):
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
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def prepare_dataset(self, df):
        """
        Convert dataframe to Dataset and tokenize
        Args:
            df (pandas.DataFrame): Input dataframe
        Returns:
            datasets.Dataset: Tokenized dataset
        """
        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

def main():
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load and preprocess data
    df_train, df_test, df_validation = load_and_preprocess_data()

    # Initialize tokenizer
    tokenizer = ToxicityDataTokenizer()

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
    train_dataset, test_dataset, validation_dataset = main()
