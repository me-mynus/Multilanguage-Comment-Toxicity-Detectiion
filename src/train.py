import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
from sklearn.metrics import classification_report
from database import df_train, df_test, df_validation

# Enable memory efficient loading
torch.cuda.empty_cache()  # Clear GPU memory if available




train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)
validation_dataset = Dataset.from_pandas(df_validation)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_function(examples):
    return tokenizer(
        examples['comment_text'],
        truncation=True
    )

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True
)

tokenized_validation = validation_dataset.map(
    preprocess_function,
    batched=True
)


model = AutoModelForMaskedLM.from_pretrained(
    "xlm-roberta-base"
)
