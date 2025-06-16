import torch
from transformers import (
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tokenize_text import get_tokenized_data
import os
import numpy as np

def compute_metrics(pred):
    """
    Compute metrics for evaluation
    """
    # Convert predictions to binary labels
    predictions = (pred.predictions.squeeze() >= 0.5).astype(float)
    labels = pred.label_ids.astype(float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(train_dataset, eval_dataset, model_dir='models'):
    """
    Train the toxicity detection model
    Args:
        train_dataset: Tokenized training dataset
        eval_dataset: Tokenized validation dataset
        model_dir: Directory to save the model
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Initialize model
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-base',
        num_labels=1,
        problem_type="regression"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        seed=42,
        report_to="none"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the best model
    best_model_path = os.path.join(model_dir, "best_model")
    trainer.save_model(best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    # Final evaluation
    print("\nPerforming final evaluation...")
    final_metrics = trainer.evaluate()
    
    print("\nFinal Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    return final_metrics

def main():
    try:
        # Get tokenized data
        print("Loading tokenized data...")
        train_dataset, test_dataset, validation_dataset = get_tokenized_data()

        # Train model
        train_model(train_dataset, validation_dataset)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
