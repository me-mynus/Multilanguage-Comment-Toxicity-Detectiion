import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from langdetect import detect
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import sys

# Import datasets
try:
    from data_adjust import df_train, df_test, df_validation
except Exception as e:
    print(f"Error loading datasets: {str(e)}")
    sys.exit(1)

class ToxicityEDA:
    def __init__(self):
        """
        Initialize EDA class with project-based directory structure
        """
        # Get the project root directory (parent of src)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Create eda_results directory inside project root
        self.save_dir = os.path.join(self.project_root, 'eda_results')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"EDA results will be saved to: {self.save_dir}")
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
            print("Some features might be limited.")

    def validate_dataframe(self, df, title):
        """Validate dataframe has required columns"""
        required_columns = ['comment_text', 'toxic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"{title} dataset is missing required columns: {missing_columns}")
        
        if not len(df):
            raise ValueError(f"{title} dataset is empty")
        
        return True

    def plot_class_distribution(self, df, title, filename):
        """Plot distribution of toxic vs non-toxic comments"""
        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x='toxic')
            plt.title(f'Class Distribution - {title}')
            plt.xlabel('Toxic')
            plt.ylabel('Count')
            
            # Add percentage labels
            total = len(df)
            for i in range(2):
                count = len(df[df['toxic'] == i])
                pct = count/total * 100
                plt.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
            
            plt.savefig(os.path.join(self.save_dir, f'{filename}_class_dist.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting class distribution for {title}: {str(e)}")

    def plot_text_length_distribution(self, df, title, filename):
        """Plot distribution of text lengths"""
        try:
            df = df.copy()  # Create copy to avoid modifying original
            df['text_length'] = df['comment_text'].str.len()
            
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df, x='text_length', hue='toxic', bins=50, multiple="stack")
            plt.title(f'Text Length Distribution - {title}')
            plt.xlabel('Text Length')
            plt.ylabel('Count')
            plt.savefig(os.path.join(self.save_dir, f'{filename}_length_dist.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting text length distribution for {title}: {str(e)}")

    def plot_language_distribution(self, df, title, filename):
        """Plot distribution of languages"""
        try:
            df = df.copy()  # Create copy to avoid modifying original
            if 'lang' not in df.columns:
                print(f"Computing language distribution for {title} dataset (this may take a while)...")
                tqdm.pandas()
                df['lang'] = df['comment_text'].progress_apply(
                    lambda x: detect(x) if isinstance(x, str) and len(str(x).strip()) > 0 else 'unknown'
                )
            
            plt.figure(figsize=(12, 6))
            lang_counts = df['lang'].value_counts()
            lang_counts.plot(kind='bar')
            plt.title(f'Language Distribution - {title}')
            plt.xlabel('Language')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{filename}_lang_dist.png'))
            plt.close()
        except Exception as e:
            print(f"Error plotting language distribution for {title}: {str(e)}")

    def calculate_metrics(self, df, title):
        """Calculate various metrics about the dataset"""
        try:
            metrics = {
                'Total Samples': len(df),
                'Toxic Samples': int(df['toxic'].sum()),
                'Non-toxic Samples': int(len(df) - df['toxic'].sum()),
                'Toxic Ratio': float(df['toxic'].mean()),
                'Average Text Length': float(df['comment_text'].str.len().mean()),
                'Median Text Length': float(df['comment_text'].str.len().median()),
                'Max Text Length': int(df['comment_text'].str.len().max()),
                'Min Text Length': int(df['comment_text'].str.len().min()),
            }
            
            # Save metrics to file
            with open(os.path.join(self.save_dir, f'{title.lower()}_metrics.txt'), 'w') as f:
                for metric, value in metrics.items():
                    f.write(f'{metric}: {value}\n')
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics for {title}: {str(e)}")
            return {}

    def analyze_dataset(self, df, title, filename):
        """Perform complete analysis on a dataset"""
        try:
            print(f"\nAnalyzing {title} dataset...")
            
            # Validate dataframe
            self.validate_dataframe(df, title)
            
            # Calculate and print metrics
            metrics = self.calculate_metrics(df, title)
            if metrics:
                print("\nDataset Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
            
            # Generate all plots
            print(f"\nGenerating plots for {title} dataset...")
            self.plot_class_distribution(df, title, filename)
            self.plot_text_length_distribution(df, title, filename)
            self.plot_language_distribution(df, title, filename)
            
        except Exception as e:
            print(f"Error analyzing {title} dataset: {str(e)}")

    def analyze_all_datasets(self):
        """Analyze all datasets (train, test, validation)"""
        datasets = [
            (df_train, "Training", "train"),
            (df_test, "Test", "test"),
            (df_validation, "Validation", "val")
        ]
        
        for df, title, filename in datasets:
            self.analyze_dataset(df, title, filename)

def main():
    try:
        # Initialize EDA
        eda = ToxicityEDA()
        
        # Run analysis on all datasets
        print("Starting Exploratory Data Analysis...")
        eda.analyze_all_datasets()
        
        print(f"\nEDA completed! Results saved in '{eda.save_dir}' directory")
        
    except Exception as e:
        print(f"An error occurred during EDA: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
