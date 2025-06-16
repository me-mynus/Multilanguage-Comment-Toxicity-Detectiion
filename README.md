# Multilingual Comment Toxicity Detection

This project implements a multilingual toxic comment detection system using XLM-RoBERTa. It includes data processing, exploratory data analysis (EDA), and model training pipelines for detecting toxic comments across multiple languages.

## Project Structure

```
Multilanguage-Comment-Toxicity-Detectiion/
├── README.md
├── requirements.txt
├── dataset/
│   ├── jigsaw-toxic-comment-train.csv
│   ├── test_labels.csv
│   ├── test.csv
│   └── validation.csv
├── models/                    # Directory for saved models
└── src/
    ├── config.py             # Configuration settings
    ├── data_adjust.py        # Dataset loading and preparation
    ├── preprocessor.py       # Text preprocessing pipeline
    ├── tokenize.py          # Text tokenization using XLM-RoBERTa
    └── eda.py               # Exploratory Data Analysis scripts
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/me-mynus/Multilanguage-Comment-Toxicity-Detectiion.git
cd Multilanguage-Comment-Toxicity-Detectiion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

### Data Processing
- Robust data loading and merging of train, test, and validation datasets
- Text cleaning and preprocessing (HTML removal, special character handling)
- Tokenization using XLM-RoBERTa for multilingual support

### Exploratory Data Analysis (EDA)
The EDA pipeline (`src/eda.py`) generates the following analyses:

1. Dataset Metrics:
   - Total samples
   - Toxic vs non-toxic distribution
   - Text length statistics
   - Language distribution

2. Visualizations:
   - Class distribution plots
   - Text length distribution histograms
   - Language distribution charts

All EDA results are saved in the `eda_results/` directory.

### Model Training
- Fine-tuning XLM-RoBERTa for toxicity detection
- Local model checkpointing
- Performance metrics tracking

## Usage

### Running EDA

```bash
python src/eda.py
```

The script will:
1. Load and validate all datasets
2. Generate metrics and visualizations
3. Save results in the `eda_results/` directory

### Training the Model

```bash
python src/model_trainer.py
```

The model will be trained using the preprocessed data and saved in the `models/` directory.

## Data Requirements

Ensure the following files are present in the `dataset/` directory:
- `jigsaw-toxic-comment-train.csv`
- `test.csv`
- `test_labels.csv`
- `validation.csv`

## Notes

- The project uses local file storage for all outputs and models
- Progress bars and status messages keep you informed of long-running processes
