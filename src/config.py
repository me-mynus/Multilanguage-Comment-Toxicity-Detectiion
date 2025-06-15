import os

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define dataset paths
test_data_path = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
test_label_path = os.path.join(PROJECT_ROOT, "dataset", "test_labels.csv")
validation_path = os.path.join(PROJECT_ROOT, "dataset", "validation.csv")
train_data_path = os.path.join(PROJECT_ROOT, "dataset", "jigsaw-toxic-comment-train.csv")