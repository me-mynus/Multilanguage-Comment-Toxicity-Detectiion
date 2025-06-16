import config
import pandas as pd

def load_datasets():
    """
    Load and prepare all datasets
    Returns:
        tuple: (df_train, df_test, df_validation)
    """
    # Load training data
    df_train = pd.read_csv(config.train_data_path, usecols=['comment_text', 'toxic'])

    # Load and merge test data
    df1_test = pd.read_csv(config.test_data_path, usecols=['id', 'content'])
    df2_test = pd.read_csv(config.test_label_path)
    df_test = pd.merge(df1_test, df2_test, on='id', how='left')
    df_test = df_test.rename(columns={'content': 'comment_text'})
    df_test = df_test[['comment_text', 'toxic']]

    # Load validation data
    df_validation = pd.read_csv(config.validation_path, usecols=['comment_text', 'toxic'])

    # Convert toxic columns to int if they aren't already
    df_train['toxic'] = df_train['toxic'].astype(int)
    df_test['toxic'] = df_test['toxic'].astype(int)
    df_validation['toxic'] = df_validation['toxic'].astype(int)

    return df_train, df_test, df_validation

# Load datasets when module is imported
df_train, df_test, df_validation = load_datasets()

