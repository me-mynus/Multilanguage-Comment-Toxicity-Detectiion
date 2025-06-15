import config
import pandas as pd

print(config.train_data_path)
df_train = pd.read_csv(config.train_data_path, usecols=['comment_text', 'toxic'])


df1_test = pd.read_csv(config.test_data_path, usecols=['id', 'content'])
df2_test = pd.read_csv(config.test_label_path)

df_test = pd.merge(df1_test, df2_test, on='id', how='left')
df_test = df_test.rename(columns={'content': 'comment_text', 'toxic': 'toxic'})
df_test = df_test[['comment_text', 'toxic']]

df_validation = pd.read_csv(config.validation_path, usecols=['comment_text', 'toxic'])

print(df_train.head())
print(df_test.head())
print(df_validation.head()) 

