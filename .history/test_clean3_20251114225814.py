import pandas as pd
from data_processing.text_cleaning import clean_text
df = pd.read_csv('data/raw/comments_Instagram_update_20251114_224320.csv')
df['cleaned_text'] = df['text'].apply(clean_text)
print('After cleaning:')
print(df['cleaned_text'].head())
print('Empty cleaned texts:')
empty_df = df[df['cleaned_text'].str.strip() == '']
print('Number of empty:', len(empty_df))
print('Total rows:', len(df))
print('Non-empty:', len(df) - len(empty_df))
