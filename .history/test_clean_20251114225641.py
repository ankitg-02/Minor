import pandas as pd
from data_processing.text_cleaning import clean_text
df = pd.read_csv('data/raw/comments_Instagram_update_20251114_224320.csv')
print('Before cleaning:')
print(df['text'].head())
df['cleaned_text'] = df['text'].apply(clean_text)
print('After cleaning:')
print(df['cleaned_text'].head())
