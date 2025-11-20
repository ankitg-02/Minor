import pandas as pd
from data_processing.text_cleaning import clean_text
df = pd.read_csv('data/raw/comments_Instagram_update_20251114_224320.csv')
print('Original df shape:', df.shape)
df = df.drop_duplicates(subset=['text']).dropna(subset=['text'])
print('After drop duplicates and na:', df.shape)
df['cleaned_text'] = df['text'].apply(clean_text)
print('After cleaning:', df.shape)
df = df[df['cleaned_text'].str.strip() != '']
print('After filtering empty:', df.shape)
print('Final df:')
print(df.head())
