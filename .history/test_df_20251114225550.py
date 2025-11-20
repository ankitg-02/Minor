import pandas as pd
df = pd.read_csv('data/raw/comments_Instagram_update_20251114_224320.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Text column type:', df['text'].dtype)
print('Sample text:')
print(df['text'].head())
