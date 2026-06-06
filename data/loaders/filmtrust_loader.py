import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

path = os.path.join('data', 'raw', 'filmtrust', 'ratings.txt')
df   = pd.read_csv(path, sep=' ', header=None,
                   names=['user_id', 'item_id', 'rating', 'timestamp'])
df   = df[['user_id', 'item_id', 'rating']].dropna()
df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)
df['rating']  = df['rating'].astype(float)

os.makedirs('data/processed', exist_ok=True)
df.to_parquet('data/processed/filmtrust.parquet', index=False)
print(f"FilmTrust: {len(df):,} ratings, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
print("Saved -> data/processed/filmtrust.parquet")