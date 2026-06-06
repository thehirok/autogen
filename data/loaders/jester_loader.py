import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

dfs = []
for fname in ['jester-data-1.xls', 'jester-data-2.xls']:
    path = os.path.join('data', 'raw', 'jester', fname)
    print(f"Loading {path}...")
    df = pd.read_excel(path, header=None)
    ratings = df.iloc[:, 1:]
    ratings.replace(99, np.nan, inplace=True)
    for user_idx, row in ratings.iterrows():
        for item_idx, val in row.items():
            if not np.isnan(val):
                dfs.append({
                    'user_id': user_idx,
                    'item_id': item_idx,
                    'rating':  float(val)
                })

out = pd.DataFrame(dfs)
os.makedirs('data/processed', exist_ok=True)
out.to_parquet('data/processed/jester.parquet', index=False)
print(f"Jester: {len(out):,} ratings, {out['user_id'].nunique():,} users, {out['item_id'].nunique():,} items")
print("Saved -> data/processed/jester.parquet")