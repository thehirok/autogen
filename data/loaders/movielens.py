"""
MovieLens dataset loader.
Supports ML-100K, ML-1M, ML-10M, ML-20M.

Usage:
    from data.loaders.movielens import load_movielens
    df = load_movielens('data/raw/movielens_1m/ratings.dat')
    df.to_parquet('data/processed/movielens_1m.parquet', index=False)
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_movielens(path):
    """
    Auto-detect MovieLens format and return a clean DataFrame
    with columns [user_id, item_id, rating].

    Parameters
    ----------
    path : str   path to the ratings file

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext  = os.path.splitext(path)[1].lower()
    name = os.path.basename(path).lower()

    # ML-100K: u.data  tab-separated, no header
    if 'u.data' in name:
        df = pd.read_csv(
            path, sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'])

    # ML-1M / ML-10M: ratings.dat  "::" separated, no header
    elif name.endswith('.dat'):
        df = pd.read_csv(
            path, sep='::', engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'])

    # ML-20M / ML-25M: ratings.csv  comma-separated, with header
    elif ext == '.csv':
        df = pd.read_csv(path)
        df = df.rename(columns={
            'userId':  'user_id',
            'movieId': 'item_id',
        })

    else:
        raise ValueError(
            f"Unrecognised MovieLens file format: {name}\n"
            "Expected: u.data, ratings.dat, or ratings.csv")

    df = df[['user_id', 'item_id', 'rating']].dropna()
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['rating']  = df['rating'].astype(float)

    logger.info(
        f"Loaded MovieLens from {path}: "
        f"{len(df):,} ratings, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['item_id'].nunique():,} items")
    return df


def prepare_movielens(raw_path, out_path):
    """
    Load, clean, and save as parquet.

    Parameters
    ----------
    raw_path : str   path to the raw ratings file
    out_path : str   path to save the parquet file
    """
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = load_movielens(raw_path)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved → {out_path}")
    return df


if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        print("Usage: python movielens.py <raw_path> <out_parquet>")
        sys.exit(1)

    prepare_movielens(sys.argv[1], sys.argv[2])