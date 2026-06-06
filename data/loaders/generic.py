"""
Generic loader for any CSV/TSV dataset that has
user_id, item_id, rating columns (or equivalents).

Usage:
    from data.loaders.generic import load_generic
    df = load_generic(
        'data/raw/epinions/ratings.txt',
        sep=' ',
        user_col='user', item_col='item', rating_col='stars'
    )
    df.to_parquet('data/processed/epinions.parquet', index=False)
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_generic(path, sep=',', user_col='user_id',
                 item_col='item_id', rating_col='rating',
                 header='infer', names=None,
                 min_rating=None, max_rating=None,
                 min_user_ratings=0, min_item_ratings=0):
    """
    Load any tabular ratings file into a standard DataFrame.

    Parameters
    ----------
    path             : str
    sep              : str    delimiter
    user_col         : str    column name for users
    item_col         : str    column name for items
    rating_col       : str    column name for ratings
    header           : 'infer' or None
    names            : list   column names if header=None
    min_rating       : float  clip ratings below this
    max_rating       : float  clip ratings above this
    min_user_ratings : int    drop users with fewer ratings
    min_item_ratings : int    drop items with fewer ratings

    Returns
    -------
    pd.DataFrame  columns [user_id, item_id, rating]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=sep, header=header,
                     names=names, engine='python')

    # rename to standard columns
    rename = {}
    if user_col   != 'user_id':   rename[user_col]   = 'user_id'
    if item_col   != 'item_id':   rename[item_col]   = 'item_id'
    if rating_col != 'rating':    rename[rating_col]  = 'rating'
    if rename:
        df = df.rename(columns=rename)

    df = df[['user_id', 'item_id', 'rating']].dropna()
    df['rating'] = df['rating'].astype(float)

    # clip rating range
    if min_rating is not None:
        df = df[df['rating'] >= min_rating]
    if max_rating is not None:
        df = df[df['rating'] <= max_rating]

    # encode string IDs if necessary
    if df['user_id'].dtype == object:
        df['user_id'] = pd.Categorical(df['user_id']).codes
    else:
        df['user_id'] = df['user_id'].astype(int)

    if df['item_id'].dtype == object:
        df['item_id'] = pd.Categorical(df['item_id']).codes
    else:
        df['item_id'] = df['item_id'].astype(int)

    # filter low-activity users/items
    if min_user_ratings > 0:
        counts = df.groupby('user_id').size()
        df = df[df['user_id'].isin(counts[counts >= min_user_ratings].index)]
    if min_item_ratings > 0:
        counts = df.groupby('item_id').size()
        df = df[df['item_id'].isin(counts[counts >= min_item_ratings].index)]

    df = df.reset_index(drop=True)

    logger.info(
        f"Loaded {path}: {len(df):,} ratings, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['item_id'].nunique():,} items")
    return df


def prepare_generic(raw_path, out_path, **kwargs):
    """
    Load, clean, and save as parquet.
    All kwargs are passed to load_generic().
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = load_generic(raw_path, **kwargs)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved → {out_path}")
    return df


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python generic.py <raw_path> <out_parquet>")
        sys.exit(1)
    prepare_generic(sys.argv[1], sys.argv[2])