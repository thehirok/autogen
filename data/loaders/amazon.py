"""
Amazon Reviews dataset loader.
Supports the UCSD Amazon Review datasets (2014, 2018, 2023).
Download from: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/

Usage:
    from data.loaders.amazon import load_amazon
    df = load_amazon('data/raw/amazon_books/Books.json.gz')
    df.to_parquet('data/processed/amazon_books.parquet', index=False)
"""

import os
import json
import gzip
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_amazon(path, max_rows=None):
    """
    Load Amazon Reviews from a .json or .json.gz file.

    Parameters
    ----------
    path     : str   path to the raw file
    max_rows : int   optional row limit (useful for testing)

    Returns
    -------
    pd.DataFrame  columns [user_id, item_id, rating]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.lower()
    records = []

    opener = gzip.open if ext.endswith('.gz') else open

    with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            try:
                obj = json.loads(line.strip())
                # 2014 / 2018 format
                reviewer = obj.get('reviewerID') or obj.get('user_id')
                product  = obj.get('asin')       or obj.get('parent_asin')
                rating   = obj.get('overall')    or obj.get('rating')
                if reviewer and product and rating is not None:
                    records.append({
                        'user_id': reviewer,
                        'item_id': product,
                        'rating':  float(rating),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

    if not records:
        raise ValueError(f"No valid records found in {path}.")

    df = pd.DataFrame(records)

    # encode string IDs to integers
    df['user_id'] = pd.Categorical(df['user_id']).codes
    df['item_id'] = pd.Categorical(df['item_id']).codes

    logger.info(
        f"Loaded Amazon from {path}: "
        f"{len(df):,} ratings, "
        f"{df['user_id'].nunique():,} users, "
        f"{df['item_id'].nunique():,} items")
    return df


def prepare_amazon(raw_path, out_path, max_rows=None):
    """
    Load, clean, and save as parquet.

    Parameters
    ----------
    raw_path : str
    out_path : str
    max_rows : int  optional
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = load_amazon(raw_path, max_rows=max_rows)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved → {out_path}")
    return df


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python amazon.py <raw_path> <out_parquet> [max_rows]")
        sys.exit(1)
    max_r = int(sys.argv[3]) if len(sys.argv) > 3 else None
    prepare_amazon(sys.argv[1], sys.argv[2], max_rows=max_r)