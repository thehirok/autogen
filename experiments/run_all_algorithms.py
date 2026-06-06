"""
Run all algorithms on all datasets and record performance.
Output: results/algorithm_performance.csv

This is the first script to run. It produces the performance
matrix that everything else depends on.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (DATASETS, ALGORITHMS, RESULTS_DIR,
                    RANDOM_SEED, TEST_RATIO, K_VALUES)
from src import RECOMMENDER_MAP
from src.evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def load_dataset(name, cfg, max_matrix_gb=2.0):
    path = cfg['path']
    if not os.path.exists(path):
        logger.warning(f"[{name}] File not found: {path} — skipping.")
        return None
    df = pd.read_parquet(path)
    required = {'user_id', 'item_id', 'rating'}
    if not required.issubset(df.columns):
        logger.warning(f"[{name}] Missing columns — skipping.")
        return None
    df = df[['user_id', 'item_id', 'rating']].dropna()
    if len(df) < cfg.get('min_ratings', 0):
        logger.warning(f"[{name}] Too few ratings ({len(df)}) — skipping.")
        return None
    logger.info(f"[{name}] Loaded {len(df):,} ratings.")

    # Subsample by rating count first
    if len(df) > 500_000:
        df = df.sample(n=500_000, random_state=RANDOM_SEED)
        logger.info(f"[{name}] Sampled down to {len(df):,} ratings.")

    # Check if dense matrix fits in memory
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    matrix_gb = n_users * n_items * 8 / 1e9

    if matrix_gb > max_matrix_gb:
        logger.info(f"[{name}] Matrix = {n_users}×{n_items} = "
                    f"{matrix_gb:.1f}GB (>{max_matrix_gb}GB). "
                    f"Reducing dimensions...")
        # Keep the most active users and most popular items
        max_dim = int(np.sqrt(max_matrix_gb * 1e9 / 8))  # ~15,800 for 2GB
        top_users = (df.groupby('user_id').size()
                     .nlargest(max_dim).index)
        top_items = (df.groupby('item_id').size()
                     .nlargest(max_dim).index)
        df = df[df['user_id'].isin(top_users) & df['item_id'].isin(top_items)]
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        matrix_gb = n_users * n_items * 8 / 1e9
        logger.info(f"[{name}] After filtering: {len(df):,} ratings, "
                    f"{n_users}×{n_items} = {matrix_gb:.1f}GB")

    return df


def run_one(algo_name, rec_class, train, test, evaluator):
    try:
        rec = rec_class()
        t0  = time.time()
        rec.fit(train)
        fit_time = time.time() - t0
        metrics  = evaluator.evaluate_recommender(rec, test, k_values=K_VALUES)
        return {
            'rmse':     metrics['rmse'],
            'mae':      metrics['mae'],
            'prec@10':  metrics['precision_at_k'].get(10, np.nan),
            'recall@10':metrics['recall_at_k'].get(10, np.nan),
            'ndcg@10':  metrics['ndcg_at_k'].get(10, np.nan),
            'fit_time': fit_time,
            'error':    None,
        }
    except Exception as e:
        logger.error(f"  [{algo_name}] Failed: {e}")
        return {
            'rmse': np.nan, 'mae': np.nan,
            'prec@10': np.nan, 'recall@10': np.nan,
            'ndcg@10': np.nan, 'fit_time': np.nan,
            'error': str(e),
        }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    evaluator = Evaluator(random_state=RANDOM_SEED)
    records   = []

    for ds_name, ds_cfg in DATASETS.items():
        data = load_dataset(ds_name, ds_cfg)
        if data is None:
            continue

        train, test = train_test_split(
            data, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        logger.info(f"[{ds_name}] Train={len(train):,}  Test={len(test):,}")

        for algo_name in tqdm(ALGORITHMS, desc=ds_name):
            rec_class = RECOMMENDER_MAP[algo_name]
            logger.info(f"  Running {algo_name}...")
            result = run_one(algo_name, rec_class, train, test, evaluator)
            records.append({
                'dataset':   ds_name,
                'algorithm': algo_name,
                **result,
            })

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, 'algorithm_performance.csv')
    df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    # print best algorithm per dataset (by NDCG@10, matching oracle in build_meta_dataset.py)
    print("\n--- Best algorithm per dataset (by NDCG@10) ---")
    valid = df[df['ndcg@10'].notna()]
    for ds_name, grp in valid.groupby('dataset'):
        best = grp.loc[grp['ndcg@10'].idxmax()]
        print(f"  {ds_name:<25} {best['algorithm']:<10} "
              f"NDCG@10={best['ndcg@10']:.4f}  RMSE={best['rmse']:.4f}")


if __name__ == '__main__':
    main()