"""
Quick integration test: run all algorithms on 2 small datasets.
Uses movielens_100k and filmtrust only.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import time
from sklearn.model_selection import train_test_split

from config import RESULTS_DIR, RANDOM_SEED, TEST_RATIO, K_VALUES
from src import RECOMMENDER_MAP
from src.evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

QUICK_DATASETS = {
    'movielens_100k': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'movielens_100k.parquet'),
    'filmtrust': os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'filmtrust.parquet'),
}

ALGORITHMS = ["SVD", "KNN", "NMF", "NCF", "SlopeOne", "SVDpp"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    evaluator = Evaluator(random_state=RANDOM_SEED)
    records = []

    for ds_name, path in QUICK_DATASETS.items():
        if not os.path.exists(path):
            logger.warning(f"[{ds_name}] Not found — skipping.")
            continue

        data = pd.read_parquet(path)
        data = data[['user_id', 'item_id', 'rating']].dropna()
        logger.info(f"[{ds_name}] Loaded {len(data):,} ratings.")

        train, test = train_test_split(
            data, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        logger.info(f"[{ds_name}] Train={len(train):,}  Test={len(test):,}")

        for algo_name in ALGORITHMS:
            rec_class = RECOMMENDER_MAP[algo_name]
            logger.info(f"  Running {algo_name}...")
            try:
                rec = rec_class()
                t0 = time.time()
                rec.fit(train)
                fit_time = time.time() - t0

                t1 = time.time()
                metrics = evaluator.evaluate_recommender(
                    rec, test, k_values=K_VALUES)
                eval_time = time.time() - t1

                result = {
                    'dataset':    ds_name,
                    'algorithm':  algo_name,
                    'rmse':       metrics['rmse'],
                    'mae':        metrics['mae'],
                    'prec@10':    metrics['precision_at_k'].get(10, np.nan),
                    'recall@10':  metrics['recall_at_k'].get(10, np.nan),
                    'ndcg@10':    metrics['ndcg_at_k'].get(10, np.nan),
                    'fit_time':   round(fit_time, 2),
                    'eval_time':  round(eval_time, 2),
                    'error':      None,
                }
                logger.info(f"    RMSE={result['rmse']:.4f}  "
                            f"fit={fit_time:.1f}s  eval={eval_time:.1f}s")
            except Exception as e:
                logger.error(f"  [{algo_name}] Failed: {e}")
                result = {
                    'dataset': ds_name, 'algorithm': algo_name,
                    'rmse': np.nan, 'mae': np.nan,
                    'prec@10': np.nan, 'recall@10': np.nan,
                    'ndcg@10': np.nan, 'fit_time': np.nan,
                    'eval_time': np.nan, 'error': str(e),
                }
            records.append(result)

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, 'algorithm_performance.csv')
    df.to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("QUICK TEST RESULTS")
    print("=" * 70)
    for ds_name, grp in df.groupby('dataset'):
        print(f"\n--- {ds_name} ---")
        for _, row in grp.iterrows():
            status = f"RMSE={row['rmse']:.4f}" if not np.isnan(row['rmse']) else f"ERROR: {row['error']}"
            print(f"  {row['algorithm']:<10} {status}  "
                  f"fit={row['fit_time']}s  eval={row['eval_time']}s")

    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
