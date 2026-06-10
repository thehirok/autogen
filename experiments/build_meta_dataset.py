"""
Build the meta-learning dataset from algorithm performance results.
Requires: results/algorithm_performance.csv (from run_all_algorithms.py)
Output:   data/meta/full_features.csv
          data/meta/algorithm_performance.csv  (clean copy)

Defines the oracle label for each dataset as the algorithm
with lowest RMSE on the full dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging

from config import DATASETS, RESULTS_DIR, DATA_META_DIR, TEST_RATIO, RANDOM_SEED
from src.dataset_analyzer import DatasetAnalyzer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(DATA_META_DIR, exist_ok=True)

    # --- load performance matrix ---
    perf_path = os.path.join(RESULTS_DIR, 'algorithm_performance.csv')
    if not os.path.exists(perf_path):
        raise FileNotFoundError(
            f"Run experiments/run_all_algorithms.py first.\n"
            f"Expected: {perf_path}")

    perf = pd.read_csv(perf_path)

    # oracle = best algorithm per dataset (lowest RMSE, ignoring errors)
    perf_clean = perf[perf['error'].isna()].copy()
    oracle = (
        perf_clean
        .sort_values('ndcg@10', ascending=False)
        .groupby('dataset')
        .first()['algorithm']
        .rename('best_algorithm')
    )
    logger.info(f"Oracle labels:\n{oracle.to_string()}")

    # --- extract meta-features for each dataset ---
    analyzer = DatasetAnalyzer()
    feature_records = []

    for ds_name, ds_cfg in DATASETS.items():
        path = ds_cfg['path']
        if not os.path.exists(path):
            logger.warning(f"[{ds_name}] Not found — skipping.")
            continue
        if ds_name not in oracle.index:
            logger.warning(f"[{ds_name}] No oracle label — skipping.")
            continue

        try:
            data = pd.read_parquet(path)
            data = data[['user_id', 'item_id', 'rating']].dropna()
            # Prevent data leakage: extract meta-features only from the train split
            train, _ = train_test_split(
                data, test_size=TEST_RATIO, random_state=RANDOM_SEED)
            analyzer.load_dataset(train)
            feats = analyzer.extract_features()
            feats['dataset']        = ds_name
            feats['best_algorithm'] = oracle[ds_name]
            feature_records.append(feats)
            logger.info(f"[{ds_name}] Features extracted from train split. "
                        f"Oracle={oracle[ds_name]}")
        except Exception as e:
            logger.error(f"[{ds_name}] Failed: {e}")

    if not feature_records:
        raise RuntimeError("No features extracted. "
                           "Check your data/processed/ folder.")

    features_df = pd.DataFrame(feature_records)

    # save
    feat_out = os.path.join(DATA_META_DIR, 'full_features.csv')
    perf_out = os.path.join(DATA_META_DIR, 'algorithm_performance.csv')
    features_df.to_csv(feat_out, index=False)
    perf_clean.to_csv(perf_out, index=False)

    logger.info(f"Saved meta-features  → {feat_out}")
    logger.info(f"Saved performance    → {perf_out}")
    print(f"\nMeta-dataset: {len(features_df)} datasets × "
          f"{len(features_df.columns)-2} features")
    print(f"Algorithm distribution:\n"
          f"{features_df['best_algorithm'].value_counts().to_string()}")


if __name__ == '__main__':
    main()