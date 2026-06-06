"""
RQ4: What is the practical cost of early selection?

Compares final recommendation quality when algorithm is selected at:
  - N = each sample size (early)
  - N = full dataset    (full)
  - oracle              (always the true best)
  - random              (random algorithm choice)

Output: results/end_to_end_quality.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (DATASETS, RESULTS_DIR, DATA_META_DIR,
                    SAMPLE_SIZES, RANDOM_SEED, TEST_RATIO,
                    K_VALUES, ALGORITHMS)
from src import RECOMMENDER_MAP
from src.early_selector import EarlySelector
from src.evaluate       import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

def train_and_eval(algo_name, train, test, evaluator):
    try:
        rec = RECOMMENDER_MAP[algo_name]()
        rec.fit(train)
        m = evaluator.evaluate_recommender(rec, test, k_values=K_VALUES)
        return m['rmse'], m['ndcg_at_k'].get(10, np.nan)
    except Exception as e:
        logger.debug(f"[{algo_name}] eval failed: {e}")
        return np.nan, np.nan


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feat_path = os.path.join(DATA_META_DIR, 'full_features.csv')
    if not os.path.exists(feat_path):
        raise FileNotFoundError("Run build_meta_dataset.py first.")

    meta_df      = pd.read_csv(feat_path)
    feature_cols = [c for c in meta_df.columns
                    if c not in ('dataset', 'best_algorithm')]
    evaluator    = Evaluator(random_state=RANDOM_SEED)
    rng          = random.Random(RANDOM_SEED)
    records      = []

    for test_ds in tqdm(meta_df['dataset'].tolist(), desc="RQ4"):
        ds_cfg = DATASETS.get(test_ds)
        if ds_cfg is None or not os.path.exists(ds_cfg['path']):
            continue

        data = pd.read_parquet(ds_cfg['path'])
        data = data[['user_id', 'item_id', 'rating']].dropna()
        train, test = train_test_split(
            data, test_size=TEST_RATIO, random_state=RANDOM_SEED)

        oracle = meta_df.loc[
            meta_df['dataset'] == test_ds, 'best_algorithm'].values[0]

        # train meta-model leaving this dataset out
        train_meta = meta_df[meta_df['dataset'] != test_ds]
        selector   = EarlySelector(random_state=RANDOM_SEED)
        selector.fit_meta_model(
            train_meta[feature_cols],
            train_meta['best_algorithm'].values)

        # Cache: avoid training the same algorithm twice on the same split
        eval_cache = {}
        def cached_train_eval(algo_name):
            if algo_name not in eval_cache:
                eval_cache[algo_name] = train_and_eval(
                    algo_name, train, test, evaluator)
            return eval_cache[algo_name]

        # Collect all algorithm selections first
        rand_algo = rng.choice(ALGORITHMS)
        try:
            full_algo = selector.select(data)['algorithm']
        except Exception:
            full_algo = None

        early_algos = {}
        for n in SAMPLE_SIZES:
            if n >= len(data):
                continue
            try:
                early_algos[n] = selector.select_at_n(data, n, trial=0)
            except Exception:
                early_algos[n] = None

        # Now train & evaluate only the unique algorithms
        # oracle
        rmse_o, ndcg_o = cached_train_eval(oracle)
        records.append({
            'dataset': test_ds, 'condition': 'oracle',
            'sample_size': 'full', 'rmse': rmse_o, 'ndcg@10': ndcg_o})

        # random baseline
        rmse_r, ndcg_r = cached_train_eval(rand_algo)
        records.append({
            'dataset': test_ds, 'condition': 'random',
            'sample_size': 'full', 'rmse': rmse_r, 'ndcg@10': ndcg_r})

        # full-data selection
        if full_algo is not None:
            rmse_f, ndcg_f = cached_train_eval(full_algo)
        else:
            rmse_f, ndcg_f = np.nan, np.nan
        records.append({
            'dataset': test_ds, 'condition': 'full_selection',
            'sample_size': 'full', 'rmse': rmse_f, 'ndcg@10': ndcg_f})

        # early selection at each sample size
        for n in SAMPLE_SIZES:
            if n >= len(data):
                continue
            early_algo = early_algos[n]
            if early_algo is not None:
                rmse_e, ndcg_e = cached_train_eval(early_algo)
            else:
                rmse_e, ndcg_e = np.nan, np.nan
            records.append({
                'dataset': test_ds, 'condition': 'early_selection',
                'sample_size': n, 'rmse': rmse_e, 'ndcg@10': ndcg_e})

        logger.info(f"[{test_ds}] Cache hits: {len(eval_cache)} unique algos "
                    f"out of {3 + len(early_algos)} selections")

    out_df = pd.DataFrame(records)
    out    = os.path.join(RESULTS_DIR, 'end_to_end_quality.csv')
    out_df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    print("\n--- Mean NDCG@10 by condition ---")
    summary = out_df.groupby('condition')['ndcg@10'].mean()
    print(summary.to_string())


if __name__ == '__main__':
    main()