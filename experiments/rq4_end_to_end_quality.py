"""
RQ4: What is the practical cost of early selection?

Compares final recommendation quality when algorithm is selected at:
  - N = each sample size (early)
  - N = full dataset    (full)
  - oracle              (always the true best)
  - random              (random algorithm choice)

OPTIMIZED VERSION v2: Caches feature extraction per dataset so it is done
ONCE, not once-per-seed.  The multi-seed variation only affects the
meta-model's fitted weights, not the feature vectors.

Output: results/end_to_end_quality.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from config import (DATASETS, RESULTS_DIR, DATA_META_DIR,
                    SAMPLE_SIZES, RANDOM_SEED, ALGORITHMS,
                    EVAL_SEEDS, N_SUBSAMPLE_TRIALS)
from src.early_selector import EarlySelector
from src.dataset_analyzer import DatasetAnalyzer
from src.subsampler import Subsampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load pre-computed data ---
    feat_path = os.path.join(DATA_META_DIR, 'full_features.csv')
    perf_path = os.path.join(RESULTS_DIR, 'algorithm_performance.csv')

    if not os.path.exists(feat_path):
        raise FileNotFoundError("Run build_meta_dataset.py first.")
    if not os.path.exists(perf_path):
        raise FileNotFoundError("Run run_all_algorithms.py first.")

    meta_df      = pd.read_csv(feat_path)
    perf_df      = pd.read_csv(perf_path)
    feature_cols = [c for c in meta_df.columns
                    if c not in ('dataset', 'best_algorithm')]

    # Build a lookup: perf_cache[dataset][algorithm] = (rmse, ndcg@10)
    perf_cache = {}
    for _, row in perf_df.iterrows():
        ds   = row['dataset']
        algo = row['algorithm']
        if ds not in perf_cache:
            perf_cache[ds] = {}
        perf_cache[ds][algo] = (row.get('rmse', np.nan),
                                 row.get('ndcg@10', np.nan))

    # --- Check for partial results to resume from ---
    out_path = os.path.join(RESULTS_DIR, 'end_to_end_quality.csv')
    completed_datasets = set()
    prior_records = []
    if os.path.exists(out_path):
        prior_df = pd.read_csv(out_path)
        completed_datasets = set(prior_df['dataset'].unique())
        prior_records = prior_df.to_dict('records')
        logger.info(f"Resuming — {len(completed_datasets)} datasets already "
                    f"done: {sorted(completed_datasets)}")

    analyzer   = DatasetAnalyzer()
    subsampler = Subsampler(random_state=RANDOM_SEED)
    records    = list(prior_records)

    for test_ds in tqdm(meta_df['dataset'].tolist(), desc="RQ4"):
        # Skip already-completed datasets
        if test_ds in completed_datasets:
            logger.info(f"[{test_ds}] Already complete, skipping.")
            continue

        ds_cfg = DATASETS.get(test_ds)
        if ds_cfg is None or not os.path.exists(ds_cfg['path']):
            logger.warning(f"[{test_ds}] Data file not found, skipping.")
            continue

        if test_ds not in perf_cache:
            logger.warning(f"[{test_ds}] No performance data, skipping.")
            continue

        data = pd.read_parquet(ds_cfg['path'])
        data = data[['user_id', 'item_id', 'rating']].dropna()

        oracle = meta_df.loc[
            meta_df['dataset'] == test_ds, 'best_algorithm'].values[0]

        # Pre-computed evaluation results for this dataset
        eval_cache = perf_cache[test_ds]

        # Expected value of random baseline
        valid_rmse = [eval_cache[a][0] for a in ALGORITHMS
                      if a in eval_cache and not np.isnan(eval_cache[a][0])]
        valid_ndcg = [eval_cache[a][1] for a in ALGORITHMS
                      if a in eval_cache and not np.isnan(eval_cache[a][1])]
        rmse_r = np.mean(valid_rmse) if valid_rmse else np.nan
        ndcg_r = np.mean(valid_ndcg) if valid_ndcg else np.nan

        # ===== CACHE FEATURE EXTRACTION (once per dataset) =====

        # 1. Full-data features (extracted ONCE, reused across all seeds)
        logger.info(f"[{test_ds}] Extracting full-data features...")
        analyzer.load_dataset(data)
        full_features = analyzer.extract_features()
        full_features_df = pd.DataFrame([full_features])
        logger.info(f"[{test_ds}] Full-data features cached.")

        # 2. Subsample features (extracted ONCE per (n, trial))
        subsample_features_cache = {}  # (n, trial) -> features_df
        for n in SAMPLE_SIZES:
            if n >= len(data):
                continue
            for trial in range(N_SUBSAMPLE_TRIALS):
                try:
                    sample = subsampler.subsample(data, n, trial=trial)
                    analyzer.load_dataset(sample)
                    feats = analyzer.extract_features()
                    subsample_features_cache[(n, trial)] = pd.DataFrame([feats])
                except Exception as e:
                    logger.warning(f"[{test_ds}] Feature extraction failed "
                                   f"n={n} trial={trial}: {e}")
                    subsample_features_cache[(n, trial)] = None

        logger.info(f"[{test_ds}] Cached {len(subsample_features_cache)} "
                    f"subsample feature vectors.")

        # ===== LOOP OVER SEEDS (only meta-model changes) =====

        train_meta = meta_df[meta_df['dataset'] != test_ds]

        for seed in EVAL_SEEDS:
            # Fit the early selector meta-model for this seed
            selector = EarlySelector(random_state=seed)
            selector.fit_meta_model(
                train_meta[feature_cols],
                train_meta['best_algorithm'].values)

            # Oracle
            rmse_o, ndcg_o = eval_cache.get(oracle, (np.nan, np.nan))
            records.append({
                'dataset': test_ds, 'condition': 'oracle',
                'sample_size': 'full', 'seed': seed, 'trial': 0,
                'rmse': rmse_o, 'ndcg@10': ndcg_o})

            # Random baseline
            records.append({
                'dataset': test_ds, 'condition': 'random',
                'sample_size': 'full', 'seed': seed, 'trial': 0,
                'rmse': rmse_r, 'ndcg@10': ndcg_r})

            # Full-data selection (use cached features, only predict)
            try:
                full_algo = selector.meta_model.predict(full_features_df)
                rmse_f, ndcg_f = eval_cache.get(full_algo, (np.nan, np.nan))
            except Exception:
                rmse_f, ndcg_f = np.nan, np.nan
            records.append({
                'dataset': test_ds, 'condition': 'full_selection',
                'sample_size': 'full', 'seed': seed, 'trial': 0,
                'rmse': rmse_f, 'ndcg@10': ndcg_f})

            # Early selection at each sample size (use cached features)
            for n in SAMPLE_SIZES:
                if n >= len(data):
                    continue
                for trial in range(N_SUBSAMPLE_TRIALS):
                    cached = subsample_features_cache.get((n, trial))
                    if cached is None:
                        rmse_e, ndcg_e = np.nan, np.nan
                    else:
                        try:
                            early_algo = selector.meta_model.predict(cached)
                            rmse_e, ndcg_e = eval_cache.get(
                                early_algo, (np.nan, np.nan))
                        except Exception:
                            rmse_e, ndcg_e = np.nan, np.nan
                    records.append({
                        'dataset': test_ds, 'condition': 'early_selection',
                        'sample_size': n, 'seed': seed, 'trial': trial,
                        'rmse': rmse_e, 'ndcg@10': ndcg_e})

            logger.info(f"[{test_ds}] Seed {seed} done")

        # --- Save after each dataset (incremental) ---
        out_df = pd.DataFrame(records)
        out_df.to_csv(out_path, index=False)
        logger.info(f"[{test_ds}] Complete — saved {len(out_df)} rows.")

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved -> {out_path}")

    print("\n--- Mean NDCG@10 by condition ---")
    summary = out_df.groupby('condition')['ndcg@10'].mean()
    print(summary.to_string())


if __name__ == '__main__':
    main()