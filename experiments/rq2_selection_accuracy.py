"""
RQ2: Does selection accuracy hold under subsampling?

Trains the meta-model using leave-one-dataset-out CV.
At test time feeds subsampled meta-features.
Output: results/selection_accuracy_by_n.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from config import (DATASETS, RESULTS_DIR, DATA_META_DIR,
                    SAMPLE_SIZES, N_SUBSAMPLE_TRIALS,
                    RANDOM_SEED, TEST_RATIO, META_MODELS)
from src.early_selector     import EarlySelector
from src.dataset_analyzer   import DatasetAnalyzer
from src.subsampler         import Subsampler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feat_path = os.path.join(DATA_META_DIR, 'full_features.csv')
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            "Run build_meta_dataset.py first.\n"
            f"Expected: {feat_path}")

    meta_df  = pd.read_csv(feat_path)
    datasets_present = meta_df['dataset'].tolist()
    feature_cols     = [c for c in meta_df.columns
                        if c not in ('dataset', 'best_algorithm')]

    subsampler = Subsampler(
        sample_sizes=SAMPLE_SIZES,
        n_trials=N_SUBSAMPLE_TRIALS,
        random_state=RANDOM_SEED,
    )
    analyzer   = DatasetAnalyzer()
    records    = []

    for model_type in META_MODELS:
        logger.info(f"Meta-model: {model_type}")

        for test_ds in tqdm(datasets_present, desc=model_type):
            # leave-one-out: train on all other datasets
            train_df = meta_df[meta_df['dataset'] != test_ds]
            if len(train_df) < 1:
                continue

            try:
                selector = EarlySelector(
                    model_type=model_type, random_state=RANDOM_SEED)
                selector.fit_meta_model(
                    train_df[feature_cols],
                    train_df['best_algorithm'].values,
                )
            except Exception as e:
                logger.warning(f"[{test_ds}] Meta-model fit failed: {e}")
                continue
            oracle = meta_df.loc[
                meta_df['dataset'] == test_ds, 'best_algorithm'].values[0]

            # load the actual dataset for subsampling
            ds_cfg = DATASETS.get(test_ds)
            if ds_cfg is None or not os.path.exists(ds_cfg['path']):
                logger.warning(f"[{test_ds}] Raw data not found — skipping.")
                continue
            data = pd.read_parquet(ds_cfg['path'])
            data = data[['user_id', 'item_id', 'rating']].dropna()

            # Prevent data leakage: perform selection strictly on the train split
            train, _ = train_test_split(
                data, test_size=TEST_RATIO, random_state=RANDOM_SEED)

            # full dataset selection
            try:
                full_sel = selector.select(train)['algorithm']
            except Exception as e:
                logger.warning(f"[{test_ds}] Full selection failed: {e}")
                full_sel = None

            records.append({
                'model_type':  model_type,
                'dataset':     test_ds,
                'sample_size': 'full',
                'trial':       0,
                'selected':    full_sel,
                'oracle':      oracle,
                'correct':     full_sel == oracle,
            })

            # subsampled selection
            for n in SAMPLE_SIZES:
                if n >= len(train):
                    continue
                for trial in range(N_SUBSAMPLE_TRIALS):
                    try:
                        selected = selector.select_at_n(train, n, trial=trial)
                    except Exception as e:
                        logger.debug(
                            f"[{test_ds}] n={n} t={trial} failed: {e}")
                        selected = None
                    records.append({
                        'model_type':  model_type,
                        'dataset':     test_ds,
                        'sample_size': n,
                        'trial':       trial,
                        'selected':    selected,
                        'oracle':      oracle,
                        'correct':     selected == oracle,
                    })

    out_df = pd.DataFrame(records)
    out    = os.path.join(RESULTS_DIR, 'selection_accuracy_by_n.csv')
    out_df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    # quick summary
    print("\n--- Selection accuracy by sample size (RandomForest) ---")
    rf = out_df[out_df['model_type'] == 'RandomForest']
    for n, grp in rf.groupby('sample_size'):
        acc = grp['correct'].mean()
        print(f"  N={str(n):<8} accuracy={acc:.3f}")


if __name__ == '__main__':
    main()