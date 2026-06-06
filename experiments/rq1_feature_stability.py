"""
RQ1: Are meta-features sample-stable?

Computes stability curves for every dataset and feature.
Output: results/stability_curves.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging

from config import DATASETS, RESULTS_DIR, N_SUBSAMPLE_TRIALS
from src.stability_analyzer import StabilityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_curves = []

    for ds_name, ds_cfg in DATASETS.items():
        path = ds_cfg['path']
        if not os.path.exists(path):
            logger.warning(f"[{ds_name}] Not found — skipping.")
            continue
        try:
            data = pd.read_parquet(path)
            data = data[['user_id', 'item_id', 'rating']].dropna()
            if len(data) < ds_cfg.get('min_ratings', 0):
                logger.warning(f"[{ds_name}] Too few ratings — skipping.")
                continue
            # Fresh analyzer per dataset to avoid stale internal state
            analyzer = StabilityAnalyzer(n_trials=N_SUBSAMPLE_TRIALS)
            curves = analyzer.compute_stability_curves(data, dataset_name=ds_name)
            all_curves.append(curves)
            analyzer.summary()
        except Exception as e:
            logger.error(f"[{ds_name}] Failed: {e}")

    if not all_curves:
        raise RuntimeError("No stability curves computed.")

    out_df = pd.concat(all_curves, ignore_index=True)
    out    = os.path.join(RESULTS_DIR, 'stability_curves.csv')
    out_df.to_csv(out, index=False)
    logger.info(f"Saved → {out}  ({len(out_df):,} rows)")


if __name__ == '__main__':
    main()