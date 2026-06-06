"""
Statistical significance tests across datasets.
Requires: results/end_to_end_quality.csv
Output:   results/significance_tests.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from scipy import stats
from itertools import combinations

from config import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test with degenerate-case handling."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    # Remove NaN pairs
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 5:
        return np.nan, np.nan, len(a)
    diff = a - b
    if np.all(diff == 0):
        # All identical — no difference, perfectly non-significant
        return 0.0, 1.0, len(a)
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p), len(a)
    except Exception:
        return np.nan, np.nan, len(a)


def paired_ttest(a, b):
    """Paired t-test with degenerate-case handling."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return np.nan, np.nan, len(a)
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0, len(a)
    try:
        t_stat, p = stats.ttest_rel(a, b)
        return float(t_stat), float(p), len(a)
    except Exception:
        return np.nan, np.nan, len(a)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    e2e_path = os.path.join(RESULTS_DIR, 'end_to_end_quality.csv')
    if not os.path.exists(e2e_path):
        raise FileNotFoundError("Run rq4_end_to_end_quality.py first.")

    df = pd.read_csv(e2e_path)

    # focus on full-size conditions for cross-condition comparisons
    full = df[df['sample_size'] == 'full']
    conditions = full['condition'].unique().tolist()
    datasets   = full['dataset'].unique().tolist()

    records = []

    # pairwise Wilcoxon signed-rank tests on RMSE across datasets
    for c1, c2 in combinations(conditions, 2):
        rmse1, rmse2 = [], []
        for ds in datasets:
            r1 = full[(full['condition'] == c1) &
                      (full['dataset']   == ds)]['rmse'].values
            r2 = full[(full['condition'] == c2) &
                      (full['dataset']   == ds)]['rmse'].values
            if len(r1) > 0 and len(r2) > 0:
                rmse1.append(float(r1[0]))
                rmse2.append(float(r2[0]))

        stat, p, n = wilcoxon_test(rmse1, rmse2)
        records.append({
            'condition_a':  c1,
            'condition_b':  c2,
            'metric':       'rmse',
            'test_type':    'wilcoxon',
            'n_datasets':   n,
            'mean_a':       np.nanmean(rmse1) if rmse1 else np.nan,
            'mean_b':       np.nanmean(rmse2) if rmse2 else np.nan,
            'test_stat':    stat,
            'p_value':      p,
            'significant':  p < 0.05 if not np.isnan(p) else False,
        })

    # early vs full selection at each sample size (paired t-test)
    early = df[df['condition'] == 'early_selection']
    full_sel = df[df['condition'] == 'full_selection']

    for n in sorted([s for s in df['sample_size'].unique()
                      if s not in ('full', 'full_selection') and not isinstance(s, str)]):
        e_rmse, f_rmse = [], []
        for ds in datasets:
            e = early[(early['sample_size'] == n) &
                      (early['dataset'] == ds)]['rmse'].values
            f = full_sel[full_sel['dataset'] == ds]['rmse'].values
            if len(e) > 0 and len(f) > 0:
                e_rmse.append(float(e[0]))
                f_rmse.append(float(f[0]))

        t_stat, p, n_valid = paired_ttest(e_rmse, f_rmse)
        records.append({
            'condition_a':   f'early_N{n}',
            'condition_b':   'full_selection',
            'metric':        'rmse',
            'test_type':     'paired_ttest',
            'n_datasets':    n_valid,
            'mean_a':        np.nanmean(e_rmse) if e_rmse else np.nan,
            'mean_b':        np.nanmean(f_rmse) if f_rmse else np.nan,
            'test_stat':     t_stat,
            'p_value':       p,
            'significant':   p < 0.05 if not np.isnan(p) else False,
        })

    out_df = pd.DataFrame(records)

    # Holm–Bonferroni correction for multiple comparisons
    p_values = out_df['p_value'].values.copy()
    n_tests = len(p_values)
    valid_p = ~np.isnan(p_values)

    if valid_p.any():
        # Sort p-values, apply Holm step-down correction
        sorted_idx = np.argsort(p_values[valid_p])
        original_idx = np.where(valid_p)[0][sorted_idx]
        corrected = np.full(n_tests, np.nan)
        for rank, orig_i in enumerate(original_idx):
            corrected[orig_i] = min(
                p_values[orig_i] * (valid_p.sum() - rank), 1.0)
        # Enforce monotonicity (corrected p can't decrease as raw p increases)
        running_max = 0.0
        for orig_i in original_idx:
            corrected[orig_i] = max(corrected[orig_i], running_max)
            running_max = corrected[orig_i]

        out_df['p_value_corrected'] = corrected
    else:
        out_df['p_value_corrected'] = np.nan

    out_df['significant_corrected'] = out_df['p_value_corrected'].apply(
        lambda p: p < 0.05 if not np.isnan(p) else False)

    out    = os.path.join(RESULTS_DIR, 'significance_tests.csv')
    out_df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    print("\n--- Significance tests (with Holm–Bonferroni correction) ---")
    print(out_df[['condition_a', 'condition_b',
                  'mean_a', 'mean_b',
                  'test_stat', 'p_value', 'significant',
                  'p_value_corrected', 'significant_corrected']].to_string(index=False))


if __name__ == '__main__':
    main()