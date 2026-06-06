"""
RQ3: Which meta-features are most sample-efficient?

Computes SHAP values from the fitted meta-model and combines
with stabilization points to produce the ESS ranking.
Output: results/shap_values.csv
        results/ess_scores.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import shap
import logging

from config import DATA_META_DIR, RESULTS_DIR, RANDOM_SEED
from src.meta_model         import MetaModel
from src.stability_analyzer import StabilityAnalyzer

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
            "Run build_meta_dataset.py first.")

    meta_df = pd.read_csv(feat_path)
    feature_cols = [c for c in meta_df.columns
                    if c not in ('dataset', 'best_algorithm')]
    X = meta_df[feature_cols]
    y = meta_df['best_algorithm']

    # fit RandomForest meta-model on all data for SHAP
    model = MetaModel(model_type='RandomForest', random_state=RANDOM_SEED)
    model.fit(X, y)

    # SHAP values
    X_scaled = model.scaler.transform(X)
    explainer = shap.TreeExplainer(model.model)
    shap_vals = explainer.shap_values(X_scaled)

    # mean absolute SHAP per feature (averaged over classes if needed)
    if isinstance(shap_vals, list):
        mean_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_shap = mean_abs.mean(axis=tuple(range(1, mean_abs.ndim)))
        else:
            mean_shap = mean_abs

    shap_df = pd.DataFrame({
        'feature':         feature_cols,
        'mean_abs_shap':   mean_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    shap_out = os.path.join(RESULTS_DIR, 'shap_values.csv')
    shap_df.to_csv(shap_out, index=False)
    logger.info(f"Saved SHAP → {shap_out}")

    # ESS — combine SHAP with stabilization points
    stab_path = os.path.join(RESULTS_DIR, 'stability_curves.csv')
    if not os.path.exists(stab_path):
        logger.warning("stability_curves.csv not found — "
                       "skipping ESS computation. "
                       "Run rq1_feature_stability.py first.")
        return

    sa = StabilityAnalyzer()
    sa.stability_curves = pd.read_csv(stab_path)
    importances = dict(zip(shap_df['feature'], shap_df['mean_abs_shap']))
    sa.set_importances(importances)
    ess = sa.compute_ess()

    ess_df = ess.reset_index()
    ess_df.columns = ['feature', 'ess']
    stab   = sa.compute_stabilization_points()
    ess_df['stabilization_n'] = ess_df['feature'].map(stab)
    ess_df['shap_importance'] = ess_df['feature'].map(importances)
    ess_df = ess_df.sort_values('ess', ascending=False)

    ess_out = os.path.join(RESULTS_DIR, 'ess_scores.csv')
    ess_df.to_csv(ess_out, index=False)
    logger.info(f"Saved ESS → {ess_out}")

    print("\n--- Top 10 features by ESS ---")
    print(ess_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()