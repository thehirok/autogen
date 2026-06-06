"""
Ablation: compare RandomForest vs XGBoost vs MLP as meta-model.
Output: results/ablation_meta_models.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import LeaveOneOut

from config import DATA_META_DIR, RESULTS_DIR, RANDOM_SEED, META_MODELS
from src.meta_model import MetaModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feat_path = os.path.join(DATA_META_DIR, 'full_features.csv')
    if not os.path.exists(feat_path):
        raise FileNotFoundError("Run build_meta_dataset.py first.")

    meta_df      = pd.read_csv(feat_path)
    feature_cols = [c for c in meta_df.columns
                    if c not in ('dataset', 'best_algorithm')]
    X = meta_df[feature_cols].values
    y = meta_df['best_algorithm'].values
    datasets = meta_df['dataset'].values

    records = []
    loo     = LeaveOneOut()

    for model_type in META_MODELS:
        logger.info(f"Evaluating {model_type}...")
        preds = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train         = y[train_idx]
            ds_test         = datasets[test_idx[0]]

            X_train_df = pd.DataFrame(X_train, columns=feature_cols)
            X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)

            model = MetaModel(
                model_type=model_type, random_state=RANDOM_SEED)
            try:
                model.fit(X_train_df, y_train)
                pred       = model.predict(X_test_df)
                confidence = model.confidence(X_test_df)
            except Exception as e:
                logger.warning(f"  [{model_type}] fold failed: {e}")
                pred, confidence = None, None

            preds.append({
                'model_type':  model_type,
                'dataset':     ds_test,
                'oracle':      y[test_idx[0]],
                'predicted':   pred,
                'confidence':  confidence,
                'correct':     pred == y[test_idx[0]],
            })

        acc = np.mean([p['correct'] for p in preds if p['predicted']])
        logger.info(f"  {model_type} LOO accuracy: {acc:.3f}")
        records.extend(preds)

    out_df = pd.DataFrame(records)
    out    = os.path.join(RESULTS_DIR, 'ablation_meta_models.csv')
    out_df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")

    print("\n--- Meta-model ablation (LOO accuracy) ---")
    print(out_df.groupby('model_type')['correct'].mean().to_string())


if __name__ == '__main__':
    main()