import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.stability_analyzer import StabilityAnalyzer


def make_data(n=8000, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'user_id': rng.randint(0, 300, n),
        'item_id': rng.randint(0, 500, n),
        'rating':  rng.uniform(1, 5, n).round(1),
    })


def test_stability_curves_shape():
    data = make_data()
    sa   = StabilityAnalyzer(
        sample_sizes=[500, 1000], n_trials=2)
    curves = sa.compute_stability_curves(data, dataset_name='test')
    assert 'feature'        in curves.columns
    assert 'sample_size'    in curves.columns
    assert 'relative_error' in curves.columns
    assert len(curves) > 0


def test_full_features_computed():
    data = make_data()
    sa   = StabilityAnalyzer(sample_sizes=[500], n_trials=1)
    sa.compute_full_features(data)
    assert sa.full_features is not None
    assert 'sparsity' in sa.full_features


def test_stabilization_points_returns_series():
    data = make_data()
    sa   = StabilityAnalyzer(sample_sizes=[500, 1000, 2000], n_trials=2)
    sa.compute_stability_curves(data)
    stab = sa.compute_stabilization_points()
    assert isinstance(stab, pd.Series)
    assert len(stab) > 0


def test_ess_requires_importances():
    data = make_data()
    sa   = StabilityAnalyzer(sample_sizes=[500, 1000], n_trials=2)
    sa.compute_stability_curves(data)
    with pytest.raises(ValueError):
        sa.compute_ess()


def test_ess_computed():
    data = make_data()
    sa   = StabilityAnalyzer(sample_sizes=[500, 1000], n_trials=2)
    sa.compute_stability_curves(data)
    importances = {f: np.random.rand()
                   for f in sa.full_features.keys()}
    sa.set_importances(importances)
    ess = sa.compute_ess()
    assert isinstance(ess, pd.Series)
    assert (ess >= 0).all()