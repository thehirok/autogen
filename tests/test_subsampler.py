import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.subsampler import Subsampler


def make_data(n=5000, n_users=200, n_items=500, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'user_id': rng.randint(0, n_users, n),
        'item_id': rng.randint(0, n_items, n),
        'rating':  rng.uniform(1, 5, n).round(1),
    })


def test_subsample_size():
    data = make_data()
    s    = Subsampler(sample_sizes=[500, 1000])
    sub  = s.subsample(data, 500)
    assert len(sub) == 500


def test_subsample_columns():
    data = make_data()
    s    = Subsampler()
    sub  = s.subsample(data, 500)
    assert set(sub.columns) >= {'user_id', 'item_id', 'rating'}


def test_subsample_full_passthrough():
    data = make_data(n=300)
    s    = Subsampler(sample_sizes=[500])
    sub  = s.subsample(data, 500)
    assert len(sub) == len(data)


def test_multi_trial_variance():
    data   = make_data()
    s      = Subsampler(n_trials=5)
    trials = s.subsample_multi_trial(data, 500)
    assert len(trials) == 5
    sizes  = [len(t) for t in trials]
    assert all(sz == 500 for sz in sizes)


def test_validate_passes():
    data = make_data()
    s    = Subsampler(sample_sizes=[100])
    assert s.validate(data)


def test_validate_missing_column():
    data = make_data().drop(columns=['rating'])
    s    = Subsampler(sample_sizes=[100])
    with pytest.raises(ValueError):
        s.validate(data)


def test_stratification_covers_users():
    data  = make_data(n=5000, n_users=50)
    s     = Subsampler()
    sub   = s.subsample(data, 1000)
    users_full = set(data['user_id'].unique())
    users_sub  = set(sub['user_id'].unique())
    # at least 80% of users should appear in the subsample
    assert len(users_sub) / len(users_full) >= 0.8