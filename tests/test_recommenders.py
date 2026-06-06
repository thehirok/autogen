import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.recommenders import (SVDRecommender, KNNRecommender,
                               NCFRecommender, NMFRecommender,
                               SVDppRecommender, SlopeOneRecommender)


def make_data(n=1000, n_users=50, n_items=100, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'user_id': rng.randint(0, n_users, n),
        'item_id': rng.randint(0, n_items, n),
        'rating':  rng.uniform(1, 5, n).round(1),
    })


DATA = make_data()
RECOMMENDERS = [
    SVDRecommender(n_factors=10),
    KNNRecommender(k=10),
    NCFRecommender(embedding_dim=8, layers=[16, 8], num_epochs=2),
    NMFRecommender(n_factors=5, n_epochs=5),
    SVDppRecommender(n_factors=5, n_epochs=5),
    SlopeOneRecommender(),
]


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_fit_sets_is_fitted(rec):
    rec.fit(DATA)
    assert rec.is_fitted


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_predict_returns_float(rec):
    rec.fit(DATA)
    user = DATA['user_id'].iloc[0]
    item = DATA['item_id'].iloc[0]
    pred = rec.predict(user, item)
    assert isinstance(pred, float)


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_predict_in_rating_range(rec):
    rec.fit(DATA)
    user = DATA['user_id'].iloc[0]
    item = DATA['item_id'].iloc[0]
    pred = rec.predict(user, item)
    assert 1.0 <= pred <= 5.0


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_recommend_returns_dataframe(rec):
    rec.fit(DATA)
    user = DATA['user_id'].iloc[0]
    recs = rec.recommend(user, n=5)
    assert isinstance(recs, pd.DataFrame)
    assert 'item_id' in recs.columns
    assert 'predicted_rating' in recs.columns


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_cold_start_user_returns_global_mean(rec):
    rec.fit(DATA)
    pred = rec.predict(user_id=99999, item_id=DATA['item_id'].iloc[0])
    assert isinstance(pred, float)


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_not_fitted_raises(rec):
    fresh = rec.__class__()
    with pytest.raises(ValueError):
        fresh.predict(0, 0)