import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.meta_model import MetaModel


def make_meta_data(n=20, seed=42):
    rng = np.random.RandomState(seed)
    features = pd.DataFrame({
        'sparsity':              rng.uniform(0.8, 0.99, n),
        'n_users':               rng.randint(100, 5000, n).astype(float),
        'n_items':               rng.randint(100, 5000, n).astype(float),
        'rating_mean':           rng.uniform(2, 5, n),
        'rating_std':            rng.uniform(0.5, 2, n),
        'avg_ratings_per_user':  rng.uniform(5, 100, n),
        'item_popularity_gini':  rng.uniform(0.2, 0.9, n),
    })
    labels = rng.choice(['SVD', 'KNN', 'NMF'], size=n)
    return features, labels


FEATURES, LABELS = make_meta_data()


@pytest.mark.parametrize("model_type", ['RandomForest', 'XGBoost', 'MLP'])
def test_fit_and_predict(model_type):
    m = MetaModel(model_type=model_type)
    m.fit(FEATURES, LABELS)
    pred = m.predict(FEATURES.iloc[[0]])
    assert pred in ['SVD', 'KNN', 'NMF']


@pytest.mark.parametrize("model_type", ['RandomForest', 'XGBoost', 'MLP'])
def test_predict_proba_sums_to_one(model_type):
    m = MetaModel(model_type=model_type)
    m.fit(FEATURES, LABELS)
    probas = m.predict_proba(FEATURES.iloc[[0]])
    assert abs(sum(probas.values()) - 1.0) < 1e-6


def test_confidence_between_zero_and_one():
    m = MetaModel(model_type='RandomForest')
    m.fit(FEATURES, LABELS)
    conf = m.confidence(FEATURES.iloc[[0]])
    assert 0.0 <= conf <= 1.0


def test_predict_before_fit_raises():
    m = MetaModel()
    with pytest.raises(ValueError):
        m.predict(FEATURES.iloc[[0]])


def test_feature_importances_keys_match():
    m = MetaModel(model_type='RandomForest')
    m.fit(FEATURES, LABELS)
    imps = m.feature_importances()
    assert set(imps.keys()) == set(FEATURES.columns)


def test_save_and_load(tmp_path):
    m = MetaModel(model_type='RandomForest')
    m.fit(FEATURES, LABELS)
    path = str(tmp_path / 'meta_model.pkl')
    m.save(path)
    m2 = MetaModel()
    m2.load(path)
    pred1 = m.predict(FEATURES.iloc[[0]])
    pred2 = m2.predict(FEATURES.iloc[[0]])
    assert pred1 == pred2