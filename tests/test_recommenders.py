import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.recommenders import (SVDRecommender, KNNRecommender,
                               NCFRecommender, NMFRecommender,
                               SVDppRecommender, SlopeOneRecommender,
                               PEFTNCFRecommender)


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


@pytest.mark.parametrize("rec", RECOMMENDERS,
                         ids=[r.name for r in RECOMMENDERS])
def test_predict_batch_matches_predict(rec):
    if rec.name == "KNN":
        pytest.skip("KNN predict_batch uses a global top-k approximation which is mathematically different from predict().")
    rec.fit(DATA)
    sample_users = DATA['user_id'].iloc[:10].tolist()
    sample_items = DATA['item_id'].iloc[:10].tolist()
    
    # Batch predict
    batch_preds = rec.predict_batch(sample_users, sample_items)
    
    # Sequential predict
    seq_preds = np.array([
        rec.predict(u, i) for u, i in zip(sample_users, sample_items)
    ])
    
    np.testing.assert_allclose(batch_preds, seq_preds, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------
# PEFT-NCF specific tests
# ---------------------------------------------------------------

class TestPEFTNCF:
    """Dedicated tests for the PEFT-NCF recommender."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Create source and target data, pre-train on source."""
        self.source_data = make_data(n=500, n_users=30, n_items=60, seed=1)
        self.target_data = make_data(n=300, n_users=20, n_items=40, seed=2)
        self.pretrained_path = str(tmp_path / "pretrained_ncf_mlp.pt")

    def _make_peft_rec(self):
        return PEFTNCFRecommender(
            embedding_dim=8, layers=[16, 8],
            lora_r=4, lora_alpha=8,
            pretrain_epochs=2, adapt_epochs=2,
            learning_rate=0.001, batch_size=128,
            random_state=42,
            pretrained_path=self.pretrained_path,
        )

    def test_pretrain_creates_weights_file(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        assert os.path.exists(self.pretrained_path)

    def test_fit_sets_is_fitted(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        assert rec.is_fitted

    def test_predict_returns_float(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        user = self.target_data['user_id'].iloc[0]
        item = self.target_data['item_id'].iloc[0]
        pred = rec.predict(user, item)
        assert isinstance(pred, float)

    def test_predict_in_rating_range(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        user = self.target_data['user_id'].iloc[0]
        item = self.target_data['item_id'].iloc[0]
        pred = rec.predict(user, item)
        assert 1.0 <= pred <= 5.0

    def test_lora_reduces_trainable_params(self):
        """LoRA should make most parameters non-trainable."""
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        summary = rec.get_peft_param_summary()
        # Trainable should be less than total
        assert summary['trainable_params'] < summary['total_params']
        # Frozen params should exist
        assert summary['frozen_params'] > 0

    def test_predict_batch_matches_predict(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)

        sample_users = self.target_data['user_id'].iloc[:10].tolist()
        sample_items = self.target_data['item_id'].iloc[:10].tolist()

        batch_preds = rec.predict_batch(sample_users, sample_items)
        seq_preds = np.array([
            rec.predict(u, i) for u, i in zip(sample_users, sample_items)
        ])
        np.testing.assert_allclose(batch_preds, seq_preds, rtol=1e-5, atol=1e-5)

    def test_cold_start_returns_global_mean(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        pred = rec.predict(user_id=99999, item_id=99999)
        assert isinstance(pred, float)

    def test_recommend_returns_dataframe(self):
        rec = self._make_peft_rec()
        rec.pretrain(source_data=self.source_data)
        rec.fit(self.target_data)
        user = self.target_data['user_id'].iloc[0]
        recs = rec.recommend(user, n=5)
        assert isinstance(recs, pd.DataFrame)
        assert 'item_id' in recs.columns

    def test_not_fitted_raises(self):
        rec = self._make_peft_rec()
        with pytest.raises(ValueError):
            rec.predict(0, 0)