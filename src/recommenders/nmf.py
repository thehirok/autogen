import numpy as np
import logging
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class NMFRecommender(BaseRecommender):

    def __init__(self, n_factors=15, n_epochs=50, learning_rate=0.005,
                 reg=0.02, random_state=42):
        super().__init__("NMF")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self.W = None
        self.H = None
        self.global_mean = None

    def fit(self, train_data):
        self.train_data = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        rng = np.random.RandomState(self.random_state)
        self.W = rng.uniform(0.5, 1.0, (n_users, self.n_factors))
        self.H = rng.uniform(0.5, 1.0, (self.n_factors, n_items))

        user_idxs = processed['user_idx'].to_numpy()
        item_idxs = processed['item_idx'].to_numpy()
        ratings   = processed['rating'].to_numpy()
        indices   = np.arange(len(ratings))

        for _ in range(self.n_epochs):
            rng.shuffle(indices)
            for idx in indices:
                u, i, r = user_idxs[idx], item_idxs[idx], ratings[idx]
                pred = np.dot(self.W[u], self.H[:, i])
                pred = max(pred, 1e-9)
                err = r - pred
                w_old = self.W[u].copy()
                self.W[u] += self.learning_rate * (
                    err * self.H[:, i] - self.reg * self.W[u])
                self.H[:, i] += self.learning_rate * (
                    err * w_old - self.reg * self.H[:, i])
                self.W[u] = np.maximum(self.W[u], 0)
                self.H[:, i] = np.maximum(self.H[:, i], 0)

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        pred = np.dot(self.W[u], self.H[:, i])
        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u = self.user_mapping[user_id]
        scores = self.W[u] @ self.H
        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        """Vectorized batch prediction using numpy einsum."""
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)

        valid_mask = np.array([
            uid in self.user_mapping and iid in self.item_mapping
            for uid, iid in zip(user_ids, item_ids)
        ], dtype=bool)

        if valid_mask.any():
            u_idxs = np.array([self.user_mapping[uid]
                               for uid in user_ids[valid_mask]])
            i_idxs = np.array([self.item_mapping[iid]
                               for iid in item_ids[valid_mask]])
            preds[valid_mask] = np.einsum(
                'ij,ji->i', self.W[u_idxs], self.H[:, i_idxs])

        return self._clip_array(preds)

    def get_hyperparameters(self):
        return {
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "reg": self.reg,
        }

    def set_hyperparameters(self, params):
        self.n_factors       = params.get("n_factors", self.n_factors)
        self.n_epochs        = params.get("n_epochs", self.n_epochs)
        self.learning_rate   = params.get("learning_rate", self.learning_rate)
        self.reg             = params.get("reg", self.reg)
        self.is_fitted = False