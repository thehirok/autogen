import numpy as np
import logging
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class SVDppRecommender(BaseRecommender):

    def __init__(self, n_factors=20, n_epochs=20, learning_rate=0.007,
                 reg=0.02, random_state=42):
        super().__init__("SVDpp")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self.global_mean = None
        self.bu = None
        self.bi = None
        self.P  = None
        self.Q  = None
        self.Y  = None
        self.user_implicit = None

    def fit(self, train_data):
        self.train_data = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        rng = np.random.RandomState(self.random_state)
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.P  = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q  = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.Y  = rng.normal(0, 0.1, (n_items, self.n_factors))

        # build implicit feedback sets per user (as numpy arrays)
        self.user_implicit = [[] for _ in range(n_users)]
        for u, i in zip(processed['user_idx'], processed['item_idx']):
            self.user_implicit[u].append(i)

        # Pre-compute as arrays + norms for speed
        impl_arrays = [np.array(items, dtype=np.int64) for items in self.user_implicit]
        impl_norms = np.array([1.0 / np.sqrt(max(len(items), 1))
                               for items in self.user_implicit])

        user_idxs = processed['user_idx'].to_numpy()
        item_idxs = processed['item_idx'].to_numpy()
        ratings   = processed['rating'].to_numpy()
        lr = self.learning_rate
        reg = self.reg
        indices = np.arange(len(ratings))

        for epoch in range(self.n_epochs):
            rng.shuffle(indices)
            for idx in indices:
                u, i, r = user_idxs[idx], item_idxs[idx], ratings[idx]
                impl = impl_arrays[u]
                norm = impl_norms[u]
                ysum = np.sum(self.Y[impl], axis=0) * norm
                pred = (self.global_mean + self.bu[u] + self.bi[i]
                        + np.dot(self.Q[i], self.P[u] + ysum))
                err = r - pred
                # Clip error to prevent gradient explosion on wide-range datasets
                err = np.clip(err, -10.0, 10.0)
                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                puf = self.P[u].copy()
                self.P[u] += lr * (err * self.Q[i] - reg * self.P[u])
                self.Q[i] += lr * (err * (puf + ysum) - reg * self.Q[i])
                # Clip factors to prevent overflow accumulation
                np.clip(self.P[u], -10, 10, out=self.P[u])
                np.clip(self.Q[i], -10, 10, out=self.Q[i])
                # Vectorized Y update for all implicit items at once
                y_update = lr * (err * norm * self.Q[i][np.newaxis, :]
                                      - reg * self.Y[impl])
                self.Y[impl] = np.clip(self.Y[impl] + y_update, -10, 10)

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        impl = self.user_implicit[u]
        norm = 1.0 / np.sqrt(max(len(impl), 1))
        ysum = np.sum(self.Y[impl], axis=0) * norm
        pred = (self.global_mean + self.bu[u] + self.bi[i]
                + np.dot(self.Q[i], self.P[u] + ysum))
        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u = self.user_mapping[user_id]
        impl = self.user_implicit[u]
        norm = 1.0 / np.sqrt(max(len(impl), 1))
        ysum = np.sum(self.Y[impl], axis=0) * norm
        # Q @ (P[u] + ysum) for all items at once
        scores = (self.global_mean
                  + self.bu[u]
                  + self.bi
                  + self.Q @ (self.P[u] + ysum))
        return self._clip_array(scores)


    def get_hyperparameters(self):
        return {
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "reg": self.reg,
        }

    def set_hyperparameters(self, params):
        self.n_factors     = params.get("n_factors", self.n_factors)
        self.n_epochs      = params.get("n_epochs", self.n_epochs)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.reg           = params.get("reg", self.reg)
        self.is_fitted = False