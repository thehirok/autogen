from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):

    def __init__(self, name):
        self.name = name
        self.is_fitted = False
        self.train_data = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.rating_min = 1.0
        self.rating_max = 5.0

    def _preprocess_data(self, data):
        unique_users = data['user_id'].unique()
        unique_items = data['item_id'].unique()
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {it: i for i, it in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: it for it, i in self.item_mapping.items()}
        processed = data.copy()
        processed['user_idx'] = processed['user_id'].map(self.user_mapping)
        processed['item_idx'] = processed['item_id'].map(self.item_mapping)
        self.rating_min = float(data['rating'].min())
        self.rating_max = float(data['rating'].max())
        # Pre-build user -> rated items index for fast lookup in recommend()
        self._user_rated_items = (
            data.groupby('user_id')['item_id'].apply(set).to_dict()
        )
        return processed

    def _fill_matrix(self, processed_data, n_users, n_items):
        matrix = np.zeros((n_users, n_items))
        user_idxs = processed_data['user_idx'].to_numpy()
        item_idxs = processed_data['item_idx'].to_numpy()
        ratings   = processed_data['rating'].to_numpy()
        matrix[user_idxs, item_idxs] = ratings
        return matrix

    def _clip(self, value):
        return float(np.clip(value, self.rating_min, self.rating_max))

    def _clip_array(self, arr):
        return np.clip(arr, self.rating_min, self.rating_max)

    def predict_batch(self, user_ids, item_ids):
        """Predict ratings for arrays of (user_id, item_id) pairs.

        Subclasses should override this with a vectorized implementation.
        Default falls back to per-element predict().

        Parameters
        ----------
        user_ids : array-like of user IDs
        item_ids : array-like of item IDs

        Returns
        -------
        np.ndarray of predicted ratings
        """
        preds = np.empty(len(user_ids), dtype=np.float64)
        for idx in range(len(user_ids)):
            try:
                preds[idx] = self.predict(user_ids[idx], item_ids[idx])
            except Exception:
                preds[idx] = np.nan
        return preds

    def predict_all_items(self, user_id):
        """Predict ratings for one user across ALL items (vectorized).

        Subclasses should override this for maximum speed.
        Default falls back to predict_batch.

        Returns
        -------
        np.ndarray of shape (n_items,) — predicted rating per internal item index
        """
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        n_items = len(self.item_mapping)
        all_item_ids = [self.reverse_item_mapping[i] for i in range(n_items)]
        user_ids = [user_id] * n_items
        return self.predict_batch(user_ids, all_item_ids)

    def recommend(self, user_id, n=10, exclude_rated=True):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")

        scores = self.predict_all_items(user_id)
        n_items = len(self.item_mapping)

        if exclude_rated and self.train_data is not None:
            rated = self._user_rated_items.get(user_id, set())
            if rated:
                rated_idxs = [self.item_mapping[it] for it in rated if it in self.item_mapping]
                scores[rated_idxs] = -np.inf

        top_idxs = np.argpartition(scores, -min(n, len(scores)))[-n:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        items = [self.reverse_item_mapping[i] for i in top_idxs if scores[i] > -np.inf]
        ratings = [float(scores[i]) for i in top_idxs if scores[i] > -np.inf]
        return pd.DataFrame({'item_id': items[:n], 'predicted_rating': ratings[:n]})

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, user_id, item_id):
        pass

    def get_hyperparameters(self):
        return {}

    def set_hyperparameters(self, params):
        pass