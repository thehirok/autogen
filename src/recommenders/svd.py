import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import logging
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):

    def __init__(self, n_factors=100, random_state=42):
        super().__init__("SVD")
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_matrix = None
        self.item_matrix = None
        self.global_mean = None
        self.user_biases = None
        self.item_biases = None

    def fit(self, train_data):
        self.train_data = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        matrix = self._fill_matrix(processed, n_users, n_items)

        # Boolean mask: True where a rating exists (handles 0-valued ratings)
        mask = np.zeros((n_users, n_items), dtype=bool)
        user_idxs = processed['user_idx'].to_numpy()
        item_idxs = processed['item_idx'].to_numpy()
        mask[user_idxs, item_idxs] = True

        # user biases — only over rated entries (vectorized)
        user_sums   = (matrix * mask).sum(axis=1)
        user_counts = mask.sum(axis=1)
        user_counts = np.where(user_counts == 0, 1, user_counts)
        self.user_biases = user_sums / user_counts - self.global_mean

        # item biases — only over rated entries (vectorized)
        item_sums   = (matrix * mask).sum(axis=0)
        item_counts = mask.sum(axis=0)
        item_counts = np.where(item_counts == 0, 1, item_counts)
        self.item_biases = item_sums / item_counts - self.global_mean

        # subtract biases from observed entries only (vectorized)
        bias_matrix = (self.global_mean
                       + self.user_biases[:, np.newaxis]
                       + self.item_biases[np.newaxis, :])
        matrix[mask] -= bias_matrix[mask]
        # unrated entries stay 0 (= mean residual) to limit SVD distortion

        n_factors = min(self.n_factors, n_users - 1, n_items - 1)
        model = TruncatedSVD(n_components=n_factors,
                             random_state=self.random_state)
        model.fit(matrix)
        self.user_matrix = model.transform(matrix)
        self.item_matrix = model.components_
        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        pred = (self.global_mean
                + self.user_biases[u]
                + self.item_biases[i]
                + np.dot(self.user_matrix[u], self.item_matrix[:, i]))
        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u = self.user_mapping[user_id]
        scores = (self.global_mean
                  + self.user_biases[u]
                  + self.item_biases
                  + self.user_matrix[u] @ self.item_matrix)
        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)

        # Build index arrays for valid (known user, known item) pairs
        valid_mask = np.array([
            uid in self.user_mapping and iid in self.item_mapping
            for uid, iid in zip(user_ids, item_ids)
        ], dtype=bool)

        if valid_mask.any():
            v_user_ids = user_ids[valid_mask]
            v_item_ids = item_ids[valid_mask]
            u_idxs = np.array([self.user_mapping[uid] for uid in v_user_ids])
            i_idxs = np.array([self.item_mapping[iid] for iid in v_item_ids])
            # Vectorized: global_mean + user_bias + item_bias + dot(user_vec, item_vec)
            dot_products = np.einsum('ij,ij->i',
                                     self.user_matrix[u_idxs],
                                     self.item_matrix[:, i_idxs].T)
            preds[valid_mask] = (self.global_mean
                                 + self.user_biases[u_idxs]
                                 + self.item_biases[i_idxs]
                                 + dot_products)
        return self._clip_array(preds)


    def get_hyperparameters(self):
        return {"n_factors": self.n_factors}

    def set_hyperparameters(self, params):
        self.n_factors = params.get("n_factors", self.n_factors)
        self.is_fitted = False