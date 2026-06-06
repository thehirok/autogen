import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class KNNRecommender(BaseRecommender):

    def __init__(self, k=40, user_based=True, random_state=42):
        super().__init__("KNN")
        self.k = k
        self.user_based = user_based
        self.random_state = random_state
        self.similarities = None
        self.user_item_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None

    def fit(self, train_data):
        self.train_data = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        self.user_item_matrix = self._fill_matrix(processed, n_users, n_items)

        # Boolean mask: True where a rating exists (handles 0-valued ratings)
        self._rated_mask = np.zeros((n_users, n_items), dtype=bool)
        user_idxs = processed['user_idx'].to_numpy()
        item_idxs = processed['item_idx'].to_numpy()
        self._rated_mask[user_idxs, item_idxs] = True
        mask = self._rated_mask

        user_counts = mask.sum(axis=1)
        user_counts = np.where(user_counts == 0, 1, user_counts)
        self.user_means = (self.user_item_matrix * mask).sum(axis=1) / user_counts

        item_counts = mask.sum(axis=0)
        item_counts = np.where(item_counts == 0, 1, item_counts)
        self.item_means = (self.user_item_matrix * mask).sum(axis=0) / item_counts

        centered = self.user_item_matrix.copy()
        for u in range(n_users):
            rated = mask[u]
            if rated.any():
                centered[u, rated] -= self.user_means[u]

        if self.user_based:
            self.similarities = cosine_similarity(centered)
        else:
            self.similarities = cosine_similarity(centered.T)

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean

        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]

        if self.user_based:
            sims = self.similarities[u]
            ratings_col = self.user_item_matrix[:, i]
            rated_mask = self._rated_mask[:, i].copy()
            rated_mask[u] = False
            if not rated_mask.any():
                return self.global_mean
            neighbor_sims = sims[rated_mask]
            neighbor_ratings = ratings_col[rated_mask]
            neighbor_means = self.user_means[rated_mask]
            top_k = min(self.k, rated_mask.sum())
            top_idx = np.argpartition(neighbor_sims, -top_k)[-top_k:]
            w = neighbor_sims[top_idx]
            r = neighbor_ratings[top_idx] - neighbor_means[top_idx]
            denom = np.abs(w).sum()
            if denom == 0:
                return self.user_means[u]
            pred = self.user_means[u] + np.dot(w, r) / denom
        else:
            sims = self.similarities[i]
            ratings_row = self.user_item_matrix[u]
            rated_mask = self._rated_mask[u].copy()
            rated_mask[i] = False
            if not rated_mask.any():
                return self.global_mean
            neighbor_sims = sims[rated_mask]
            neighbor_ratings = ratings_row[rated_mask]
            neighbor_means = self.item_means[rated_mask]
            top_k = min(self.k, rated_mask.sum())
            top_idx = np.argpartition(neighbor_sims, -top_k)[-top_k:]
            w = neighbor_sims[top_idx]
            r = neighbor_ratings[top_idx] - neighbor_means[top_idx]
            denom = np.abs(w).sum()
            if denom == 0:
                return self.item_means[i]
            pred = self.item_means[i] + np.dot(w, r) / denom

        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        scores = np.full(n_items, self.global_mean, dtype=np.float64)

        if self.user_based:
            sims = self.similarities[u].copy()
            sims[u] = 0  # exclude self

            n_users = len(self.user_mapping)
            top_k = min(self.k, n_users - 1)

            # Global top-k: find the k most similar users overall
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            w = sims[top_idx]                                    # (k,)
            neighbor_ratings = self.user_item_matrix[top_idx]     # (k, n_items)
            neighbor_means = self.user_means[top_idx]             # (k,)
            neighbor_rated = self._rated_mask[top_idx]            # (k, n_items)

            # Centered ratings, zero out unrated entries
            centered = (neighbor_ratings - neighbor_means[:, np.newaxis]) * neighbor_rated

            # Weighted sum: w @ centered, with weights zeroed for unrated
            w_masked = w[:, np.newaxis] * neighbor_rated          # (k, n_items)
            num = (w_masked * centered).sum(axis=0)               # (n_items,)
            denom = np.abs(w_masked).sum(axis=0)                  # (n_items,)

            valid = denom > 0
            scores[valid] = self.user_means[u] + num[valid] / denom[valid]
        else:
            # Item-based: find top-k similar items among those rated by user u
            rated_mask_u = self._rated_mask[u]
            rated_items = np.where(rated_mask_u)[0]

            if len(rated_items) == 0:
                return self._clip_array(scores)

            ratings_sub = self.user_item_matrix[u, rated_items]   # (n_rated,)
            means_sub = self.item_means[rated_items]              # (n_rated,)
            centered = ratings_sub - means_sub                    # (n_rated,)

            n_rated = len(rated_items)
            if n_rated <= self.k:
                # Use all rated items as neighbors
                sims_sub = self.similarities[:, rated_items].copy()  # (n_items, n_rated)
                # Zero out self-similarity for rated items
                for idx_j, item_j in enumerate(rated_items):
                    sims_sub[item_j, idx_j] = 0.0
                num = sims_sub @ centered                         # (n_items,)
                denom = np.abs(sims_sub).sum(axis=1)              # (n_items,)
                valid = denom > 0
                scores[valid] = self.item_means[valid] + num[valid] / denom[valid]
            else:
                # For each item, we need top-k from rated_items.
                # Use global top-k among rated items per-item via matrix ops.
                sims_sub = self.similarities[:, rated_items].copy()  # (n_items, n_rated)
                # Zero out self-similarity for rated items
                for idx_j, item_j in enumerate(rated_items):
                    sims_sub[item_j, idx_j] = 0.0

                # For each item (row), find top-k columns
                # Use argpartition along axis=1 for all items at once
                top_k_idx = np.argpartition(sims_sub, -self.k, axis=1)[:, -self.k:]  # (n_items, k)

                # Gather the top-k similarities and centered ratings
                row_idx = np.arange(n_items)[:, np.newaxis]       # (n_items, 1)
                w = sims_sub[row_idx, top_k_idx]                  # (n_items, k)
                r = centered[top_k_idx]                           # (n_items, k)

                num = (w * r).sum(axis=1)                         # (n_items,)
                denom = np.abs(w).sum(axis=1)                     # (n_items,)
                valid = denom > 0
                scores[valid] = self.item_means[valid] + num[valid] / denom[valid]

        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        """Batch prediction using cached predict_all_items per user."""
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)

        # Cache predict_all_items per unique user
        cache = {}
        for idx in range(len(user_ids)):
            uid = user_ids[idx]
            iid = item_ids[idx]
            if uid not in self.user_mapping or iid not in self.item_mapping:
                continue
            if uid not in cache:
                try:
                    cache[uid] = self.predict_all_items(uid)
                except Exception:
                    cache[uid] = None
            if cache[uid] is not None:
                preds[idx] = cache[uid][self.item_mapping[iid]]
        return preds
    def get_hyperparameters(self):
        return {"k": self.k, "user_based": self.user_based}

    def set_hyperparameters(self, params):
        self.k = params.get("k", self.k)
        self.user_based = params.get("user_based", self.user_based)
        self.is_fitted = False