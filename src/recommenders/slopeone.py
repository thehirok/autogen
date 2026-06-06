import numpy as np
import logging
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class SlopeOneRecommender(BaseRecommender):

    def __init__(self, random_state=42):
        super().__init__("SlopeOne")
        self.random_state = random_state
        self.freq  = None
        self.dev   = None
        self.global_mean = None
        self.user_ratings = None

    def fit(self, train_data):
        self.train_data = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_items = len(self.item_mapping)

        self.freq = np.zeros((n_items, n_items), dtype=np.int32)
        self.dev  = np.zeros((n_items, n_items), dtype=np.float64)

        # build per-user item→rating dict using vectorized access
        self.user_ratings = {}
        u_arr = processed['user_idx'].to_numpy(dtype=int)
        i_arr = processed['item_idx'].to_numpy(dtype=int)
        r_arr = processed['rating'].to_numpy(dtype=float)
        for idx in range(len(u_arr)):
            u = u_arr[idx]
            if u not in self.user_ratings:
                self.user_ratings[u] = {}
            self.user_ratings[u][i_arr[idx]] = r_arr[idx]

        # compute deviations — vectorized per user
        for u_items in self.user_ratings.values():
            items = np.array(list(u_items.keys()), dtype=np.int64)
            vals = np.array(list(u_items.values()), dtype=np.float64)
            if len(items) < 2:
                continue
            # outer difference: diffs[a, b] = vals[a] - vals[b]
            diffs = vals[:, np.newaxis] - vals[np.newaxis, :]
            # exclude diagonal (self-pairs)
            np.fill_diagonal(diffs, 0.0)
            # accumulate into freq and dev using advanced indexing
            ix = np.ix_(items, items)
            self.freq[ix] += 1
            self.dev[ix] += diffs
            # fix diagonal — undo the +1 on diagonal
            self.freq[items, items] -= 1

        # normalise
        mask = self.freq > 0
        self.dev[mask] /= self.freq[mask]

        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        if u not in self.user_ratings:
            return self.global_mean

        num = 0.0
        den = 0.0
        for j, r_uj in self.user_ratings[u].items():
            if j == i:
                continue
            f = self.freq[i][j]
            if f > 0:
                num += (self.dev[i][j] + r_uj) * f
                den += f
        if den == 0:
            return self.global_mean
        return self._clip(num / den)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        scores = np.full(n_items, self.global_mean, dtype=np.float64)

        if u not in self.user_ratings or not self.user_ratings[u]:
            return scores

        # Get rated items and ratings as arrays
        rated_items = np.array(list(self.user_ratings[u].keys()), dtype=np.int64)
        rated_vals  = np.array(list(self.user_ratings[u].values()), dtype=np.float64)

        # Vectorized: slice freq and dev for all items × rated items
        freqs_all = self.freq[:, rated_items]           # (n_items, n_rated)
        devs_all  = self.dev[:, rated_items]             # (n_items, n_rated)

        # Zero out self-references (where item == rated_item)
        for idx_j, rj in enumerate(rated_items):
            freqs_all[rj, idx_j] = 0

        # Mask valid entries (freq > 0)
        valid_mask = freqs_all > 0
        freqs_f = freqs_all.astype(np.float64) * valid_mask

        # Weighted numerator: sum of (dev + rating) * freq for valid entries
        num = ((devs_all + rated_vals[np.newaxis, :]) * freqs_f).sum(axis=1)
        den = freqs_f.sum(axis=1)

        has_den = den > 0
        scores[has_den] = num[has_den] / den[has_den]

        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        """Batch prediction using cached predict_all_items per user."""
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)

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
        return {}

    def set_hyperparameters(self, params):
        self.is_fitted = False