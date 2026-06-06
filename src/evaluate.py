import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, random_state=42):
        self.random_state = random_state

    def evaluate_recommender(self, recommender, test_data, k_values=None):
        if not recommender.is_fitted:
            raise ValueError("Recommender not fitted.")
        if k_values is None:
            k_values = [5, 10, 20]

        rmse, mae = self._compute_rating_metrics(recommender, test_data)

        precision_at_k = {}
        recall_at_k    = {}
        ndcg_at_k      = {}

        # Pre-compute per-user actual item sets via groupby (O(n) not O(n×users))
        user_actual = (
            test_data.groupby('user_id')['item_id']
            .apply(set).to_dict()
        )
        # Remove empty sets
        user_actual = {uid: items for uid, items in user_actual.items() if items}

        # Pre-compute NDCG discount array once for max k
        max_k = max(k_values)
        discounts = 1.0 / np.log2(np.arange(2, max_k + 2))  # discount[rank] for rank 0..max_k-1

        # Call recommend ONCE per user with max(k), then slice for smaller k values
        user_recs = {}
        for user_id in user_actual:
            try:
                recs = recommender.recommend(user_id, n=max_k)
                user_recs[user_id] = list(recs['item_id'])
            except Exception as e:
                logger.debug(f"Skipping user {user_id}: {e}")

        for k in k_values:
            p_scores, r_scores, n_scores = [], [], []
            for user_id, actual_items in user_actual.items():
                if user_id not in user_recs:
                    continue
                rec_list = user_recs[user_id][:k]
                rec_items = set(rec_list)
                hits = len(actual_items & rec_items)
                p_scores.append(hits / k if k > 0 else 0.0)
                r_scores.append(hits / len(actual_items))
                n_scores.append(
                    self._ndcg_from_list(rec_list, actual_items, k, discounts))

            precision_at_k[k] = float(np.mean(p_scores)) if p_scores else 0.0
            recall_at_k[k]    = float(np.mean(r_scores)) if r_scores else 0.0
            ndcg_at_k[k]      = float(np.mean(n_scores)) if n_scores else 0.0

        return {
            'rmse':           rmse,
            'mae':            mae,
            'precision_at_k': precision_at_k,
            'recall_at_k':    recall_at_k,
            'ndcg_at_k':      ndcg_at_k,
        }

    def _compute_rating_metrics(self, recommender, test_data):
        """Compute RMSE and MAE using vectorized predict_batch."""
        user_ids = test_data['user_id'].values
        item_ids = test_data['item_id'].values
        actual   = test_data['rating'].values.astype(np.float64)

        # Use predict_batch for vectorized prediction
        predicted = recommender.predict_batch(user_ids, item_ids)

        # Filter out NaN predictions (unknown users/items)
        valid = ~np.isnan(predicted)
        if not valid.any():
            return float('nan'), float('nan')

        actual    = actual[valid]
        predicted = predicted[valid]
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae  = float(np.mean(np.abs(actual - predicted)))
        return rmse, mae

    def _ndcg_from_list(self, rec_list, actual_items, k, discounts):
        try:
            dcg = sum(discounts[rank]
                      for rank, item in enumerate(rec_list)
                      if item in actual_items)
            idcg = discounts[:min(k, len(actual_items))].sum()
            return dcg / idcg if idcg > 0 else 0.0
        except Exception as e:
            logger.debug(f"NDCG computation error: {e}")
            return 0.0