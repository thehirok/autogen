import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DatasetAnalyzer:

    def __init__(self):
        self.dataset  = None
        self.features = {}

    def load_dataset(self, data, user_col='user_id',
                     item_col='item_id', rating_col='rating'):
        if isinstance(data, str):
            self.dataset = pd.read_parquet(data)
        else:
            self.dataset = data.copy()
        self.dataset = self.dataset.rename(columns={
            user_col:   'user_id',
            item_col:   'item_id',
            rating_col: 'rating',
        })
        self.dataset = self.dataset[['user_id', 'item_id', 'rating']].dropna()
        self.features = {}

    def extract_features(self):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        df = self.dataset
        n_users   = df['user_id'].nunique()
        n_items   = df['item_id'].nunique()
        n_ratings = len(df)

        # --- sparsity ---
        total   = n_users * n_items
        sparsity = 1.0 - (n_ratings / total) if total > 0 else 1.0

        # --- user/item ratio ---
        user_item_ratio = n_users / n_items if n_items > 0 else 0.0

        # --- rating distribution ---
        ratings      = df['rating'].to_numpy()
        rating_mean  = float(ratings.mean())
        rating_median= float(np.median(ratings))
        rating_std   = float(ratings.std()) if len(ratings) > 1 else 0.0
        rating_min   = float(ratings.min())
        rating_max   = float(ratings.max())
        rating_range = rating_max - rating_min
        rating_skew  = float(stats.skew(ratings))     if len(ratings) > 2 else 0.0
        rating_kurt  = float(stats.kurtosis(ratings)) if len(ratings) > 3 else 0.0

        # rating entropy
        counts = np.bincount(
            np.round(ratings).astype(int) - int(np.round(ratings).min()))
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        rating_entropy = float(-np.sum(probs * np.log2(probs)))

        # --- user activity ---
        rpu = df.groupby('user_id').size()
        avg_rpu    = float(rpu.mean())
        median_rpu = float(rpu.median())
        std_rpu    = float(rpu.std()) if len(rpu) > 1 else 0.0
        min_rpu    = float(rpu.min())
        max_rpu    = float(rpu.max())

        # power-law exponent (log-log slope of complementary CDF)
        sorted_rpu = np.sort(rpu.values)[::-1]
        ranks      = np.arange(1, len(sorted_rpu) + 1)
        valid      = sorted_rpu > 0
        if valid.sum() > 2:
            slope, _, _, _, _ = stats.linregress(
                np.log(ranks[valid]), np.log(sorted_rpu[valid]))
            power_law_exp = float(-slope)
        else:
            power_law_exp = 0.0

        # cold-start ratio (users with < 5 ratings)
        cold_start_users = float((rpu < 5).sum() / len(rpu)) if len(rpu) > 0 else 0.0

        # --- item popularity ---
        rpi = df.groupby('item_id').size()
        avg_rpi    = float(rpi.mean())
        median_rpi = float(rpi.median())
        std_rpi    = float(rpi.std()) if len(rpi) > 1 else 0.0
        min_rpi    = float(rpi.min())
        max_rpi    = float(rpi.max())

        cold_start_items = float((rpi < 5).sum() / len(rpi)) if len(rpi) > 0 else 0.0

        # --- Gini coefficient (item popularity) ---
        sorted_rpi = np.sort(rpi.values)
        n          = len(sorted_rpi)
        if n > 0 and sorted_rpi.sum() > 0:
            idx  = np.arange(1, n + 1)
            gini = float((2 * (idx * sorted_rpi).sum()
                          / (n * sorted_rpi.sum())) - (n + 1) / n)
        else:
            gini = 0.0

        # --- interaction density in top 20% of users ---
        top_users     = rpu.nlargest(max(1, int(0.2 * len(rpu))))
        top20_density = float(top_users.sum() / n_ratings) if n_ratings > 0 else 0.0

        self.features = {
            'n_users':            n_users,
            'n_items':            n_items,
            'n_ratings':          n_ratings,
            'sparsity':           sparsity,
            'user_item_ratio':    user_item_ratio,
            'rating_mean':        rating_mean,
            'rating_median':      rating_median,
            'rating_std':         rating_std,
            'rating_min':         rating_min,
            'rating_max':         rating_max,
            'rating_range':       rating_range,
            'rating_skew':        rating_skew,
            'rating_kurtosis':    rating_kurt,
            'rating_entropy':     rating_entropy,
            'avg_ratings_per_user':    avg_rpu,
            'median_ratings_per_user': median_rpu,
            'std_ratings_per_user':    std_rpu,
            'min_ratings_per_user':    min_rpu,
            'max_ratings_per_user':    max_rpu,
            'power_law_exponent':      power_law_exp,
            'cold_start_user_ratio':   cold_start_users,
            'avg_ratings_per_item':    avg_rpi,
            'median_ratings_per_item': median_rpi,
            'std_ratings_per_item':    std_rpi,
            'min_ratings_per_item':    min_rpi,
            'max_ratings_per_item':    max_rpi,
            'cold_start_item_ratio':   cold_start_items,
            'item_popularity_gini':    gini,
            'top20_user_density':      top20_density,
        }

        return self.features

    def get_feature_names(self):
        if not self.features:
            raise ValueError("No features extracted yet.")
        return list(self.features.keys())

    def as_dataframe(self):
        if not self.features:
            raise ValueError("No features extracted yet.")
        return pd.DataFrame([self.features])

    def summary_report(self):
        f = self.features
        if not f:
            raise ValueError("No features extracted yet.")
        lines = [
            f"Users: {f['n_users']}  Items: {f['n_items']}  "
            f"Ratings: {f['n_ratings']}",
            f"Sparsity: {f['sparsity']*100:.2f}%  "
            f"User/item ratio: {f['user_item_ratio']:.2f}",
            f"Rating range: {f['rating_min']}–{f['rating_max']}  "
            f"Mean: {f['rating_mean']:.2f}  Std: {f['rating_std']:.2f}",
            f"Rating entropy: {f['rating_entropy']:.3f}  "
            f"Skew: {f['rating_skew']:.3f}",
            f"Avg ratings/user: {f['avg_ratings_per_user']:.1f}  "
            f"Cold-start users: {f['cold_start_user_ratio']*100:.1f}%",
            f"Avg ratings/item: {f['avg_ratings_per_item']:.1f}  "
            f"Cold-start items: {f['cold_start_item_ratio']*100:.1f}%",
            f"Item Gini: {f['item_popularity_gini']:.3f}  "
            f"Power-law exp: {f['power_law_exponent']:.3f}",
        ]
        return "\n".join(lines)