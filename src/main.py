"""
AutoRecSys — top-level convenience wrapper.
For paper experiments use the scripts in experiments/ directly.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_analyzer import DatasetAnalyzer
from .meta_model       import MetaModel
from .early_selector   import EarlySelector
from .evaluate         import Evaluator
from .optimizer        import HyperparameterOptimizer
from .recommenders     import (SVDRecommender, KNNRecommender,
                               NCFRecommender, NMFRecommender,
                               SVDppRecommender, SlopeOneRecommender)
from config import RANDOM_SEED, ALGORITHMS, TEST_RATIO

logger = logging.getLogger(__name__)

# Import centralized map; also available via `from src import RECOMMENDER_MAP`
from . import RECOMMENDER_MAP


class AutoRecSys:
    """
    High-level API wrapping the full pipeline:
      load → select algorithm → train → evaluate → recommend
    """

    def __init__(self, model_type='RandomForest', random_state=None):
        self.random_state  = RANDOM_SEED if random_state is None else random_state
        self.selector      = EarlySelector(
            model_type=model_type, random_state=self.random_state)
        self.evaluator     = Evaluator(random_state=self.random_state)
        self.analyzer      = DatasetAnalyzer()
        self.dataset       = None
        self.recommender   = None
        self.selected_algo = None
        self.metrics       = None

    def load_dataset(self, path_or_df, user_col='user_id',
                     item_col='item_id', rating_col='rating'):
        if isinstance(path_or_df, str):
            ext = os.path.splitext(path_or_df)[1].lower()
            if ext == '.parquet':
                df = pd.read_parquet(path_or_df)
            elif ext == '.csv':
                df = pd.read_csv(path_or_df)
            else:
                df = pd.read_csv(path_or_df, sep='\t',
                                 names=[user_col, item_col,
                                        rating_col, 'ts'])
        else:
            df = path_or_df.copy()

        df = df.rename(columns={
            user_col: 'user_id', item_col: 'item_id',
            rating_col: 'rating'})
        self.dataset = df[['user_id', 'item_id', 'rating']].dropna()
        self.analyzer.load_dataset(self.dataset)
        logger.info(f"Loaded {len(self.dataset):,} ratings.")
        return self

    def fit(self, force_algorithm=None, optimize=False,
            n_trials=20, timeout=180):
        if self.dataset is None:
            raise ValueError("Call load_dataset() first.")

        if force_algorithm:
            self.selected_algo = force_algorithm
        elif self.selector.is_fitted:
            self.selected_algo = self.selector.select(self.dataset)['algorithm']
        else:
            logger.warning("Meta-model not fitted — defaulting to SVD.")
            self.selected_algo = 'SVD'

        logger.info(f"Selected algorithm: {self.selected_algo}")
        rec_class = RECOMMENDER_MAP[self.selected_algo]

        if optimize and self.selected_algo in ('SVD', 'KNN', 'NCF', 'NMF'):
            opt = HyperparameterOptimizer(
                rec_class, n_trials=n_trials,
                timeout=timeout, random_state=self.random_state)
            opt.optimize(self.dataset)
            self.recommender = opt.get_best_recommender()
        else:
            self.recommender = rec_class()
            self.recommender.fit(self.dataset)

        return self

    def evaluate(self, test_ratio=None):
        """Evaluate the fitted recommender on a held-out test split.

        Note: algorithm selection in fit() uses meta-features extracted
        from the full dataset, which includes what later becomes the test
        split here.  This is acceptable for the convenience API, but
        experiment scripts (experiments/) perform their own train/test
        splits *before* selection to avoid this leakage.
        """
        if self.recommender is None:
            raise ValueError("Call fit() first.")
        ratio = test_ratio or TEST_RATIO
        train, test = train_test_split(
            self.dataset, test_size=ratio,
            random_state=self.random_state)
        self.recommender.fit(train)
        self.metrics = self.evaluator.evaluate_recommender(
            self.recommender, test)
        self.recommender.fit(self.dataset)
        return self.metrics

    def recommend(self, user_id, n=10):
        if self.recommender is None or not self.recommender.is_fitted:
            raise ValueError("Call fit() first.")
        return self.recommender.recommend(user_id, n=n)

    def dataset_summary(self):
        return self.analyzer.summary_report()