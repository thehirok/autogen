import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from .dataset_analyzer  import DatasetAnalyzer
from .subsampler         import Subsampler
from .meta_model         import MetaModel
from config import RANDOM_SEED, ALGORITHMS

logger = logging.getLogger(__name__)


class EarlySelector:
    """
    Main class for sample-efficient algorithm selection.

    Given a (possibly small) dataset, it:
      1. Extracts meta-features from the available data
      2. Queries the fitted meta-model for algorithm probabilities
      3. Returns a ranked list of algorithms with confidence

    Also supports the full experimental pipeline:
      - fit_meta_model  : train on many datasets
      - select          : predict for a new dataset (any size)
      - evaluate_at_n   : measure selection accuracy vs sample size
    """

    def __init__(self, model_type='RandomForest', random_state=None):
        self.model_type   = model_type
        self.random_state = RANDOM_SEED if random_state is None else random_state
        self.meta_model   = MetaModel(
            model_type=model_type, random_state=self.random_state)
        self.analyzer     = DatasetAnalyzer()
        self.subsampler   = Subsampler(random_state=self.random_state)
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # Training the meta-model
    # ------------------------------------------------------------------

    def fit_meta_model(self, meta_features_df, labels):
        """
        Train the internal meta-model.

        Parameters
        ----------
        meta_features_df : pd.DataFrame
            One row per dataset, columns = feature names.
        labels : array-like
            Best algorithm name for each dataset.
        """
        self.meta_model.fit(meta_features_df, labels)
        self.is_fitted = True
        logger.info(
            f"EarlySelector meta-model ({self.model_type}) fitted on "
            f"{len(meta_features_df)} datasets.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def select(self, data):
        """
        Select the best algorithm for a dataset (any size).

        Parameters
        ----------
        data : pd.DataFrame  columns [user_id, item_id, rating]

        Returns
        -------
        dict with keys:
            'algorithm'   : str   top recommendation
            'confidence'  : float margin between top-2 probabilities
            'ranking'     : list  all algorithms sorted by probability
            'probabilities': dict {algorithm: probability}
        """
        if not self.is_fitted:
            raise ValueError("EarlySelector not fitted. "
                             "Call fit_meta_model() first.")
        self.analyzer.load_dataset(data)
        features = self.analyzer.extract_features()
        features_df = pd.DataFrame([features])

        algo      = self.meta_model.predict(features_df)
        probas    = self.meta_model.predict_proba(features_df)
        confidence= self.meta_model.confidence(features_df)
        ranking   = sorted(probas, key=probas.get, reverse=True)

        return {
            'algorithm':    algo,
            'confidence':   confidence,
            'ranking':      ranking,
            'probabilities': probas,
        }

    def select_at_n(self, data, n, trial=0):
        """
        Select algorithm using only n ratings from data.

        Parameters
        ----------
        data  : pd.DataFrame  full dataset
        n     : int           subsample size
        trial : int           random trial index

        Returns
        -------
        str  selected algorithm name
        """
        sample = self.subsampler.subsample(data, n, trial=trial)
        return self.select(sample)['algorithm']

    # ------------------------------------------------------------------
    # Experimental evaluation
    # ------------------------------------------------------------------

    def evaluate_selection_accuracy(self, datasets, oracle_labels,
                                    sample_sizes=None, n_trials=10):
        """
        Measure selection accuracy at each sample size across datasets.

        Parameters
        ----------
        datasets      : dict {name: pd.DataFrame}
        oracle_labels : dict {name: str}  true best algorithm per dataset
        sample_sizes  : list of int
        n_trials      : int

        Returns
        -------
        pd.DataFrame  columns [dataset, sample_size, trial, selected,
                               oracle, correct]
        """
        if not self.is_fitted:
            raise ValueError("EarlySelector not fitted.")

        from config import SAMPLE_SIZES
        sizes = sample_sizes or SAMPLE_SIZES
        records = []

        for name, data in datasets.items():
            oracle = oracle_labels.get(name)
            if oracle is None:
                logger.warning(f"No oracle label for {name}, skipping.")
                continue

            for n in sizes:
                if n > len(data):
                    continue
                for trial in range(n_trials):
                    try:
                        selected = self.select_at_n(data, n, trial=trial)
                    except Exception as e:
                        logger.warning(
                            f"Selection failed {name} n={n} t={trial}: {e}")
                        selected = None
                    records.append({
                        'dataset':     name,
                        'sample_size': n,
                        'trial':       trial,
                        'selected':    selected,
                        'oracle':      oracle,
                        'correct':     selected == oracle,
                    })

        return pd.DataFrame(records)

    def save(self, path):
        self.meta_model.save(path)
        logger.info(f"EarlySelector saved to {path}")

    def load(self, path):
        self.meta_model.load(path)
        self.is_fitted = True
        logger.info(f"EarlySelector loaded from {path}")
        return self