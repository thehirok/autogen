import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from config import SAMPLE_SIZES, N_SUBSAMPLE_TRIALS, STABILITY_TOLERANCE

from .dataset_analyzer import DatasetAnalyzer
from .subsampler        import Subsampler

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Measures how quickly each meta-feature stabilises as sample
    size grows toward the full dataset value.

    Key outputs
    -----------
    stability_curves : pd.DataFrame
        rows = (dataset, feature, sample_size, trial)
        columns include 'value' and 'relative_error'

    stabilization_points : pd.Series
        for each feature, the smallest N where relative error
        stays within STABILITY_TOLERANCE for all larger sizes

    ess : pd.Series
        Early Selection Score — combines stability threshold
        and feature importance (set externally via set_importances)
    """

    def __init__(self, sample_sizes=None, n_trials=None,
                 tolerance=None, random_state=42):
        self.sample_sizes = sample_sizes or SAMPLE_SIZES
        self.n_trials     = n_trials     or N_SUBSAMPLE_TRIALS
        self.tolerance    = tolerance    or STABILITY_TOLERANCE
        self.random_state = random_state
        self.subsampler   = Subsampler(
            sample_sizes=self.sample_sizes,
            n_trials=self.n_trials,
            random_state=random_state,
        )
        self.analyzer         = DatasetAnalyzer()
        self.stability_curves = None
        self.full_features    = None
        self.importances      = None

    def compute_full_features(self, data):
        """Extract features from the complete dataset."""
        self.analyzer.load_dataset(data)
        self.full_features = self.analyzer.extract_features()
        return self.full_features

    def compute_stability_curves(self, data, dataset_name='dataset'):
        """
        For every (sample_size, trial) pair, extract features and
        record the relative error vs. full-dataset values.

        Parameters
        ----------
        data         : pd.DataFrame
        dataset_name : str   label used in output rows

        Returns
        -------
        pd.DataFrame  stability curve records
        """
        full = self.compute_full_features(data)
        feature_names = list(full.keys())
        records = []

        for n in tqdm(self.sample_sizes, desc=f"Stability [{dataset_name}]"):
            if n >= len(data):
                # treat full dataset as a "trial" at this size
                for feat in feature_names:
                    records.append({
                        'dataset':        dataset_name,
                        'feature':        feat,
                        'sample_size':    n,
                        'trial':          0,
                        'value':          full[feat],
                        'full_value':     full[feat],
                        'relative_error': 0.0,
                    })
                continue

            for trial in range(self.n_trials):
                sample = self.subsampler.subsample(data, n, trial=trial)
                self.analyzer.load_dataset(sample)
                try:
                    feats = self.analyzer.extract_features()
                except Exception as e:
                    logger.warning(
                        f"Feature extraction failed for {dataset_name} "
                        f"n={n} trial={trial}: {e}")
                    continue

                for feat in feature_names:
                    full_val   = full[feat]
                    sample_val = feats.get(feat, np.nan)
                    if full_val != 0:
                        rel_err = abs(sample_val - full_val) / abs(full_val)
                    else:
                        rel_err = abs(sample_val)
                    records.append({
                        'dataset':        dataset_name,
                        'feature':        feat,
                        'sample_size':    n,
                        'trial':          trial,
                        'value':          sample_val,
                        'full_value':     full_val,
                        'relative_error': rel_err,
                    })

        self.stability_curves = pd.DataFrame(records)
        return self.stability_curves

    def compute_stabilization_points(self):
        """
        For each feature find N* — the smallest sample size where
        mean relative error across trials is within self.tolerance
        for all larger sizes.

        Returns
        -------
        pd.Series  {feature: N*}   (NaN if never stable)
        """
        if self.stability_curves is None:
            raise ValueError("Run compute_stability_curves() first.")

        # mean relative error per (feature, sample_size)
        mean_err = (
            self.stability_curves
            .groupby(['feature', 'sample_size'])['relative_error']
            .mean()
        )

        results = {}
        feature_names = self.stability_curves['feature'].unique()
        sorted_sizes  = sorted(self.sample_sizes)

        for feat in feature_names:
            stable_at = np.nan
            for i, n in enumerate(sorted_sizes):
                # check this size and all larger sizes are within tolerance
                remaining = sorted_sizes[i:]
                errors = [
                    mean_err.get((feat, m), np.nan)
                    for m in remaining
                ]
                errors = [e for e in errors if not np.isnan(e)]
                if errors and max(errors) <= self.tolerance:
                    stable_at = n
                    break
            results[feat] = stable_at

        return pd.Series(results, name='stabilization_point')

    def set_importances(self, importances):
        """
        Provide feature importances from the meta-model (e.g. SHAP).

        Parameters
        ----------
        importances : dict  {feature_name: importance_score}
        """
        self.importances = importances

    def compute_ess(self):
        """
        Early Selection Score (ESS) for each feature.

        ESS = importance / log2(stabilization_point + 2)

        Higher ESS = feature is both important AND stabilises early.

        Returns
        -------
        pd.Series  {feature: ESS}
        """
        if self.importances is None:
            raise ValueError("Set importances first via set_importances().")

        stab = self.compute_stabilization_points()
        ess  = {}
        for feat, stab_n in stab.items():
            imp = self.importances.get(feat, 0.0)
            if np.isnan(stab_n) or stab_n <= 0:
                ess[feat] = 0.0
            else:
                ess[feat] = imp / np.log2(stab_n + 2)
        return pd.Series(ess, name='ESS').sort_values(ascending=False)

    def summary(self):
        """
        Print a quick summary of stabilization points.
        """
        stab = self.compute_stabilization_points()
        stab_sorted = stab.sort_values()
        print(f"\nFeature stabilization points "
              f"(tolerance={self.tolerance*100:.0f}%):\n")
        for feat, n in stab_sorted.items():
            status = f"N* = {int(n)}" if not np.isnan(n) else "never stable"
            print(f"  {feat:<35} {status}")