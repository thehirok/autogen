import numpy as np
import pandas as pd
import logging
from config import SAMPLE_SIZES, RANDOM_SEED

logger = logging.getLogger(__name__)


class Subsampler:
    """
    Generates stratified subsamples of a rating dataset at multiple
    sample sizes. Stratification is by user — each user contributes
    proportionally to the subsample so that active users do not
    dominate small samples.
    """

    def __init__(self, sample_sizes=None, n_trials=10, random_state=None):
        self.sample_sizes = sample_sizes or SAMPLE_SIZES
        self.n_trials     = n_trials
        self.random_state = RANDOM_SEED if random_state is None else random_state

    def subsample(self, data, n, trial=0):
        """
        Return a single stratified subsample of size n from data.

        Parameters
        ----------
        data : pd.DataFrame  columns [user_id, item_id, rating]
        n    : int           target number of ratings
        trial: int           trial index (shifts random seed)

        Returns
        -------
        pd.DataFrame  subsample of size min(n, len(data))
        """
        if n >= len(data):
            return data.copy()

        rng   = np.random.RandomState(self.random_state + trial)
        users = data['user_id'].unique()

        # how many ratings each user contributes proportionally
        user_counts = data.groupby('user_id').size()
        total       = user_counts.sum()
        allocated   = (user_counts / total * n).astype(int)

        # give leftover slots to users with most ratings
        remainder = n - allocated.sum()
        if remainder > 0:
            top_users = user_counts.nlargest(remainder).index
            allocated[top_users] += 1

        parts = []
        for user_id, quota in allocated.items():
            if quota <= 0:
                continue
            user_rows = data[data['user_id'] == user_id]
            take      = min(quota, len(user_rows))
            parts.append(user_rows.sample(n=take, random_state=rng))

        result = pd.concat(parts)

        # if we're still short (due to int rounding), top-up randomly
        # Use original index to correctly exclude already-selected rows
        if len(result) < n:
            remaining = data.drop(result.index, errors='ignore')
            if len(remaining) > 0:
                extra = remaining.sample(
                    n=min(n - len(result), len(remaining)),
                    random_state=rng)
                result = pd.concat([result, extra])

        result = result.reset_index(drop=True)

        return result

    def subsample_all_sizes(self, data, trial=0):
        """
        Return a dict mapping each sample size to one subsample.

        Parameters
        ----------
        data  : pd.DataFrame
        trial : int

        Returns
        -------
        dict  {sample_size: pd.DataFrame}
        """
        return {
            n: self.subsample(data, n, trial=trial)
            for n in self.sample_sizes
            if n <= len(data)
        }

    def subsample_multi_trial(self, data, n):
        """
        Return a list of n_trials subsamples of size n.
        Used to average out sampling variance.

        Parameters
        ----------
        data : pd.DataFrame
        n    : int

        Returns
        -------
        list of pd.DataFrame
        """
        return [
            self.subsample(data, n, trial=t)
            for t in range(self.n_trials)
        ]

    def validate(self, data):
        """
        Check that data has the required columns and enough ratings
        for the smallest sample size.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        bool
        """
        required = {'user_id', 'item_id', 'rating'}
        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            raise ValueError(f"Dataset missing columns: {missing}")
        min_size = min(self.sample_sizes)
        if len(data) < min_size:
            raise ValueError(
                f"Dataset has {len(data)} ratings — "
                f"smaller than minimum sample size {min_size}.")
        return True