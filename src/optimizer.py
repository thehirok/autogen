import numpy as np
import pandas as pd
import optuna
import logging
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:

    def __init__(self, recommender_class, n_trials=30,
                 timeout=300, random_state=None):
        self.recommender_class = recommender_class
        self.n_trials          = n_trials
        self.timeout           = timeout
        self.random_state      = RANDOM_SEED if random_state is None else random_state
        self.best_params       = None
        self.best_recommender  = None

    def _param_space(self, trial):
        name = self.recommender_class.__name__
        if name == 'SVDRecommender':
            return {'n_factors': trial.suggest_int('n_factors', 20, 200)}
        elif name == 'KNNRecommender':
            return {
                'k':          trial.suggest_int('k', 10, 100),
                'user_based': trial.suggest_categorical(
                    'user_based', [True, False]),
            }
        elif name == 'NCFRecommender':
            return {
                'embedding_dim': trial.suggest_int('embedding_dim', 16, 64),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-4, 1e-2, log=True),
                'num_epochs':    trial.suggest_int('num_epochs', 5, 20),
            }
        elif name == 'NMFRecommender':
            return {
                'n_factors':     trial.suggest_int('n_factors', 5, 50),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-4, 1e-2, log=True),
            }
        elif name == 'SVDppRecommender':
            return {
                'n_factors': trial.suggest_int('n_factors', 10, 50),
                'n_epochs':  trial.suggest_int('n_epochs', 10, 30),
            }
        elif name == 'PEFTNCFRecommender':
            return {
                'lora_r':        trial.suggest_int('lora_r', 4, 16),
                'lora_alpha':    trial.suggest_int('lora_alpha', 8, 32),
                'adapt_epochs':  trial.suggest_int('adapt_epochs', 3, 10),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-4, 1e-2, log=True),
            }
        else:
            return {}

    def _objective(self, trial, train_data, val_data):
        from .evaluate import Evaluator
        params    = self._param_space(trial)
        rec       = self.recommender_class(**params)
        rec.fit(train_data)
        evaluator = Evaluator(random_state=self.random_state)
        metrics   = evaluator.evaluate_recommender(rec, val_data, k_values=[10])
        return metrics['rmse']

    def optimize(self, data, val_ratio=0.2):
        train, val = train_test_split(
            data, test_size=val_ratio, random_state=self.random_state)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(
            lambda trial: self._objective(trial, train, val),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        self.best_params = study.best_params
        self.best_recommender = self.recommender_class(**self.best_params)
        self.best_recommender.fit(data)
        logger.info(f"Best params for "
                    f"{self.recommender_class.__name__}: {self.best_params}")
        return self.best_params

    def get_best_recommender(self):
        if self.best_recommender is None:
            raise ValueError("Run optimize() first.")
        return self.best_recommender