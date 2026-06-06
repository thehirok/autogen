import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


class MetaModel:

    def __init__(self, model_type='RandomForest', random_state=42):
        self.model_type    = model_type
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model         = self._build_model()
        self.feature_names = None
        self.is_fitted     = False
        self._needs_encoded_labels = (model_type == 'XGBoost')

    def _build_model(self):
        if self.model_type == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=self.random_state)
        elif self.model_type == 'XGBoost':
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1,
                eval_metric='mlogloss',
                random_state=self.random_state)
        elif self.model_type == 'MLP':
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _encode_labels(self, y, fit=False):
        if fit:
            return self.label_encoder.fit_transform(y)
        return self.label_encoder.transform(y)

    def _decode_labels(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)

    def fit(self, X, y):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        # always fit the encoder so decode works for all model types
        y_encoded = self._encode_labels(y, fit=True)

        if self._needs_encoded_labels:
            self.model.fit(X_scaled, y_encoded)
        else:
            self.model.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("MetaModel not fitted. Call fit() first.")
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X_scaled = self.scaler.transform(X[self.feature_names])
        raw = self.model.predict(X_scaled)
        if self._needs_encoded_labels:
            raw = self._decode_labels(raw)
        return raw[0] if len(raw) == 1 else raw

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("MetaModel not fitted. Call fit() first.")
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X_scaled = self.scaler.transform(X[self.feature_names])
        probas = self.model.predict_proba(X_scaled)

        if self._needs_encoded_labels:
            classes = self._decode_labels(self.model.classes_)
        else:
            classes = self.model.classes_

        if len(probas) == 1:
            return {cls: float(p) for cls, p in zip(classes, probas[0])}
        return [{cls: float(p) for cls, p in zip(classes, row)} for row in probas]

    def confidence(self, X):
        probas = self.predict_proba(X)
        values = sorted(probas.values(), reverse=True)
        return float(values[0] - values[1]) if len(values) > 1 else 1.0

    def leave_one_out_accuracy(self, X, y):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X_scaled = self.scaler.fit_transform(X)
        y_enc    = self._encode_labels(y, fit=True)
        loo      = LeaveOneOut()
        fit_y    = y_enc if self._needs_encoded_labels else y
        scores   = cross_val_score(self.model, X_scaled, fit_y, cv=loo)
        return float(scores.mean())

    def feature_importances(self):
        if not self.is_fitted:
            raise ValueError("MetaModel not fitted.")
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names,
                            self.model.feature_importances_))
        return {}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({
            'model':                  self.model,
            'scaler':                 self.scaler,
            'label_encoder':          self.label_encoder,
            'feature_names':          self.feature_names,
            'model_type':             self.model_type,
            'is_fitted':              self.is_fitted,
            'needs_encoded_labels':   self._needs_encoded_labels,
        }, path)
        logger.info(f"MetaModel saved to {path}")

    def load(self, path):
        data = joblib.load(path)
        self.model                  = data['model']
        self.scaler                 = data['scaler']
        self.label_encoder          = data['label_encoder']
        self.feature_names          = data['feature_names']
        self.model_type             = data['model_type']
        self.is_fitted              = data['is_fitted']
        self._needs_encoded_labels  = data['needs_encoded_labels']
        logger.info(f"MetaModel loaded from {path}")
        return self