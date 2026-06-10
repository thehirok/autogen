import numpy as np
import pandas as pd
import joblib
import os
import logging
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Transformer + LoRA meta-model
# ---------------------------------------------------------------

class TransformerMetaNetwork(nn.Module):
    """Small Transformer encoder for meta-feature classification."""

    def __init__(self, n_features=29, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, n_classes=7):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, batch_first=True,
            dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, d_model)
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)  # (batch, d_model)
        return self.classifier(x)


class TransformerLoRAClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around TransformerMetaNetwork + LoRA.

    This allows the MetaModel class to use it with cross_val_score
    and the existing fit/predict interface.
    """

    def __init__(self, n_features=29, n_classes=7, d_model=64,
                 nhead=4, num_layers=2, dim_ff=128,
                 lora_r=4, lora_alpha=8, lr=0.01,
                 epochs=100, random_state=42):
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.d_model      = d_model
        self.nhead        = nhead
        self.num_layers   = num_layers
        self.dim_ff       = dim_ff
        self.lora_r       = lora_r
        self.lora_alpha   = lora_alpha
        self.lr           = lr
        self.epochs       = epochs
        self.random_state = random_state
        self.device       = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_       = None
        self.classes_     = None

    def _build(self):
        torch.manual_seed(self.random_state)
        base = TransformerMetaNetwork(
            n_features=self.n_features, d_model=self.d_model,
            nhead=self.nhead, num_layers=self.num_layers,
            dim_ff=self.dim_ff, n_classes=self.n_classes
        ).to(self.device)

        # Apply LoRA to transformer linear layers
        target_modules = []
        for name, module in base.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias='none',
        )
        return get_peft_model(base, lora_config)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([label_map[c] for c in y])

        # Rebuild model with correct n_features / n_classes
        self.n_features = X.shape[1]
        self.model_ = self._build()

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_enc, dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model_.parameters()),
            lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model_(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def get_params(self, deep=True):
        return {
            'n_features': self.n_features, 'n_classes': self.n_classes,
            'd_model': self.d_model, 'nhead': self.nhead,
            'num_layers': self.num_layers, 'dim_ff': self.dim_ff,
            'lora_r': self.lora_r, 'lora_alpha': self.lora_alpha,
            'lr': self.lr, 'epochs': self.epochs,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class MetaModel:

    def __init__(self, model_type='RandomForest', random_state=42):
        self.model_type    = model_type
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model         = self._build_model()
        self.feature_names = None
        self.is_fitted     = False
        self._needs_encoded_labels = (model_type in ('XGBoost', 'TransformerLoRA'))

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
        elif self.model_type == 'TransformerLoRA':
            return TransformerLoRAClassifier(
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