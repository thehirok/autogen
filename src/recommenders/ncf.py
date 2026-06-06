import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class NCFNetwork(nn.Module):

    def __init__(self, n_users, n_items, embedding_dim, layers):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        mlp_layers = []
        in_dim = embedding_dim * 2
        for out_dim in layers:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze(1)


class NCFRecommender(BaseRecommender):

    def __init__(self, embedding_dim=32, layers=None, learning_rate=0.001,
                 batch_size=256, num_epochs=10, random_state=42):
        super().__init__("NCF")
        self.embedding_dim = embedding_dim
        self.layers        = layers or [64, 32, 16]
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.num_epochs    = num_epochs
        self.random_state  = random_state
        self.model         = None
        self.global_mean   = None
        self.device        = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data):
        torch.manual_seed(self.random_state)
        self.train_data  = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        user_t   = torch.tensor(processed['user_idx'].to_numpy(), dtype=torch.long)
        item_t   = torch.tensor(processed['item_idx'].to_numpy(), dtype=torch.long)
        rating_t = torch.tensor(processed['rating'].to_numpy(),   dtype=torch.float32)

        dataset    = TensorDataset(user_t, item_t, rating_t)
        batch_size = min(self.batch_size, len(dataset))
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = NCFNetwork(
            n_users, n_items, self.embedding_dim, self.layers).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion  = nn.MSELoss()

        self.model.train()
        for epoch in range(self.num_epochs):
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(u_batch, i_batch)
                loss  = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = torch.tensor([self.user_mapping[user_id]], dtype=torch.long).to(self.device)
        i = torch.tensor([self.item_mapping[item_id]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            pred = self.model(u, i).item()
        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u_idx = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        u_t = torch.full((n_items,), u_idx, dtype=torch.long, device=self.device)
        i_t = torch.arange(n_items, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.model(u_t, i_t).cpu().numpy()
        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        """Vectorized batch prediction — single forward pass for all valid pairs."""
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        preds = np.full(len(user_ids), self.global_mean, dtype=np.float64)

        valid_mask = np.array([
            uid in self.user_mapping and iid in self.item_mapping
            for uid, iid in zip(user_ids, item_ids)
        ], dtype=bool)

        if valid_mask.any():
            v_uids = user_ids[valid_mask]
            v_iids = item_ids[valid_mask]
            u_idxs = np.array([self.user_mapping[uid] for uid in v_uids])
            i_idxs = np.array([self.item_mapping[iid] for iid in v_iids])

            u_t = torch.tensor(u_idxs, dtype=torch.long, device=self.device)
            i_t = torch.tensor(i_idxs, dtype=torch.long, device=self.device)
            with torch.no_grad():
                batch_preds = self.model(u_t, i_t).cpu().numpy()
            preds[valid_mask] = batch_preds

        return self._clip_array(preds)


    def get_hyperparameters(self):
        return {
            "embedding_dim": self.embedding_dim,
            "layers":        self.layers,
            "learning_rate": self.learning_rate,
            "batch_size":    self.batch_size,
            "num_epochs":    self.num_epochs,
        }

    def set_hyperparameters(self, params):
        self.embedding_dim = params.get("embedding_dim", self.embedding_dim)
        self.layers        = params.get("layers",        self.layers)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.batch_size    = params.get("batch_size",    self.batch_size)
        self.num_epochs    = params.get("num_epochs",    self.num_epochs)
        self.is_fitted = False