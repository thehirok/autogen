import numpy as np
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from peft import LoraConfig, get_peft_model, TaskType
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class PEFTNCFNetwork(nn.Module):
    """NCF network with named MLP layers for LoRA targeting."""

    def __init__(self, n_users, n_items, embedding_dim, layers):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Named linear layers so LoRA can target them
        self.mlp_layers = nn.ModuleList()
        self.mlp_activations = nn.ModuleList()
        in_dim = embedding_dim * 2
        for i, out_dim in enumerate(layers):
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
            self.mlp_activations.append(nn.ReLU())
            in_dim = out_dim
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        for linear, activation in zip(self.mlp_layers, self.mlp_activations):
            x = activation(linear(x))
        return self.output_layer(x).squeeze(1)

    def get_mlp_state_dict(self):
        """Extract only MLP weights (excluding embeddings)."""
        state = {}
        for name, param in self.named_parameters():
            if not name.startswith(('user_emb', 'item_emb')):
                state[name] = param.data.clone()
        return state

    def load_mlp_state_dict(self, state_dict):
        """Load only MLP weights, skipping embeddings."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)


class PEFTNCFRecommender(BaseRecommender):
    """
    NCF with LoRA-based Parameter-Efficient Fine-Tuning.

    Two-phase approach:
    1. Pre-train: Train full NCF on a source dataset (large)
    2. Adapt: Freeze base MLP weights, attach LoRA adapters to linear
       layers, and fine-tune only the adapters + new embeddings on the
       target dataset.

    This tests whether transfer learning via PEFT improves sample
    efficiency for neural collaborative filtering.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of user/item embeddings.
    layers : list of int
        Hidden layer sizes for the MLP.
    lora_r : int
        LoRA rank (number of low-rank dimensions).
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability for LoRA layers.
    pretrain_epochs : int
        Number of epochs for pre-training on the source dataset.
    adapt_epochs : int
        Number of epochs for adaptation on the target dataset.
    learning_rate : float
        Learning rate for both pre-training and adaptation.
    batch_size : int
        Mini-batch size for training.
    random_state : int
        Random seed for reproducibility.
    pretrained_path : str or None
        Path to save/load pre-trained MLP weights. If None, uses
        default path from config.
    source_dataset : str or None
        Name of the source dataset for pre-training. If None, uses
        config default.
    """

    def __init__(self, embedding_dim=32, layers=None,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 pretrain_epochs=15, adapt_epochs=5,
                 learning_rate=0.001, batch_size=256,
                 random_state=42, pretrained_path=None,
                 source_dataset=None):
        super().__init__("PEFT_NCF")
        self.embedding_dim    = embedding_dim
        self.layers           = layers or [64, 32, 16]
        self.lora_r           = lora_r
        self.lora_alpha       = lora_alpha
        self.lora_dropout     = lora_dropout
        self.pretrain_epochs  = pretrain_epochs
        self.adapt_epochs     = adapt_epochs
        self.learning_rate    = learning_rate
        self.batch_size       = batch_size
        self.random_state     = random_state
        self.source_dataset   = source_dataset
        self.model            = None
        self.global_mean      = None
        self._pretrained_mlp  = None  # cached pre-trained MLP state dict
        self.device           = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Resolve pre-trained model path
        if pretrained_path is None:
            try:
                from config import MODELS_DIR
                self.pretrained_path = os.path.join(
                    MODELS_DIR, "pretrained_ncf_mlp.pt")
            except ImportError:
                self.pretrained_path = "models/pretrained_ncf_mlp.pt"
        else:
            self.pretrained_path = pretrained_path

    # ------------------------------------------------------------------
    # Pre-training
    # ------------------------------------------------------------------

    def _load_source_dataset(self):
        """Load the source dataset for pre-training."""
        from config import DATASETS, PEFT_SOURCE_DATASET
        source_name = self.source_dataset or PEFT_SOURCE_DATASET
        if source_name not in DATASETS:
            raise ValueError(
                f"Source dataset '{source_name}' not in DATASETS config. "
                f"Available: {list(DATASETS.keys())}")
        import pandas as pd
        path = DATASETS[source_name]['path']
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Source dataset file not found: {path}")
        return pd.read_parquet(path)

    def pretrain(self, source_data=None):
        """
        Pre-train NCF on a large source dataset.

        Parameters
        ----------
        source_data : pd.DataFrame or None
            If provided, use this data directly. Otherwise, load from
            the configured source dataset.

        Returns
        -------
        dict : Pre-trained MLP state dict
        """
        if source_data is None:
            source_data = self._load_source_dataset()

        logger.info(f"Pre-training NCF on {len(source_data)} ratings...")
        torch.manual_seed(self.random_state)

        # Build a temporary full model for pre-training
        unique_users = source_data['user_id'].unique()
        unique_items = source_data['item_id'].unique()
        user_map = {u: i for i, u in enumerate(unique_users)}
        item_map = {it: i for i, it in enumerate(unique_items)}

        user_t   = torch.tensor(
            [user_map[u] for u in source_data['user_id']], dtype=torch.long)
        item_t   = torch.tensor(
            [item_map[it] for it in source_data['item_id']], dtype=torch.long)
        rating_t = torch.tensor(
            source_data['rating'].to_numpy(), dtype=torch.float32)

        dataset    = TensorDataset(user_t, item_t, rating_t)
        batch_size = min(self.batch_size, len(dataset))
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = PEFTNCFNetwork(
            len(user_map), len(item_map),
            self.embedding_dim, self.layers
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            n_batches  = 0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(u_batch, i_batch)
                loss  = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Pre-train epoch {epoch+1}/{self.pretrain_epochs}, "
                    f"loss: {avg_loss:.4f}")

        # Extract and cache MLP weights
        self._pretrained_mlp = model.get_mlp_state_dict()

        # Save to disk
        os.makedirs(os.path.dirname(self.pretrained_path) or '.', exist_ok=True)
        torch.save(self._pretrained_mlp, self.pretrained_path)
        logger.info(f"Pre-trained MLP weights saved to {self.pretrained_path}")

        return self._pretrained_mlp

    def _ensure_pretrained(self):
        """Load or create pre-trained MLP weights."""
        if self._pretrained_mlp is not None:
            return

        if os.path.exists(self.pretrained_path):
            logger.info(
                f"Loading pre-trained MLP from {self.pretrained_path}")
            self._pretrained_mlp = torch.load(
                self.pretrained_path, map_location=self.device,
                weights_only=True)
        else:
            logger.info("No pre-trained model found. Pre-training now...")
            self.pretrain()

    # ------------------------------------------------------------------
    # LoRA adaptation (main fit)
    # ------------------------------------------------------------------

    def fit(self, train_data):
        """
        Adapt the pre-trained NCF to a target dataset using LoRA.

        1. Load pre-trained MLP weights
        2. Create new model with target dataset's vocabulary
        3. Initialize MLP from pre-trained weights
        4. Apply LoRA to MLP linear layers
        5. Fine-tune only LoRA parameters + embeddings
        """
        self._ensure_pretrained()
        torch.manual_seed(self.random_state)

        self.train_data  = train_data.copy()
        self.global_mean = train_data['rating'].mean()
        processed = self._preprocess_data(train_data)
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        user_t   = torch.tensor(
            processed['user_idx'].to_numpy(), dtype=torch.long)
        item_t   = torch.tensor(
            processed['item_idx'].to_numpy(), dtype=torch.long)
        rating_t = torch.tensor(
            processed['rating'].to_numpy(), dtype=torch.float32)

        dataset    = TensorDataset(user_t, item_t, rating_t)
        batch_size = min(self.batch_size, len(dataset))
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build fresh model with target vocabulary
        base_model = PEFTNCFNetwork(
            n_users, n_items, self.embedding_dim, self.layers
        ).to(self.device)

        # Load pre-trained MLP weights (embeddings stay random — new vocab)
        base_model.load_mlp_state_dict(self._pretrained_mlp)

        # Identify target modules for LoRA (all nn.Linear in the MLP)
        target_modules = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)

        logger.info(f"Applying LoRA (r={self.lora_r}) to: {target_modules}")

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)

        # Log parameter counts
        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(
            f"PEFT-NCF: {trainable:,} trainable / {total:,} total params "
            f"({100*trainable/total:.1f}%)")

        # Fine-tune only LoRA params + embeddings
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.adapt_epochs):
            total_loss = 0.0
            n_batches  = 0
            for u_batch, i_batch, r_batch in loader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(u_batch, i_batch)
                loss  = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1
            avg_loss = total_loss / max(n_batches, 1)
            logger.debug(
                f"  Adapt epoch {epoch+1}/{self.adapt_epochs}, "
                f"loss: {avg_loss:.4f}")

        self.model.eval()
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction (identical interface to NCF)
    # ------------------------------------------------------------------

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        u = torch.tensor(
            [self.user_mapping[user_id]], dtype=torch.long).to(self.device)
        i = torch.tensor(
            [self.item_mapping[item_id]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            pred = self.model(u, i).item()
        return self._clip(pred)

    def predict_all_items(self, user_id):
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not in training data.")
        u_idx = self.user_mapping[user_id]
        n_items = len(self.item_mapping)
        u_t = torch.full(
            (n_items,), u_idx, dtype=torch.long, device=self.device)
        i_t = torch.arange(n_items, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.model(u_t, i_t).cpu().numpy()
        return self._clip_array(scores)

    def predict_batch(self, user_ids, item_ids):
        """Vectorized batch prediction — single forward pass."""
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

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------

    def get_hyperparameters(self):
        return {
            "embedding_dim":    self.embedding_dim,
            "layers":           self.layers,
            "lora_r":           self.lora_r,
            "lora_alpha":       self.lora_alpha,
            "lora_dropout":     self.lora_dropout,
            "pretrain_epochs":  self.pretrain_epochs,
            "adapt_epochs":     self.adapt_epochs,
            "learning_rate":    self.learning_rate,
            "batch_size":       self.batch_size,
        }

    def set_hyperparameters(self, params):
        self.embedding_dim   = params.get("embedding_dim",   self.embedding_dim)
        self.layers          = params.get("layers",          self.layers)
        self.lora_r          = params.get("lora_r",          self.lora_r)
        self.lora_alpha      = params.get("lora_alpha",      self.lora_alpha)
        self.lora_dropout    = params.get("lora_dropout",    self.lora_dropout)
        self.pretrain_epochs = params.get("pretrain_epochs", self.pretrain_epochs)
        self.adapt_epochs    = params.get("adapt_epochs",    self.adapt_epochs)
        self.learning_rate   = params.get("learning_rate",   self.learning_rate)
        self.batch_size      = params.get("batch_size",      self.batch_size)
        self.is_fitted = False

    def get_peft_param_summary(self):
        """Return a summary of trainable vs total parameters."""
        if self.model is None:
            return {"status": "model not built"}
        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return {
            "trainable_params": trainable,
            "total_params":     total,
            "trainable_pct":    100 * trainable / total if total > 0 else 0,
            "frozen_params":    total - trainable,
        }
