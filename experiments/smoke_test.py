"""Quick end-to-end smoke test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src import EarlySelector
from config import DATASETS

# Load meta-dataset
meta = pd.read_csv("data/meta/full_features.csv")
feat_cols = [c for c in meta.columns if c not in ("dataset", "best_algorithm")]

# Fit selector
sel = EarlySelector(random_state=42)
sel.fit_meta_model(meta[feat_cols], meta["best_algorithm"].values)

# Test on a small dataset
ds = DATASETS["filmtrust"]
data = pd.read_parquet(ds["path"])
data = data[["user_id", "item_id", "rating"]].dropna()

# Full selection
result = sel.select(data)
algo = result["algorithm"]
conf = result["confidence"]
print(f"Full selection: {algo} (confidence: {conf:.3f})")

# Early selection at N=250
early = sel.select_at_n(data, 250, trial=0)
print(f"Early selection (N=250): {early}")

# Verify ranking
ranking = result["ranking"]
probas = result["probabilities"]
print(f"Ranking: {ranking}")
print(f"Probabilities: { {k: round(v, 3) for k, v in probas.items()} }")

print("\nEnd-to-end smoke test: PASSED")
