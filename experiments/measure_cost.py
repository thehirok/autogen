"""Measure computational cost of early vs full algorithm evaluation."""
import time, pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS, ALGORITHMS
from src.dataset_analyzer import DatasetAnalyzer
from src.subsampler import Subsampler

analyzer = DatasetAnalyzer()
subsampler = Subsampler(random_state=42)

# Load algorithm performance to get per-dataset training times if available
perf_path = os.path.join('results', 'algorithm_performance.csv')
perf_df = pd.read_csv(perf_path) if os.path.exists(perf_path) else None

results = []
for ds_name, ds_cfg in DATASETS.items():
    if not os.path.exists(ds_cfg['path']):
        continue
    data = pd.read_parquet(ds_cfg['path'])
    data = data[['user_id', 'item_id', 'rating']].dropna()
    n_ratings = len(data)

    # Time: subsample(250) + feature extraction (= early selection cost)
    t0 = time.time()
    sample = subsampler.subsample(data, 250, trial=0)
    analyzer.load_dataset(sample)
    feats = analyzer.extract_features()
    early_time = time.time() - t0

    # Time: full-data feature extraction
    t0 = time.time()
    analyzer.load_dataset(data)
    feats_full = analyzer.extract_features()
    full_feat_time = time.time() - t0

    # Get total training time for all algorithms (from perf data if available)
    algo_time = None
    if perf_df is not None and 'train_time_sec' in perf_df.columns:
        ds_perf = perf_df[perf_df['dataset'] == ds_name]
        algo_time = ds_perf['train_time_sec'].sum()

    results.append({
        'dataset': ds_name,
        'n_ratings': n_ratings,
        'early_select_sec': round(early_time, 3),
        'full_feat_sec': round(full_feat_time, 3),
        'all_algo_train_sec': round(algo_time, 1) if algo_time else None,
    })
    print(f"{ds_name:>25s}: {n_ratings:>10,} ratings | "
          f"early={early_time:.3f}s | full_feat={full_feat_time:.3f}s"
          + (f" | algo_train={algo_time:.1f}s" if algo_time else ""))

df = pd.DataFrame(results)
out = os.path.join('results', 'computational_cost.csv')
df.to_csv(out, index=False)
print(f"\nSaved -> {out}")
print(f"\nMean early selection time: {df['early_select_sec'].mean():.3f}s")
print(f"Mean full feature time:   {df['full_feat_sec'].mean():.3f}s")
print(f"Max early selection time:  {df['early_select_sec'].max():.3f}s")
print(f"Speedup (full_feat/early): {df['full_feat_sec'].mean() / df['early_select_sec'].mean():.1f}x")
