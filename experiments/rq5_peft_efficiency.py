"""
RQ5: Does PEFT Improve Sample Efficiency for Neural Recommenders?
=================================================================

Compares vanilla NCF (trained from scratch) against PEFT-NCF
(LoRA-adapted from pre-trained weights) across multiple sample sizes.

For each dataset and each sample size:
  1. Subsample N ratings from the training split
  2. Train vanilla NCF from scratch on the subsample
  3. Train PEFT-NCF (LoRA adaptation) on the subsample
  4. Evaluate both on the full test split (NDCG@10, RMSE)
  5. Record training time

Output: results/rq5_peft_efficiency.csv
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (DATASETS, RANDOM_SEED, SAMPLE_SIZES, RESULTS_DIR,
                    TEST_RATIO, K_VALUES, PEFT_SOURCE_DATASET)
from src.recommenders.ncf import NCFRecommender
from src.recommenders.peft_ncf import PEFTNCFRecommender
from src.evaluate import Evaluator
from src.subsampler import Subsampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

N_TRIALS = 3  # trials per sample size for variance estimation


def ensure_pretrained(source_name):
    """Pre-train NCF on source dataset if not already done."""
    rec = PEFTNCFRecommender(
        pretrain_epochs=15,
        random_state=RANDOM_SEED,
        source_dataset=source_name,
    )
    rec._ensure_pretrained()
    logger.info("Pre-trained NCF weights are ready.")
    return rec.pretrained_path


def run_comparison(dataset_name, data, pretrained_path):
    """Run NCF vs PEFT-NCF comparison at all sample sizes."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Dataset: {dataset_name} ({len(data)} ratings)")
    logger.info(f"{'='*60}")

    train_data, test_data = train_test_split(
        data, test_size=TEST_RATIO, random_state=RANDOM_SEED)

    evaluator  = Evaluator(random_state=RANDOM_SEED)
    subsampler = Subsampler(random_state=RANDOM_SEED)
    records    = []

    # Also test on full training data
    sizes_to_test = [s for s in SAMPLE_SIZES if s <= len(train_data)]
    sizes_to_test.append(len(train_data))  # full data

    for n in sizes_to_test:
        is_full = (n == len(train_data))
        size_label = "full" if is_full else str(n)
        n_trials = 1 if is_full else N_TRIALS

        for trial in range(n_trials):
            if is_full:
                sample = train_data
            else:
                sample = subsampler.subsample(train_data, n, trial=trial)

            # --- Vanilla NCF ---
            try:
                t0 = time.time()
                ncf = NCFRecommender(
                    embedding_dim=32, layers=[64, 32, 16],
                    num_epochs=10, random_state=RANDOM_SEED + trial)
                ncf.fit(sample)
                ncf_time = time.time() - t0

                metrics_ncf = evaluator.evaluate_recommender(
                    ncf, test_data, k_values=K_VALUES)
                ndcg_ncf = metrics_ncf['ndcg_at_k'].get(10, 0.0)
                rmse_ncf = metrics_ncf['rmse']
            except Exception as e:
                logger.warning(f"NCF failed ({dataset_name}, n={n}): {e}")
                ndcg_ncf, rmse_ncf, ncf_time = np.nan, np.nan, np.nan

            # --- PEFT-NCF ---
            try:
                t0 = time.time()
                peft_ncf = PEFTNCFRecommender(
                    embedding_dim=32, layers=[64, 32, 16],
                    adapt_epochs=5, lora_r=8, lora_alpha=16,
                    random_state=RANDOM_SEED + trial,
                    pretrained_path=pretrained_path)
                peft_ncf.fit(sample)
                peft_time = time.time() - t0

                metrics_peft = evaluator.evaluate_recommender(
                    peft_ncf, test_data, k_values=K_VALUES)
                ndcg_peft = metrics_peft['ndcg_at_k'].get(10, 0.0)
                rmse_peft = metrics_peft['rmse']

                param_summary = peft_ncf.get_peft_param_summary()
                trainable_pct = param_summary.get('trainable_pct', 0)
            except Exception as e:
                logger.warning(f"PEFT-NCF failed ({dataset_name}, n={n}): {e}")
                ndcg_peft, rmse_peft, peft_time = np.nan, np.nan, np.nan
                trainable_pct = np.nan

            records.append({
                'dataset':        dataset_name,
                'sample_size':    size_label,
                'trial':          trial,
                'ncf_ndcg10':     ndcg_ncf,
                'ncf_rmse':       rmse_ncf,
                'ncf_time_s':     ncf_time,
                'peft_ndcg10':    ndcg_peft,
                'peft_rmse':      rmse_peft,
                'peft_time_s':    peft_time,
                'peft_trainable_pct': trainable_pct,
                'ndcg_improvement': (ndcg_peft - ndcg_ncf)
                                     if not (np.isnan(ndcg_peft) or
                                             np.isnan(ndcg_ncf)) else np.nan,
                'speedup':        (ncf_time / peft_time)
                                   if (peft_time and peft_time > 0
                                       and not np.isnan(ncf_time)) else np.nan,
            })

            logger.info(
                f"  n={size_label:>5s} trial={trial}  "
                f"NCF: NDCG={ndcg_ncf:.4f} ({ncf_time:.1f}s)  "
                f"PEFT: NDCG={ndcg_peft:.4f} ({peft_time:.1f}s)  "
                f"Δ={ndcg_peft - ndcg_ncf:+.4f}"
                if not np.isnan(ndcg_ncf) else
                f"  n={size_label:>5s} trial={trial}  FAILED")

    return records


def generate_latex_table(df, output_path):
    """Generate a LaTeX table summarizing PEFT vs NCF results."""
    # Aggregate across trials
    agg = df.groupby(['dataset', 'sample_size']).agg({
        'ncf_ndcg10':     'mean',
        'peft_ndcg10':    'mean',
        'ndcg_improvement': 'mean',
        'ncf_time_s':     'mean',
        'peft_time_s':    'mean',
        'speedup':        'mean',
    }).reset_index()

    # Build summary across all datasets per sample size
    size_agg = df.groupby('sample_size').agg({
        'ncf_ndcg10':     'mean',
        'peft_ndcg10':    'mean',
        'ndcg_improvement': 'mean',
        'speedup':        'mean',
        'peft_trainable_pct': 'mean',
    }).reset_index()

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{NCF vs.\ PEFT-NCF (LoRA) across sample sizes. '
        r'$\Delta$NDCG shows the improvement from PEFT adaptation.}',
        r'\label{tab:rq5}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'$N$ & NCF & PEFT-NCF & $\Delta$NDCG & Trainable\% & Speedup \\',
        r'\midrule',
    ]

    # Sort by numeric sample size
    def sort_key(s):
        try:
            return int(s)
        except ValueError:
            return 999999

    size_agg = size_agg.sort_values(
        'sample_size', key=lambda x: x.map(sort_key))

    for _, row in size_agg.iterrows():
        n_label = row['sample_size']
        if n_label == 'full':
            n_label = 'Full'
        lines.append(
            f"  {n_label} & "
            f"{row['ncf_ndcg10']:.4f} & "
            f"{row['peft_ndcg10']:.4f} & "
            f"{row['ndcg_improvement']:+.4f} & "
            f"{row['peft_trainable_pct']:.1f}\\% & "
            f"{row['speedup']:.1f}$\\times$ \\\\"
        )

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"LaTeX table saved to {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("  RQ5: PEFT Sample Efficiency Comparison")
    logger.info("=" * 60)

    # Step 1: Ensure pre-trained weights exist
    logger.info(f"\nSource dataset for pre-training: {PEFT_SOURCE_DATASET}")
    pretrained_path = ensure_pretrained(PEFT_SOURCE_DATASET)

    # Step 2: Run comparison on each dataset
    all_records = []
    for name, cfg in DATASETS.items():
        path = cfg['path']
        if not os.path.exists(path):
            logger.warning(f"Skipping {name}: file not found at {path}")
            continue
        data = pd.read_parquet(path)
        records = run_comparison(name, data, pretrained_path)
        all_records.extend(records)

    # Step 3: Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, 'rq5_peft_efficiency.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")

    # Step 4: Generate LaTeX table
    latex_dir = os.path.join(ROOT, 'latex', 'tables')
    generate_latex_table(df, os.path.join(latex_dir, 'tab_peft_efficiency.tex'))

    # Also save to paper/tables
    paper_dir = os.path.join(ROOT, 'paper', 'tables')
    os.makedirs(paper_dir, exist_ok=True)
    generate_latex_table(df, os.path.join(paper_dir, 'tab_peft_efficiency.tex'))

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("  RQ5 Summary")
    print("=" * 60)

    avg = df.groupby('sample_size').agg({
        'ndcg_improvement': 'mean',
        'speedup': 'mean',
    })
    print("\nAverage PEFT improvement by sample size:")
    print(avg.to_string())

    overall_improvement = df['ndcg_improvement'].mean()
    overall_speedup     = df['speedup'].mean()
    print(f"\nOverall NDCG improvement: {overall_improvement:+.4f}")
    print(f"Overall adaptation speedup: {overall_speedup:.1f}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
