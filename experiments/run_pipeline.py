"""
AutoRecSys — Full Experiment Pipeline
======================================
Runs ALL steps end-to-end on a configurable subset of datasets.

Usage:
    python run_pipeline.py                   # default: 5 small/medium datasets
    python run_pipeline.py --all             # all 13 datasets (hours!)
    python run_pipeline.py --quick           # 2 tiny datasets (~30 min)

Output:
    results/*.csv           — raw experimental results
    data/meta/*.csv         — meta-learning dataset
    paper/figures/*.pdf     — 4 publication-ready figures
    paper/tables/*.tex      — LaTeX tables for the paper
"""

import sys
import os
import time
import subprocess
import argparse
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Dataset presets
# -------------------------------------------------------------------
QUICK_DATASETS = [
    'filmtrust', 'movielens_100k',
]

DEFAULT_DATASETS = [
    'filmtrust', 'movielens_100k', 'jester',
    'amazon_clothing', 'amazon_sports',
]

ALL_DATASETS = list(DATASETS.keys())

# -------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------
EXPERIMENTS_DIR = os.path.join(ROOT, 'experiments')
PAPER_DIR = os.path.join(ROOT, 'paper')

STEPS = [
    {
        'name':   '1/9  Run all algorithms',
        'script': os.path.join(EXPERIMENTS_DIR, 'run_all_algorithms.py'),
        'desc':   'Train & evaluate every algorithm on every dataset',
    },
    {
        'name':   '2/9  Build meta-dataset',
        'script': os.path.join(EXPERIMENTS_DIR, 'build_meta_dataset.py'),
        'desc':   'Extract meta-features + oracle labels',
    },
    {
        'name':   '3/9  RQ1 — Feature stability',
        'script': os.path.join(EXPERIMENTS_DIR, 'rq1_feature_stability.py'),
        'desc':   'Stability curves for each meta-feature',
    },
    {
        'name':   '4/9  RQ2 — Selection accuracy',
        'script': os.path.join(EXPERIMENTS_DIR, 'rq2_selection_accuracy.py'),
        'desc':   'Leave-one-out selection accuracy at each sample size',
    },
    {
        'name':   '5/9  RQ3 — Feature importance',
        'script': os.path.join(EXPERIMENTS_DIR, 'rq3_feature_importance.py'),
        'desc':   'SHAP values + ESS scores',
    },
    {
        'name':   '6/9  RQ4 — End-to-end quality',
        'script': os.path.join(EXPERIMENTS_DIR, 'rq4_end_to_end_quality.py'),
        'desc':   'Compare early vs full vs oracle vs random selection',
    },
    {
        'name':   '7/9  Significance tests',
        'script': os.path.join(EXPERIMENTS_DIR, 'significance_tests.py'),
        'desc':   'Wilcoxon + paired t-tests across conditions',
    },
    {
        'name':   '8/9  Meta-model ablation',
        'script': os.path.join(EXPERIMENTS_DIR, 'ablation_meta_models.py'),
        'desc':   'Compare RandomForest vs XGBoost vs MLP',
    },
    {
        'name':   '9/9  Generate figures & tables',
        'script': os.path.join(PAPER_DIR, 'plot_all_figures.py'),
        'desc':   'Publication-ready PDFs + LaTeX tables',
    },
]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def write_temp_config(dataset_names):
    """Temporarily override DATASETS in config to use only the selected subset."""
    selected = {k: v for k, v in DATASETS.items() if k in dataset_names}
    missing = [k for k in dataset_names if k not in DATASETS]
    if missing:
        logger.warning(f"Unknown datasets (skipped): {missing}")

    # Check which datasets actually have files
    available = {}
    for name, cfg in selected.items():
        if os.path.exists(cfg['path']):
            available[name] = cfg
        else:
            logger.warning(f"[{name}] Data file not found: {cfg['path']} — skipping.")

    if not available:
        logger.error("No datasets available! Check data/processed/ folder.")
        sys.exit(1)

    logger.info(f"Selected {len(available)} datasets: {list(available.keys())}")
    return available


def run_step(step, python_exe):
    """Run a single pipeline step as a subprocess."""
    name = step['name']
    script = step['script']

    print()
    print("=" * 70)
    print(f"  {name}")
    print(f"  {step['desc']}")
    print("=" * 70)

    if not os.path.exists(script):
        logger.error(f"Script not found: {script}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [python_exe, script],
        cwd=ROOT,
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"  FAILED (exit code {result.returncode})  "
                     f"[{elapsed:.0f}s]")
        return False

    logger.info(f"  Completed in {elapsed:.0f}s")
    return True


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    elif m:
        return f"{m}m {s}s"
    return f"{s}s"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run the full AutoRecSys experiment pipeline.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quick', action='store_true',
                       help=f'Run on {len(QUICK_DATASETS)} small datasets (~30 min)')
    group.add_argument('--all', action='store_true',
                       help=f'Run on all {len(ALL_DATASETS)} datasets (hours!)')
    group.add_argument('--datasets', nargs='+',
                       help='Specific dataset names to use')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Resume from step N (1-9)')
    args = parser.parse_args()

    # Determine dataset list
    if args.quick:
        dataset_names = QUICK_DATASETS
    elif args.all:
        dataset_names = ALL_DATASETS
    elif args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = DEFAULT_DATASETS

    available = write_temp_config(dataset_names)

    # Find Python executable
    python_exe = sys.executable

    print()
    print("=" * 70)
    print("  AutoRecSys -- Full Experiment Pipeline")
    print("=" * 70)
    print(f"  Datasets: {len(available)}")
    print(f"  Steps:    {len(STEPS)}")
    print(f"  Python:   {os.path.basename(python_exe)}")
    print("=" * 70)
    print()
    for name, cfg in available.items():
        print(f"  - {name}")
    print()

    # Temporarily patch config.DATASETS so experiment scripts only use selected datasets
    # We do this by setting an environment variable that config.py can check
    os.environ['AUTORECYS_DATASETS'] = ','.join(available.keys())

    pipeline_start = time.time()
    failed = []

    for i, step in enumerate(STEPS, start=1):
        if i < args.start_from:
            logger.info(f"Skipping step {step['name']} (--start-from={args.start_from})")
            continue
        ok = run_step(step, python_exe)
        if not ok:
            failed.append(step['name'])
            # Steps 7, 8, 9 are independent - continue even if one fails
            if i <= 6:
                logger.error(f"Pipeline stopped at step {i}. "
                             f"Fix the error and re-run with --start-from={i}")
                break

    total = time.time() - pipeline_start

    print()
    print("=" * 70)
    print(f"  Pipeline finished in {format_time(total)}")
    if failed:
        print(f"  [!] Failed steps: {', '.join(failed)}")
    else:
        print("  [OK] All steps completed successfully!")
    print()
    print("  Outputs:")
    print("    results/*.csv            - raw experimental data")
    print("    data/meta/*.csv          - meta-learning dataset")
    print("    paper/figures/*.pdf      - publication-ready figures")
    print("    paper/tables/*.tex       - LaTeX tables")
    print("=" * 70)


if __name__ == '__main__':
    main()
