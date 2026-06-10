"""Comprehensive bug check for the entire project."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

errors = []

# 1. Import checks
print("=== Import checks ===")
try:
    from config import DATASETS, ALGORITHMS, SAMPLE_SIZES, EVAL_SEEDS, META_MODELS
    print("  config: OK")
except Exception as e:
    errors.append(f"config import: {e}")
    print(f"  config: FAIL - {e}")

try:
    from src import EarlySelector, DatasetAnalyzer, Subsampler, MetaModel, StabilityAnalyzer
    print("  src modules: OK")
except Exception as e:
    errors.append(f"src import: {e}")
    print(f"  src modules: FAIL - {e}")

try:
    from src import RECOMMENDER_MAP
    print(f"  recommenders ({len(RECOMMENDER_MAP)}): OK")
except Exception as e:
    errors.append(f"recommenders import: {e}")
    print(f"  recommenders: FAIL - {e}")

# 2. Check all result files
print("\n=== Result file integrity ===")

perf = pd.read_csv("results/algorithm_performance.csv")
n_ds = perf["dataset"].nunique()
n_algo = perf["algorithm"].nunique()
print(f"  algorithm_performance: {n_ds} datasets x {n_algo} algos = {len(perf)} rows")
missing_ndcg = perf[perf["ndcg@10"].isna()]
if len(missing_ndcg) > 0:
    for _, r in missing_ndcg.iterrows():
        msg = f"NaN ndcg@10: {r['dataset']}/{r['algorithm']}"
        if pd.notna(r.get("error")):
            msg += f" (error: {r['error']})"
        print(f"    WARNING: {msg}")
        errors.append(msg)

meta = pd.read_csv("data/meta/full_features.csv")
feat_cols = [c for c in meta.columns if c not in ("dataset", "best_algorithm")]
print(f"  meta-dataset: {len(meta)} datasets, {len(feat_cols)} features")
nan_cols = [c for c in feat_cols if meta[c].isna().any()]
if nan_cols:
    print(f"    WARNING: NaN in features: {nan_cols}")
    errors.append(f"NaN features: {nan_cols}")

e2e = pd.read_csv("results/end_to_end_quality.csv")
e2e_ds = e2e["dataset"].nunique()
print(f"  end_to_end_quality: {e2e_ds} datasets, {len(e2e)} rows")
if e2e_ds != n_ds:
    msg = f"e2e has {e2e_ds} datasets, expected {n_ds}"
    print(f"    WARNING: {msg}")
    errors.append(msg)

sig = pd.read_csv("results/significance_tests.csv")
print(f"  significance_tests: {len(sig)} comparisons")

abl = pd.read_csv("results/ablation_meta_models.csv")
abl_models = sorted(abl["model_type"].unique())
print(f"  ablation: models={abl_models}")

stab = pd.read_csv("results/stability_curves.csv")
stab_ds = stab["dataset"].nunique()
print(f"  stability_curves: {stab_ds} datasets, {len(stab)} rows")

sel = pd.read_csv("results/selection_accuracy_by_n.csv")
sel_ds = sel["dataset"].nunique()
print(f"  selection_accuracy: {sel_ds} datasets, {len(sel)} rows")

# 3. Check figures exist
print("\n=== Figure/Table files ===")
expected_figs = [
    "paper/figures/fig1_stability_curves.pdf",
    "paper/figures/fig2_selection_accuracy.pdf",
    "paper/figures/fig3_shap_ess.pdf",
    "paper/figures/fig4_end_to_end.pdf",
]
expected_tabs = [
    "paper/tables/tab_algorithm_performance.tex",
    "paper/tables/tab_selection_accuracy.tex",
    "paper/tables/tab_feature_importance.tex",
    "paper/tables/tab_significance.tex",
    "paper/tables/tab_ablation.tex",
    "paper/tables/tab_cost.tex",
]
for f in expected_figs + expected_tabs:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = f"OK ({size:,} bytes)" if exists else "MISSING!"
    print(f"  {f}: {status}")
    if not exists:
        errors.append(f"Missing file: {f}")

# 4. Check latex/ copies match paper/
print("\n=== LaTeX folder sync ===")
for f in expected_figs + expected_tabs:
    latex_f = f.replace("paper/", "latex/")
    if os.path.exists(f) and os.path.exists(latex_f):
        s1 = os.path.getsize(f)
        s2 = os.path.getsize(latex_f)
        if s1 == s2:
            print(f"  {os.path.basename(f)}: synced")
        else:
            msg = f"{os.path.basename(f)}: SIZE MISMATCH paper={s1} vs latex={s2}"
            print(f"  {msg}")
            errors.append(msg)
    elif os.path.exists(f) and not os.path.exists(latex_f):
        msg = f"{latex_f}: MISSING in latex/"
        print(f"  {msg}")
        errors.append(msg)

# 5. Check LaTeX references
print("\n=== LaTeX reference checks ===")
with open("latex/main.tex", "r") as f:
    tex = f.read()

# Check all \input files exist
import re
inputs = re.findall(r"\\input\{([^}]+)\}", tex)
for inp in inputs:
    path = os.path.join("latex", inp)
    if not path.endswith(".tex"):
        path += ".tex"
    exists = os.path.exists(path)
    print(f"  \\input{{{inp}}}: {'OK' if exists else 'MISSING!'}")
    if not exists:
        errors.append(f"Missing LaTeX input: {path}")

# Check all figure files exist
figs = re.findall(r"\\includegraphics.*?\{([^}]+)\}", tex)
for fig in figs:
    path = os.path.join("latex", fig)
    exists = os.path.exists(path)
    print(f"  \\includegraphics{{{fig}}}: {'OK' if exists else 'MISSING!'}")
    if not exists:
        errors.append(f"Missing figure: {path}")

# Check for undefined labels/refs
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
undefined = refs - labels
if undefined:
    print(f"  WARNING: Undefined refs: {undefined}")
    errors.append(f"Undefined refs: {undefined}")
else:
    print(f"  Labels/refs: {len(labels)} labels, {len(refs)} refs, all resolved")

# 6. Check bibliography
print("\n=== Bibliography check ===")
cites = set(re.findall(r"\\cite\{([^}]+)\}", tex))
all_citekeys = set()
for c in cites:
    for key in c.split(","):
        all_citekeys.add(key.strip())

bib_path = "latex/reference.bib"
if os.path.exists(bib_path):
    with open(bib_path, "r", encoding="utf-8") as f:
        bib = f.read()
    bib_keys = set(re.findall(r"@\w+\{(\w+)", bib))
    missing_bib = all_citekeys - bib_keys
    if missing_bib:
        print(f"  WARNING: Citations not in .bib: {missing_bib}")
        errors.append(f"Missing bib entries: {missing_bib}")
    else:
        print(f"  All {len(all_citekeys)} citations found in reference.bib")
else:
    print("  WARNING: reference.bib not found!")
    errors.append("Missing reference.bib")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"FOUND {len(errors)} ISSUE(S):")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
else:
    print("ALL CHECKS PASSED - NO BUGS FOUND!")
print("=" * 60)
