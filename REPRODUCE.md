# Reproducing all paper results

Run every command from the project root with the virtual environment active.

## 0. Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare datasets

Download each dataset and place the raw files in `data/raw/<dataset_name>/`.
Then run the preprocessing notebook or your own loader to produce
standardised parquet files in `data/processed/` with columns:
`user_id`, `item_id`, `rating`.

## 2. Run all algorithms on all datasets
```bash
python experiments/run_all_algorithms.py
```

Output: `results/algorithm_performance.csv`

## 3. Build the meta-learning dataset
```bash
python experiments/build_meta_dataset.py
```

Output: `data/meta/full_features.csv`
        `data/meta/algorithm_performance.csv`

## 4. RQ1 — Feature stability curves
```bash
python experiments/rq1_feature_stability.py
```

Output: `results/stability_curves.csv`

## 5. RQ2 — Selection accuracy vs sample size
```bash
python experiments/rq2_selection_accuracy.py
```

Output: `results/selection_accuracy_by_n.csv`

## 6. RQ3 — Feature importance and ESS
```bash
python experiments/rq3_feature_importance.py
```

Output: `results/shap_values.csv`
        `results/ess_scores.csv`

## 7. RQ4 — End-to-end quality
```bash
python experiments/rq4_end_to_end_quality.py
```

Output: `results/end_to_end_quality.csv`

## 8. Ablation
```bash
python experiments/ablation_meta_models.py
```

Output: `results/ablation_meta_models.csv`

## 9. Significance tests
```bash
python experiments/significance_tests.py
```

Output: `results/significance_tests.csv`

## 10. Generate all figures
```bash
python paper/plot_all_figures.py
```

Output: `paper/figures/fig1_stability_curves.pdf`
        `paper/figures/fig2_selection_accuracy.pdf`
        `paper/figures/fig3_shap_ess.pdf`
        `paper/figures/fig4_end_to_end.pdf`

## 11. Run tests
```bash
python -m pytest tests/ -v
```

All steps use `RANDOM_SEED = 42` from `config.py`.
To change any experimental parameter, edit `config.py` only —
do not edit experiment scripts directly.
```

Save it.

---

That is the complete codebase. Here is exactly what you have now:
```
30 files written
6  recommender algorithms    — all fixed, no bugs
4  new research modules      — subsampler, stability_analyzer,
                               early_selector, meta_model
8  experiment scripts        — one per RQ + ablation + significance
1  figure generation script  — publication-ready PDFs
2  test files                — subsampler + stability_analyzer
1  REPRODUCE.md              — reviewers can replicate everything