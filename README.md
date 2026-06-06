# AutoRecSys — Sample-Efficient Algorithm Selection for Recommender Systems

Research codebase for the paper submitted to ACM RecSys 2026.

## Research question

How few ratings are needed before the right recommendation algorithm
can be reliably predicted from dataset meta-features?

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce all results

See `REPRODUCE.md` for the full step-by-step pipeline.

## Project structure
```
src/                 Core library
  recommenders/      Six CF algorithms (SVD, KNN, NCF, NMF, SVDpp, SlopeOne)
  dataset_analyzer   Meta-feature extraction (28 features)
  subsampler         Stratified subsampling at multiple sizes
  stability_analyzer Feature stability curves + ESS metric
  early_selector     Sample-efficient algorithm selection
  meta_model         RF / XGBoost / MLP meta-learner
  evaluate           RMSE, MAE, Precision@K, Recall@K, NDCG@K

experiments/         One script per research question
data/loaders/        Dataset-specific loaders (MovieLens, Amazon, generic)
results/             CSV outputs from experiments (git-ignored)
paper/               Figure generation → publication-ready PDFs
tests/               Pytest test suite
config.py            All parameters in one place
REPRODUCE.md         Exact commands to replicate every result
```

## Datasets

Minimum 13 datasets required. Download instructions in `REPRODUCE.md`.
All datasets are standardised to parquet with columns:
`user_id`, `item_id`, `rating`.

## Citation

(To be added after acceptance.)