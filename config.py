import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_META_DIR      = os.path.join(BASE_DIR, "data", "meta")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")
PAPER_FIGURES_DIR  = os.path.join(BASE_DIR, "paper", "figures")
PAPER_TABLES_DIR   = os.path.join(BASE_DIR, "paper", "tables")

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Evaluation seeds for statistical robustness (for RQ4) ---
N_EVAL_SEEDS = 5
EVAL_SEEDS = [RANDOM_SEED + i for i in range(N_EVAL_SEEDS)]


# --- Subsampling sizes (for RQ1 and RQ2) ---
SAMPLE_SIZES = [250, 500, 1000, 2000, 5000, 10000]
N_SUBSAMPLE_TRIALS = 10  # number of random draws per sample size

# --- Stability threshold ---
STABILITY_TOLERANCE = 0.05  # feature must stay within ±5% of full-data value

# --- Algorithms to evaluate ---
ALGORITHMS = ["SVD", "KNN", "NMF", "NCF", "SlopeOne", "SVDpp"]

# --- Evaluation ---
TEST_RATIO  = 0.2
K_VALUES    = [5, 10, 20]

# --- Meta-model architectures to compare ---
META_MODELS = ["RandomForest", "XGBoost", "MLP"]

# --- Dataset registry ---
# Add each dataset here as you download it.
# min_ratings: datasets below this are skipped for subsampling experiments.
DATASETS = {
    "movielens_100k": {
        "path": os.path.join(DATA_PROCESSED_DIR, "movielens_100k.parquet"),
        "min_ratings": 1000,
    },
    "movielens_1m": {
        "path": os.path.join(DATA_PROCESSED_DIR, "movielens_1m.parquet"),
        "min_ratings": 10000,
    },
    "movielens_10m": {
        "path": os.path.join(DATA_PROCESSED_DIR, "movielens_10m.parquet"),
        "min_ratings": 10000,
    },
    "movielens_20m": {
        "path": os.path.join(DATA_PROCESSED_DIR, "movielens_20m.parquet"),
        "min_ratings": 10000,
    },
    "amazon_books": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_books.parquet"),
        "min_ratings": 10000,
    },
    "amazon_movies": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_movies.parquet"),
        "min_ratings": 10000,
    },
    "amazon_electronics": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_electronics.parquet"),
        "min_ratings": 10000,
    },
    "amazon_CDs_vinyl": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_CDs_vinyl.parquet"),
        "min_ratings": 10000,
    },
    "amazon_clothing": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_clothing.parquet"),
        "min_ratings": 10000,
    },
    "amazon_home_kitchen": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_home_kitchen.parquet"),
        "min_ratings": 10000,
    },
    "amazon_sports": {
        "path": os.path.join(DATA_PROCESSED_DIR, "amazon_sports.parquet"),
        "min_ratings": 10000,
    },
    "filmtrust": {
        "path": os.path.join(DATA_PROCESSED_DIR, "filmtrust.parquet"),
        "min_ratings": 1000,
    },
    "jester": {
        "path": os.path.join(DATA_PROCESSED_DIR, "jester.parquet"),
        "min_ratings": 10000,
    },
}

# --- Figure settings (ACM two-column format) ---
FIGURE_DPI = 300
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 9
ONE_COL_WIDTH  = 3.33  # inches
TWO_COL_WIDTH  = 7.00  # inches

# --- Dataset filtering (set by run_pipeline.py) ---
_env_datasets = os.environ.get('AUTORECYS_DATASETS')
if _env_datasets:
    _selected = set(_env_datasets.split(','))
    DATASETS = {k: v for k, v in DATASETS.items() if k in _selected}