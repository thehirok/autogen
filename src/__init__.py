from .dataset_analyzer import DatasetAnalyzer
from .evaluate         import Evaluator
from .meta_model       import MetaModel
from .subsampler       import Subsampler
from .stability_analyzer import StabilityAnalyzer
from .early_selector   import EarlySelector
from .recommenders     import (SVDRecommender, KNNRecommender,
                               NCFRecommender, NMFRecommender,
                               SVDppRecommender, SlopeOneRecommender)

RECOMMENDER_MAP = {
    'SVD':      SVDRecommender,
    'KNN':      KNNRecommender,
    'NCF':      NCFRecommender,
    'NMF':      NMFRecommender,
    'SVDpp':    SVDppRecommender,
    'SlopeOne': SlopeOneRecommender,
}