import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.model_selection import train_test_split
from src.recommenders.knn import KNNRecommender
from src.evaluate import Evaluator

data = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'movielens_100k.parquet'))
data = data[['user_id','item_id','rating']].dropna()
train, test = train_test_split(data, test_size=0.2, random_state=42)
print(f"Train={len(train)}, Test={len(test)}")

rec = KNNRecommender()
rec.fit(train)
print("KNN fitted. Testing predict_batch...")

preds = rec.predict_batch(test['user_id'].values[:100], test['item_id'].values[:100])
print(f"predict_batch OK: {preds[:5]}")

print("Testing evaluate...")
ev = Evaluator()
metrics = ev.evaluate_recommender(rec, test, k_values=[10])
print(f"RMSE={metrics['rmse']:.4f}")
