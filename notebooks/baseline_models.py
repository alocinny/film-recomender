import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import json
import pickle

DATA_DIR = 'data'
IMAGES_DIR = 'images'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# carregar dados da EDA

ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
movies_path = os.path.join(DATA_DIR, 'movies.csv')

try: 
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    raise

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# modelos baseline  
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# modelo svd
model_svd = SVD()
model_svd.fit(trainset)

# modelo knn
model_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
model_knn.fit(trainset)

# calcular RMSE e MAE
predictions_svd = model_svd.test(testset)
predictions_knn = model_knn.test(testset)

mae_svd = accuracy.mae(predictions_svd)
rmse_svd = accuracy.rmse(predictions_svd)

rmse_knn = accuracy.rmse(predictions_knn)
mae_knn = accuracy.mae(predictions_knn)

baseline_results = {
    'SVD': {'RMSE': rmse_svd, 'MAE': mae_svd},
    'KNN': {'RMSE': rmse_knn, 'MAE': mae_knn}
}

with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
    json.dump(baseline_results, f, indent=4)

# salvar modelos treinados
svd_path = os.path.join(MODELS_DIR, 'svd_model.pkl')
knn_path = os.path.join(MODELS_DIR, 'knn_model.pkl')

with open(svd_path, 'wb') as f:
    pickle.dump(model_svd, f)

with open(knn_path, 'wb') as f:
    pickle.dump(model_knn, f)