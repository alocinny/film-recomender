import pandas as pd
import pickle
from surprise import Reader, Dataset, SVD, KNNBaseline
from surprise.model_selection import GridSearchCV
from src import config

def run():

    """Treina e salva o modelo KNN."""

    print("\n--- Treinando KNN ---")

    df = pd.read_csv(config.DATA_RAW / 'ratings.csv')

    reader = Reader(rating_scale=(0.5, 5.0))

    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    print("\n> [1/2] Treinando KNN User-Based...")

    # GridSearch
    param_grid = {
        'k': [40, 80],
        'min_k': [5, 9],
        'sim_options': {'name': ['msd', 'pearson_baseline'], 'user_based': [True]}
    }
    gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
    gs.fit(data)
    
    best_model = gs.best_estimator['rmse']
    best_model.fit(data.build_full_trainset())

    with open(config.MODELS_DIR / 'knn_user_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"KNN salvo. RMSE: {gs.best_score['rmse']:.4f}")

    print("\n> [2/2] Treinando KNN Item-Based...")

    param_grid_item = {
        'k': [40],
        'sim_options': {
            'name': ['pearson_baseline'],
            'user_based': [False],
            'shrinkage':[100]
        }
    }

    gs_item = GridSearchCV(KNNBaseline, param_grid_item, measures=['rmse'], cv=5, n_jobs=-1)
    gs_item.fit(data)
    
    best_item_model = gs_item.best_estimator['rmse']
    best_item_model.fit(data.build_full_trainset())

    item_model_path = config.MODELS_DIR / 'knn_item_model.pkl'
    with open(item_model_path, 'wb') as f:
        pickle.dump(best_item_model, f)
        
    print(f"   KNN Item-Based salvo em {item_model_path}")
    print(f"   Melhor RMSE Item: {gs_item.best_score['rmse']:.4f}")
    
    print("\n--- Todos os modelos KNN foram treinados e salvos! ---")
    
