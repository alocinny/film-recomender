import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import GridSearchCV
import os
import pickle
import time
import ast

start_time = time.time()

DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
IMAGES_DIR = 'images'

# carregar dataset
ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
ratings_df = pd.read_csv(ratings_path)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# grid para svd
param_grid_svd = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.5]
}

# grid para knn
param_grid_knn = {
    'k': [20, 40, 60],
    'sim_options': {
        'name': ['cosine', 'pearson', 'msd'],
        'user_based': [False]
    }
}

# rodar gridsearch svd
gs_svd = GridSearchCV(
    SVD,
    param_grid_svd,
    measures=['rmse', 'mae'],
    cv=3,
    n_jobs=-1,
    joblib_verbose=5
)

gs_svd.fit(data)

results_svd_df = pd.DataFrame.from_dict(gs_svd.cv_results)
svd_csv_path = os.path.join(RESULTS_DIR, 'grid_search_svd.csv')
results_svd_df.to_csv(svd_csv_path, index=False)

# rodar gridsearch knn
gs_knn = GridSearchCV(
    KNNBasic, 
    param_grid_knn, 
    measures=['rmse', 'mae'], 
    cv=3,
    n_jobs=-1,
    joblib_verbose=5
)

gs_knn.fit(data)

results_knn_df = pd.DataFrame.from_dict(gs_knn.cv_results)
knn_csv_path = os.path.join(RESULTS_DIR, 'grid_search_knn.csv')
results_knn_df.to_csv(knn_csv_path, index=False)

# treinar modelos finais com melhores parâmetros
full_trainset = data.build_full_trainset()

best_params_svd = gs_svd.best_params['rmse']
final_model_svd = SVD(
    n_factors=best_params_svd['n_factors'], 
    lr_all=best_params_svd['lr_all'], 
    reg_all=best_params_svd['reg_all']
)
final_model_svd.fit(full_trainset)

best_params_knn = gs_knn.best_params['rmse']
final_model_knn = KNNBasic(
    k=best_params_knn['k'], 
    sim_options=best_params_knn['sim_options']
)
final_model_knn.fit(full_trainset)

# salvar modelos
svd_best_path = os.path.join(MODELS_DIR, 'final_svd_model.pkl')
knn_best_path = os.path.join(MODELS_DIR, 'final_knn_model.pkl')

with open(svd_best_path, 'wb') as f:
    pickle.dump(final_model_svd, f)

with open(knn_best_path, 'wb') as f:
    pickle.dump(final_model_knn, f)

end_time = time.time()
elapsed_time = end_time - start_time

# plotar graficos
sns.set_theme(style='whitegrid', palette='viridis')

svd_results_df = pd.read_csv(svd_csv_path)
knn_results_df = pd.read_csv(knn_csv_path)

reg_values = svd_results_df['param_reg_all'].unique()

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(reg_values),
    figsize=(16, 7),
    sharey=True
)
fig.suptitle('SVD Performance (RMSE) by Hyperparameter', fontsize=18, y=1.03)

for i, reg_values in enumerate(reg_values):
    subset_df = svd_results_df[svd_results_df['param_reg_all'] == reg_values]
    
    pivot = subset_df.pivot_table(
        index='param_n_factors', 
        columns='param_lr_all', 
        values='mean_test_rmse'
    )

    sns.heatmap(
        pivot, 
        ax=axes[i], 
        annot=True, 
        fmt='.4f', 
        cmap='viridis', 
        linewidths=.5
    )

    axes[i].set_title(f'Regularization: {reg_values}')
    axes[i].set_xlabel('Learning Rate', fontsize=12)
    axes[i].set_ylabel('Number of Factors', fontsize=12)

plt.tight_layout(rect=[0,0,1,0.98])
svd_plot_path = os.path.join(IMAGES_DIR, 'heatmap_grid_search_svd.png')
plt.savefig(svd_plot_path, bbox_inches='tight')
plt.close()

knn_results_df['similarity'] = knn_results_df['param_sim_options'].apply(
    lambda x: ast.literal_eval(x)['name']
)

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=knn_results_df, 
    x='param_k',
    y='mean_test_rmse',
    hue='similarity',
    style='similarity',
    markers=True,
    markersize=10,
    linewidth=2.5
)

plt.title('KNN Performance (RMSE) by Hyperparameter', fontsize=18)
plt.xlabel('Number of Neighbors', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.xticks(knn_results_df['param_k'].unique())
plt.legend(title='Similarity Measure', fontsize=11, title_fontsize=13)
knn_plot_path = os.path.join(IMAGES_DIR, 'lineplot_grid_search_knn.png')
plt.savefig(knn_plot_path, bbox_inches='tight')
plt.close()