import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()

DATA_DIR = 'data'
RESULTS_DIR = 'results'
IMAGES_DIR = 'images'

ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
ratings_df = pd.read_csv(ratings_path)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

space = {
    'n_factors': hp.quniform('n_factors', 50, 100, 10),
    'lr_all': hp.loguniform('lr_all', np.log(0.001), np.log(0.1)),
    'reg_all': hp.loguniform('reg_all', np.log(0.01), np.log(0.1))
}

# funcao pra tentar minimizar o rmse
def objective(params):
    params['n_factors'] = int(params['n_factors'])

    algo = SVD(**params)

    cv_results = cross_validate(
        algo, 
        data, 
        measures=['rmse'], 
        cv=3, 
        verbose=False, 
        n_jobs=1
    )

    loss = np.mean(cv_results['test_rmse'])

    return {'loss': loss, 'status': STATUS_OK}

# otimizador
trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

best_rmse = trials.best_trial['result']['loss']
best_params_real = trials.best_trial['misc']['vals']

end_time = time.time()

# processar resultados do trials

results = []
for i, t in enumerate(trials.trials):
    loss = t['result']['loss']
    params = t['misc']['vals']
    n_factors = int(params['n_factors'][0])
    lr_all = params['lr_all'][0]
    reg_all = params['reg_all'][0]

    results.append({
        'iteration': i + 1,
        'rmse': loss,
        'n_factors': n_factors,
        'lr_all': lr_all,
        'reg_all': reg_all
    })

results_df = pd.DataFrame(results)
results_df['best_rmse'] = results_df['rmse'].cummin()

csv_path = os.path.join(RESULTS_DIR, 'bayesian_opt_history.csv')
results_df.to_csv(csv_path, index=False)

# plotar gráficos
sns.set_theme(style='whitegrid', palette='viridis')

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=results_df, 
    x='iteration', 
    y='rmse', 
    label='iteration RMSE',
    marker='o',
    alpha=0.7
)
sns.lineplot(
    data=results_df, 
    x='iteration', 
    y='best_rmse',
    label='best RMSE',
    color='red',
    linewidth=2
)
plt.title('Convergence of Bayesian Optimization (SVD)', fontsize=16)
plt.xlabel('Evaluation', fontsize=12)
plt.ylabel('RMSE (cv=3)', fontsize=12)
plt.legend()
plt.xticks(range(0, 21, 2))
conv_path = os.path.join(IMAGES_DIR, 'bayesian_opt_convergence.png')
plt.savefig(conv_path, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('RMSE vs Hyperparameters', fontsize=18, y=1.03)

sns.scatterplot(ax=axes[0], data=results_df, x='n_factors', y='rmse', hue='iteration',palette='viridis', legend=None)
axes[0].set_title('n_factors', fontsize=14)
axes[0].set_xlabel('Valor', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)

sns.scatterplot(ax=axes[1], data=results_df, x='lr_all', y='rmse', hue='iteration', palette='viridis')
axes[1].set_title('lr_all', fontsize=14)
axes[1].set_xlabel('Log scale', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_xscale('log')
axes[1].legend(title='Iteration', bbox_to_anchor=(1.05, 1), loc=2)

sns.scatterplot(ax=axes[2], data=results_df, x='reg_all', y='rmse', hue='iteration', palette='viridis', legend=None)
axes[2].set_title('reg_all', fontsize=14)
axes[2].set_xlabel('Log scale', fontsize=12)
axes[2].set_ylabel('RMSE', fontsize=12)
axes[2].set_xscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.98])
params_path = os.path.join(IMAGES_DIR, 'plot_bayesian_parameters.png')
plt.savefig(params_path, bbox_inches='tight')
plt.close()