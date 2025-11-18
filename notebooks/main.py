import os
import pandas as pd
import pickle  # Importado para salvar o modelo
import json  # Importado para salvar o modelo

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from bayesian_search import run_bayesian_search
from grafico import plotar_resultados_optuna

DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
IMAGES_DIR = 'images'

def carregar_arquivos():

    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    movies_path = os.path.join(DATA_DIR, 'movies.csv')

    try:
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)

        # Garantir que as pastas existem
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        return ratings_df, movies_df

    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None, None


def treinando_modelo(ratings_df):
   
    reader = Reader(rating_scale=(0.5, 5.0)) # definir escala de avaliações, como o surprise é uma bb de RECOMENDACOES ele espera NOTAS
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader) # carregar dados do dataframe no formato do surprise
    
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # --- Trabalho 1: SVD ---
    svd_job = {
        'model_class': SVD(random_state=42),
        'model_name': "SVD_bayesian_optuna",
        'param_space': {
            'n_factors': ('int', 50, 150),
            'lr_all':    ('float', 0.001, 0.1, 'log'),
            'reg_all':   ('float', 0.01, 0.2, 'log')
        }
    }

    # --- Trabalho 2: KNN ---
    knn_job = {
        'model_class': KNNBasic(),
        'model_name': "KNN_bayesian_optuna",
        'param_space': {
            'k':                 ('int', 10, 100),
            'sim_options__name': ('categorical', ['cosine', 'pearson', 'msd']),
            'sim_options__user_based': ('categorical', [True, False])
        }
    }

    # Lista de todos os trabalhos que queremos executar
    jobs_to_run = [svd_job, knn_job]



    return data, trainset, testset, jobs_to_run
    

def salvando_modelo(model_name, best_model, results):

    print(f"\n--- Salvando modelo final para {model_name} ---")
    model_filename = f"final_{model_name}.pkl" #nome do modelo final
    model_save_path = os.path.join(MODELS_DIR, model_filename) # caminho p salvar o modelo


    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Modelo salvo com sucesso em: {model_save_path}")


    # Salvando o HISTÓRICO COMPLETO (DataFrame) como CSV
    try:
        history_df = results['tuning_results']
        csv_filename = f"optuna_history_{model_name}.csv"
        csv_save_path = os.path.join(RESULTS_DIR, csv_filename)
        history_df.to_csv(csv_save_path, index=False)
        print(f"✓ Histórico de 'trials' (CSV) salvo em: {csv_save_path}")
    except Exception as e:
        print(f"Erro ao salvar histórico CSV: {e}")


    # Salvando o SUMÁRIO (Métricas e Params) como JSON
    try:
        summary_results = {
            'model_name': results['model_name'],
            'metrics': results['metrics'],
            'best_params': results['best_params']
        }
        json_filename = f"optuna_summary_{model_name}.json"
        json_save_path = os.path.join(RESULTS_DIR, json_filename)
        
        with open(json_save_path, 'w') as f:
            json.dump(summary_results, f, indent=4)
        print(f"✓ Sumário (JSON) salvo em: {json_save_path}")
    except Exception as e:
        print(f"Erro ao salvar sumário JSON: {e}")


if __name__ == "__main__":
    

    ratings_df, movies_df = carregar_arquivos()
    data, trainset, testset, jobs_to_run = treinando_modelo(ratings_df)

    n_trials = 50  # Número de tentativas para cada modelo



    print("Iniciando script de otimização...")
    for job in jobs_to_run:
        model_name = job['model_name']
        
        print(f"\n{'='*70}")
        print(f"--- INICIANDO TRABALHO: {model_name} ---")
        print(f"{'='*70}")
        
        # 4A. EXECUTAR A OTIMIZAÇÃO
        results, best_model = run_bayesian_search(
            model_class=job['model_class'],
            param_space=job['param_space'],
            data=data,       # 'data' completo para o cross-validation (CV)
            trainset=trainset,  
            testset=testset,     
            model_name=model_name,
            n_trials=n_trials      # Número de tentativas para cada modelo
        )

        salvando_modelo(model_name, best_model, results)

    
    # Plotar os resultados
    try:
        df_para_plotar = results['tuning_results']
        plotar_resultados_optuna(df_para_plotar, model_name)
    except Exception as e:
        print(f"Não foi possível gerar os gráficos para {model_name}: {e}")

        print(testset)
        print(f"\n--- TRABALHO CONCLUÍDO: {model_name} ---")

    print(f"\n{'='*70}")
    print("✓✓✓ Todos os trabalhos de otimização foram concluídos.")
    print(f"{'='*70}")



