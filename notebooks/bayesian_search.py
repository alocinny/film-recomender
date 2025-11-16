import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import os
import pickle  # Importado para salvar o modelo
import json    # Importado para salvar o sumário

# Imports do Surprise
from surprise import accuracy, Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate, train_test_split

# --- 1. Definição das Pastas ---

DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
IMAGES_DIR = 'images'

# Desabilitar logs de cada 'trial' do Optuna para limpar o console
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 2. Função Principal de Otimização ---
def run_bayesian_search(model_class, param_space, data, trainset, testset,
                        model_name, n_trials=50):
    """
    Executa uma otimização Bayesiana (Optuna) para um modelo Surprise.
    
    Retorna:
    - final_results (dict): Dicionário com métricas, parâmetros e histórico.
    - final_model (object): O objeto do modelo treinado no trainset completo.
    """
    
    print(f"\n[Optuna] Iniciando Otimização Bayesiana para {model_name} ({n_trials} tentativas)...")

    # --- 1. Definir a "Função Objetivo" ---
    def objective(trial):
        params = {}
        sim_options = {} # Especial para KNN

        for name, (type, *args) in param_space.items():
            is_sim_option = False
            if name.startswith('sim_options__'):
                param_name = name.split('__')[1]
                is_sim_option = True
            else:
                param_name = name

            if type == 'int':
                value = trial.suggest_int(name, args[0], args[1])
            elif type == 'float':
                log = (len(args) > 2 and args[2] == 'log')
                value = trial.suggest_float(name, args[0], args[1], log=log)
            elif type == 'categorical':
                value = trial.suggest_categorical(name, args[0])
            
            if is_sim_option:
                sim_options[param_name] = value
            else:
                params[param_name] = value

        if sim_options:
            params['sim_options'] = sim_options
            
        try:
            # NOTA: Usamos a classe do modelo (ex: SVD) e passamos
            # os parâmetros da 'trial' para ela.
            # Precisamos criar uma NOVA instância a cada 'trial'.
            # A 'model_class' passada é só um "molde" (ex: SVD(random_state=42))
            # Vamos instanciar corretamente:
            model_instance = model_class.__class__(**params)
            
            cv_results = cross_validate(model_instance, data, measures=['rmse'], cv=3, verbose=False, n_jobs=-1)
            return np.mean(cv_results['test_rmse'])
        
        except Exception as e:
            # Correção de bug: model_class.__class__ pode não ser o ideal se
            # model_class for a própria classe (ex: SVD) e não uma instância.
            # Vamos tentar de novo, assumindo que model_class é a CLASSE
            try:
                model_instance = model_class(**params)
                cv_results = cross_validate(model_instance, data, measures=['rmse'], cv=3, verbose=False, n_jobs=-1)
                return np.mean(cv_results['test_rmse'])
            except TypeError:
                # Se falhar, é provável que a 'model_class' original (ex: SVD(random_state=42))
                # tenha params que conflitam. A forma mais segura é passar a CLASSE.
                # (A lógica original estava correta, mas a chamada no 'main' precisava
                # passar SVD em vez de SVD())
                # Vamos manter a lógica original que é mais robusta:
                pass # A lógica abaixo (fora do 'except') já é a correta

            print(f"[Optuna] Aviso: Tentativa falhou com params {params}. Erro: {e}")
            return 100.0 # Retorna um RMSE horrível

    # --- 2. Rodar a Otimização (Estudo) ---
    study_start_time = time.time()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    study_time = time.time() - study_start_time
    
    best_params_from_study = study.best_params
    best_rmse_cv = study.best_value

    print(f"✓ Otimização concluída em {study_time:.2f}s. Melhor RMSE (CV): {best_rmse_cv:.4f}")
    print(f"Melhores parâmetros encontrados: {best_params_from_study}")

    # --- 3. Preparar Parâmetros Finais ---
    final_params = {}
    final_sim_options = {}
    for name, value in best_params_from_study.items():
        if name.startswith('sim_options__'):
            final_sim_options[name.split('__')[1]] = value
        else:
            final_params[name] = value
    
    if final_sim_options:
        final_params['sim_options'] = final_sim_options
    
    # --- 4. Treinar o Modelo FINAL no trainset completo ---
    print(f"Treinando modelo final '{model_name}' no trainset completo...")
    train_start_time = time.time()
    
    # Aqui instanciamos o modelo final com os melhores parâmetros
    # Usamos o __class__ do "molde" para criar um novo.
    # Ex: molde = SVD(random_state=42) -> molde.__class__ = SVD
    final_model = model_class.__class__(**final_params)
    
    # Adiciona random_state se estava no molde original (como SVD)
    if hasattr(model_class, 'random_state'):
        final_model.random_state = model_class.random_state
        
    final_model.fit(trainset)
    
    train_time = time.time() - train_start_time

    # --- 5. Testar o Modelo FINAL no testset ---
    print(f"Avaliando modelo final no testset...")
    predict_start_time = time.time()
    predictions = final_model.test(testset)
    predict_time = time.time() - predict_start_time

    # --- 6. Calcular Métricas Finais ---
    final_rmse = accuracy.rmse(predictions, verbose=True)
    final_mae = accuracy.mae(predictions, verbose=True)
    
    errors = [abs(pred.r_ui - pred.est) for pred in predictions]
    
    # --- 7. Montar o Dicionário de Retorno ---
    final_results = {
        'model_name': model_name,
        'metrics': {
            'rmse': final_rmse,
            'mae': final_mae,
            'train_time': train_time,
            'predict_time': predict_time,
            'optimization_time': study_time,
            'best_cv_rmse': best_rmse_cv
        },
        'errors': errors,
        'best_params': final_params,
        'tuning_results': study.trials_dataframe() # Histórico completo
    }
    
    # Modificado para retornar o modelo e os resultados
    return final_results, final_model

# --- 3. Função de Plotagem ---
def plotar_resultados_optuna(df_trials, model_name):
    """
    Plota a convergência e os parâmetros de um DataFrame de 'trials' do Optuna.
    """
    print(f"\nGerando gráficos para {model_name}...")
    
    # Garante que a pasta de imagens existe
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    # --- 1. Plot de Convergência ---
    df_trials['best_rmse'] = df_trials['value'].cummin()
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df_trials, 
        x='number',
        y='value',
        label='Iteration RMSE',
        marker='o',
        alpha=0.7
    )
    sns.lineplot(
        data=df_trials, 
        x='number', 
        y='best_rmse',
        label='Best RMSE',
        color='red',
        linewidth=2
    )
    plt.title(f'Convergência da Otimização Bayesiana ({model_name})', fontsize=16)
    plt.xlabel('Tentativa (Evaluation)', fontsize=12)
    plt.ylabel('RMSE (cv=3)', fontsize=12)
    plt.legend()
    # Salva a imagem
    conv_path = os.path.join(IMAGES_DIR, f'optuna_convergence_{model_name}.png')
    plt.savefig(conv_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico de convergência salvo em: {conv_path}")

    # --- 2. Plots de Parâmetros ---
    param_cols = [col for col in df_trials.columns if col.startswith('params_')]
    
    if not param_cols:
        print("Nenhum parâmetro ('params_...') encontrado no DataFrame de 'trials' para plotar.")
        return

    num_params = len(param_cols)
    # Ajuste para garantir que 'axes' seja sempre uma lista, mesmo com 1 parâmetro
    fig, axes = plt.subplots(1, num_params, figsize=(7 * num_params, 6), squeeze=False)
    axes = axes.flatten() # Transforma o array 2D (de 1 linha) em 1D

    fig.suptitle(f'RMSE vs Hiperparâmetros ({model_name})', fontsize=18, y=1.03)

    for i, param_name in enumerate(param_cols):
        clean_name = param_name.replace('params_', '').replace('sim_options__', '')
        
        # Tenta plotar como scatter plot se for numérico
        try:
            # Converte para numérico, se falhar, é categórico
            numeric_data = pd.to_numeric(df_trials[param_name])
            sns.scatterplot(ax=axes[i], data=df_trials, x=numeric_data, y='value', hue='number', palette='viridis', legend=(i == num_params - 1))
            
            # Checa se o parâmetro foi buscado em escala log
            if numeric_data.min() > 0 and numeric_data.max() / numeric_data.min() > 50:
                axes[i].set_xscale('log')
                axes[i].set_xlabel('Valor (Escala Log)', fontsize=12)
            else:
                axes[i].set_xlabel('Valor', fontsize=12)

        except ValueError:
            # Se falhar, trata como categórico (usa stripplot)
            sns.stripplot(ax=axes[i], data=df_trials, x=param_name, y='value', hue='number', palette='viridis', legend=False, dodge=True, alpha=0.7)
            axes[i].set_xlabel('Categoria', fontsize=12)

        axes[i].set_title(clean_name, fontsize=14)
        axes[i].set_ylabel('RMSE', fontsize=12)

    if num_params > 0:
        # Move a legenda para fora do gráfico
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles: # Só mostra a legenda se ela existir
            axes[-1].legend().remove() # Remove a legenda de dentro
            fig.legend(handles, labels, title='Iteration', bbox_to_anchor=(1.03, 0.9), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.98, 0.98]) # Ajusta espaço para a legenda
    # Salva a imagem
    params_path = os.path.join(IMAGES_DIR, f'bayesian_params_{model_name}.png')
    plt.savefig(params_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico de parâmetros salvo em: {params_path}")


# --- 4. Bloco Principal de Execução (MODIFICADO) ---

if __name__ == "__main__":
    
    print("Iniciando script de otimização...")
    
    # 1. Carregar dados
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    try:
        ratings_df = pd.read_csv(ratings_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {ratings_path}")
        print("Por favor, garanta que 'data/ratings.csv' existe.")
        exit()
        
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # 2. Separar train/test
    print("Separando dados em treino e teste...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # 3. Definir os "trabalhos" (jobs) de otimização
    
    # --- Trabalho 1: SVD ---
    svd_job = {
        # Passamos um "molde" do modelo. O random_state é importante!
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
        # KNNBasic não aceita random_state na inicialização
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
    
    # --- Garantir que as pastas existem ANTES do loop ---
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 4. EXECUTAR A OTIMIZAÇÃO (EM LOOP)
    
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
            trainset=trainset,   # 'trainset' para treinar o modelo final
            testset=testset,     # 'testset' para validar o modelo final
            model_name=model_name,
            n_trials=50      # Número de tentativas para cada modelo
        )
        
        # 4B. SALVAR O MODELO FINAL (.pkl)
        print(f"\n--- Salvando modelo final para {model_name} ---")
        model_filename = f"final_{model_name}.pkl"
        model_save_path = os.path.join(MODELS_DIR, model_filename)
        with open(model_save_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Modelo salvo com sucesso em: {model_save_path}")

        # 4C. SALVAR OS RESULTADOS (CSV e JSON)
        print(f"\n--- Salvando resultados da otimização para {model_name} ---")

        # 4C-1. Salvar o HISTÓRICO COMPLETO (DataFrame) como CSV
        try:
            history_df = results['tuning_results']
            csv_filename = f"optuna_history_{model_name}.csv"
            csv_save_path = os.path.join(RESULTS_DIR, csv_filename)
            history_df.to_csv(csv_save_path, index=False)
            print(f"✓ Histórico de 'trials' (CSV) salvo em: {csv_save_path}")
        except Exception as e:
            print(f"Erro ao salvar histórico CSV: {e}")

        # 4C-2. Salvar o SUMÁRIO (Métricas e Params) como JSON
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

        # 4D. Plotar os resultados
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