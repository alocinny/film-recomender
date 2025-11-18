import time
import numpy as np
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

# Desabilitar logs de cada 'trial' do Optuna para limpar o console
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 2. Função Principal de Otimização ---
def run_bayesian_search(model_class, param_space, data, trainset, testset,
                        model_name, n_trials):
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

