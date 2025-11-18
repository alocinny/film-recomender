import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_DIR = 'results'
IMAGES_DIR = 'images'

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
