import pandas as pd
import numpy as np
import os
import pickle
from surprise import SVD, Dataset, Reader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src import config, data_prep
from notebooks.SVDeRF.predict_hybrido import HybridRecommenderSystem 

def split_data_temporal(df, ratio=0.75, col_ts='timestamp', col_year='rating_year'):
    
    # divide o df temporalmente e retorna objetos prontos para o Surprise.

    df_sorted = df.sort_values(by=col_ts) # (do mais antigo para o mais novo)
    
    # 2. Calcular o índice de corte
    cut_index = int(len(df_sorted) * ratio)
    
    # 3. Dividir os DataFrames
    train_df = df_sorted.iloc[:cut_index]
    test_df = df_sorted.iloc[cut_index:]
    
    # Análise rápida para você ver onde o corte aconteceu
    split_date = df_sorted.iloc[cut_index][col_ts]
    split_year = df_sorted.iloc[cut_index][col_year]
    print(f"--- Divisão Temporal Realizada ---")
    print(f"Corte realizado em: {split_date} (Ano aprox: {split_year})")
    print(f"Tamanho Treino: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Tamanho Teste:  {len(test_df)} ({len(test_df)/len(df):.1%})")
    print("-" * 30)

    # CORREÇÃO CRUCIAL: Retornar DataFrames do Pandas, não objetos do Surprise.
    # O Random Forest precisa das colunas de conteúdo, que o Surprise joga fora.
    return train_df, test_df


def train_svd_model(df):
    print(" TREINA O MODELO SVD PARA PEGAR GOSTOS LATENTES")
    print("\n[1/4] Iniciando treinamento do SVD...")

    # O Reader precisa da escala correta
    reader = Reader(rating_scale=(0.5, 5))
    
    # Aqui convertemos o DataFrame para o formato do Surprise
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    # Criar Trainset (Surprise object)
    trainset = data.build_full_trainset()

    # MEHLORES PARAMETROS OBTIDOS NO GRIND SEARCH
    svd_params = {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.05}
    algo_svd = SVD(**svd_params, random_state=42)
    algo_svd.fit(trainset)

    print("SVD treinado com sucesso.")
    with open(config.MODELS_DIR / 'svd_model.pkl', 'wb') as f:
        pickle.dump(algo_svd, f)

    return algo_svd, trainset 
    # algo_svd é o modelo treinado.
    # trainset é um objeto da bib suprise que representa a matriz de utilidade (Usuário x Item).


def create_hybrid_dataset(df, algo_svd, trainset, features_list, n_factors=100):
    """
    Cria o dataset X = (vetores user + vetores item + conteudo) e y = (nota).
    """
    print("\n[2/4] Gerando vetores latentes e montando dataset híbrido...")

    def get_vec(uid, is_user=True): # pegando do svd o vetor de item e o vetor de usu
        try:
            if is_user:
                inner_id = trainset.to_inner_uid(uid) # Traduz ID Real -> ID Interno
                return algo_svd.pu[inner_id]          # Pega o vetor do usuário
            else:
                inner_id = trainset.to_inner_iid(uid) # Traduz ID Real -> ID Interno
                return algo_svd.qi[inner_id]          # Pega o vetor do item
        except ValueError:
            return np.zeros(n_factors)                # Segurança Se aparecer um usuário ou filme novo que não estava no treino do SVD, o código da erro.
    
    # Aplicar transformações
    df_temp = df.copy()
    df_temp['user_vec'] = df_temp['userId'].apply(lambda x: get_vec(x, is_user=True))
    df_temp['item_vec'] = df_temp['movieId'].apply(lambda x: get_vec(x, is_user=False))
    
    ## userId,  movieId,  rating,    user_vec (O Gosto),           item_vec (O Perfil do Filme)
    
    # COMO OS MODELOS SAO SABEM LER VETORES, PRECISAMOS DIVIDIR CADA UM EM UMA COLINA. ai cada vetor vai ter sua coluna

    # Expandir arrays para colunas
    user_vec_df = pd.DataFrame(df_temp['user_vec'].tolist(), index=df_temp.index).add_prefix('u_lat_')
    item_vec_df = pd.DataFrame(df_temp['item_vec'].tolist(), index=df_temp.index).add_prefix('i_lat_')

    # Concatenar tudo
    X = pd.concat([user_vec_df, item_vec_df, df_temp[features_list]], axis=1)
    y = df_temp['rating']
    
    return X, y, df_temp 
    # Retorna df_temp pois ele tem os vetores cacheados útil para o catálogo


def train_random_forest(X_train, y_train, X_test, y_test):
    """Treina o modelo final que combina tudo."""
    print("\n[3/4] Treinando Random Forest Híbrido...")
    
    # Nota: Não fazemos split aqui dentro. Usamos o split temporal que veio de fora.
    
    regr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    regr.fit(X_train, y_train)

    # Métricas rápidas
    preds = regr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Modelo Híbrido - RMSE: {rmse:.4f} | MAE: {mean_absolute_error(y_test, preds):.4f}")
    
    return regr # regr é uma "Máquina" (um Objeto Python) que contém 100 árvores de decisão dentro dele.
    #Cada uma dessas 100 árvores é um fluxograma gigante de perguntas "Sim ou Não" que o computador criou sozinho.
    # o regr possui o predict() 


def save_hybrid_model(algo_svd, regr_rf, trainset, catalog, features_list, filename='modelo_hibrido_completo.pkl'):
    print(f"\nEmpacotando o modelo inteligente...")
    print("\n[4/4] Salvando arquivo e gerando obj da CLASSE HYBRIDO...")
    
    # Prepara os dados auxiliares
    catalog_vectors = catalog['item_vec'].to_dict()
    catalog_content = catalog[features_list]

    # CRIAMOS O OBJETO DA CLASSE (AQUI ESTÁ A MÁGICA)
    final_model_object = HybridRecommenderSystem(
        svd_model=algo_svd,
        rf_model=regr_rf,
        trainset=trainset,
        catalog_vectors=catalog_vectors,
        catalog_content=catalog_content
    )
    
    # Salva o objeto inteiro
    path = os.path.join('models', filename)
    
    with open(path, 'wb') as f:
        pickle.dump(final_model_object, f) # Salva a CLASSE, não um dicionário
    
    print(f"Modelo salvo em {path}. Agora ele tem o método .predict()!")



def run():
    # 1. Carregar dados
    dados_brutos = data_prep.load_data_processados()
    
    if isinstance(dados_brutos, tuple):
        df = dados_brutos[0]
        valid_features = dados_brutos[1] if len(dados_brutos) > 1 else data_prep.get_content_features()
    else:
        df = dados_brutos
        valid_features = data_prep.get_content_features()

    # 2. DIVISÃO TEMPORAL
    train_df, test_df = split_data_temporal(df)

    # 3. TREINA SVD
    algo_svd, trainset_svd = train_svd_model(train_df)

    # 4. DATASETS HÍBRIDOS
    print("Criando dataset híbrido de TREINO...")
    # AQUI ESTÁ A CORREÇÃO 1: Capturamos o 'train_hybrid_df' em vez de usar '_'
    X_train, y_train, train_hybrid_df = create_hybrid_dataset(train_df, algo_svd, trainset_svd, valid_features)

    print("Criando dataset híbrido de TESTE...")
    X_test, y_test, _ = create_hybrid_dataset(test_df, algo_svd, trainset_svd, valid_features)

    # 5. TREINO RANDOM FOREST
    regr = train_random_forest(X_train, y_train, X_test, y_test)
    
    # 6. SALVAR MODELO
    # AQUI ESTÁ A CORREÇÃO 2: Preparar o catálogo
    print("Preparando catálogo de filmes único...")
    
    # Pegamos o dataframe que JÁ TEM os vetores (train_hybrid_df)
    # Removemos duplicatas para ter apenas 1 linha por filme
    # Definimos o movieId como índice para que o dicionário fique {10: [vetor], 25: [vetor]}
    catalog_df = train_hybrid_df.drop_duplicates(subset=['movieId']).set_index('movieId')
    
    save_hybrid_model(algo_svd, regr, trainset_svd, catalog_df, valid_features)

    print("FINALIZADO.")

