import pandas as pd
import pickle
import os
import requests
import random
import streamlit as st
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

MODEL_NAME_HB = 'modelo_hibrido_completo.pkl'
MODEL_PATH_HB = os.path.join(MODELS_DIR, MODEL_NAME_HB)

MODEL_NAME_KNN_U = 'knn_user_model.pkl'
MODEL_PATH_KNN_U = os.path.join(MODELS_DIR, MODEL_NAME_KNN_U)

MODEL_NAME_KNN_I = 'knn_item_model.pkl'
MODEL_PATH_KNN_I = os.path.join(MODELS_DIR, MODEL_NAME_KNN_I)

MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
LINKS_PATH = os.path.join(DATA_DIR, 'links.csv')


TMDB_API_KEY = "d64299e581c02b89401274b04c40121c" 
BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# ==============================================================================
# CARREGAMENTO DE DADOS
# ==============================================================================

@st.cache_data
def load_data():
    """Carrega o modelo, filmes, ratings e links de forma otimizada."""
    model_hb = None
    model_knn = None
    #model_knn_item = None
    movies_df = None
    ratings_df = None
    
    # 1. Carregar Modelo
    try:
        with open(MODEL_PATH_HB, 'rb') as f:
            model_hb = pickle.load(f)
        with open(MODEL_PATH_KNN_U, 'rb') as f:
            model_knn = pickle.load(f)
        # with open(MODEL_PATH_KNN_I, 'rb') as f:
        #     model_knn_item = pickle.load(f)
            
    except FileNotFoundError:
        st.error(f"Erro: Modelo não encontrado em {MODEL_PATH_HB}")
    
    # 2. Carregar Filmes e Links
    try:
        movies_df = pd.read_csv(MOVIES_PATH)
        if os.path.exists(LINKS_PATH):
            links_df = pd.read_csv(LINKS_PATH)
            movies_df = movies_df.merge(links_df, on='movieId', how='left')
    except FileNotFoundError:
        st.error(f"Erro: Arquivo movies.csv não encontrado.")

    # 3. Carregar Ratings
    try:
        ratings_df = pd.read_csv(RATINGS_PATH)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo ratings.csv não encontrado.")

    return model_hb, model_knn , movies_df, ratings_df


# ==============================================================================
# ==============================================================================
# FUNÇÕES AUXILIARES (IMAGENS E PERFIL)
# ==============================================================================

def fetch_poster_url(tmdb_id):
    if pd.isna(tmdb_id) or not TMDB_API_KEY:
        return "https://via.placeholder.com/300x450?text=Sem+Imagem"
    try:
        movie_id = int(tmdb_id)
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=pt-BR"
        response = requests.get(url, timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            if data.get('poster_path'):
                return f"{BASE_IMAGE_URL}{data.get('poster_path')}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=Imagem+Indisponivel"

def get_user_profile(user_id, ratings_df, movies_df):
    """
    Retorna um DataFrame com contagem de gêneros assistidos pelo usuário.
    """
    try:
        # Pega filmes que o usuário avaliou (independente da nota, queremos ver o que ele consome)
        user_movies = ratings_df[ratings_df['userId'] == user_id]
        
        if user_movies.empty:
            return pd.DataFrame()

        liked_movies_data = user_movies.merge(movies_df, on='movieId')
        # Explode os gêneros e conta
        genre_counts = liked_movies_data['genres'].str.split('|').explode().value_counts()
        
        if "(no genres listed)" in genre_counts:
            genre_counts = genre_counts.drop("(no genres listed)")
            
        # Retorna os top 5 com a contagem real
        return genre_counts.head(5)
    except Exception:
        return pd.Series(dtype=int)

def get_popular_movies_for_selection(movies_df, ratings_df, n=20):
    """
    Retorna os filmes mais avaliados para o 'Novo Usuário' escolher.
    """
    top_movies = ratings_df.groupby('movieId').count()['rating'].sort_values(ascending=False).head(n).index
    return movies_df[movies_df['movieId'].isin(top_movies)][['movieId', 'title']]

# ==============================================================================
# LÓGICA DE RECOMENDAÇÃO
# ==============================================================================

def get_recommendations_existing(user_id, model_hb, model_knn, movies_df, ratings_df, model_type="Híbrido (SVD+RF)", n=10):
    """
    Lógica para USUÁRIO EXISTENTE.
    Versão COMPLETA (analisa todos os filmes, sem amostragem).
    """
    
    # 1. Prepara lista de filmes vistos e não vistos
    all_movies_id = movies_df['movieId'].unique()
    try:
        user_id_int = int(user_id)
        seen_movie_ids = ratings_df[ratings_df['userId'] == user_id_int]['movieId'].unique()
    except ValueError:
        seen_movie_ids = []
        user_id_int = user_id
    
    # Lista de TODOS os filmes que o usuário não viu
    unseen_movie_ids = list(set(all_movies_id) - set(seen_movie_ids))
    
    if not unseen_movie_ids: 
        return pd.DataFrame()

    # --- LÓGICA DO KNN ---
    if model_type == "KNN (Simples)":
        
        knn_predictions = []
        
        # AQUI MUDOU: Iteramos sobre TODOS os unseen_movie_ids, não usamos random.sample
        for movie_id in unseen_movie_ids:
            try:
                pred = model_knn.predict(uid=user_id_int, iid=movie_id)
                knn_predictions.append((movie_id, pred.est))
            except:
                continue
            
        return _format_results(knn_predictions, movies_df, n)


    # --- LÓGICA DO HÍBRIDO ---
    if model_type == "Híbrido (SVD+RF)":        
        
        hb_predictions = []
        
        # Mesma coisa aqui: Removemos a amostragem para testar o Híbrido completo também
        for movie_id in unseen_movie_ids:
            try:
                score = model_hb.predict(uid=user_id_int, iid=movie_id)
                hb_predictions.append((movie_id, score))
            except:
                continue
        
        return _format_results(hb_predictions, movies_df, n)
        
    return pd.DataFrame()

    
def get_recommendations_new_user(selected_movie_ids, model, movies_df, model_type="Híbrido (SVD+RF)", n=10):
    """
    Lógica para NOVO USUÁRIO (Cold Start).
    Recebe uma lista de IDs de filmes que ele marcou como 'Gostei'.
    """
    # -----------------------------------------------------------
    # LÓGICA PLACEHOLDER - PONTO DE ENTRADA PARA O DESENVOLVEDOR
    # -----------------------------------------------------------
    # A ideia aqui seria:
    # 1. Pegar o vetor médio desses filmes selecionados.
    # 2. Achar filmes similares (KNN).
    # 3. Ou criar um 'User Temporário' no SVD e prever.
    
    # Por enquanto, vamos simular recomendando filmes do mesmo gênero dos selecionados
    if not selected_movie_ids:
        return pd.DataFrame()

    # Pega gêneros dos filmes selecionados
    selected_genres = movies_df[movies_df['movieId'].isin(selected_movie_ids)]['genres'].str.split('|').explode().unique()
    
    # Filtra filmes candidatos que tenham esses gêneros (simulação barata)
    candidates = movies_df[movies_df['genres'].str.contains('|'.join(selected_genres), na=False)]
    candidates = candidates[~candidates['movieId'].isin(selected_movie_ids)] # Remove os já escolhidos
    
    # Pega 10 aleatórios desse pool (Só para mostrar que funcionou a UI)
    fake_recs = candidates.sample(min(n, len(candidates)))
    
    # Cria estrutura de 'score' fake só para exibir
    predictions = [(row.movieId, random.uniform(3.5, 5.0)) for row in fake_recs.itertuples()]
    
    return _format_results(predictions, movies_df, n)

def _format_results(predictions, movies_df, n):
    """Função interna para formatar a saída com Imagens e Títulos"""
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    
    results = []
    for movie_id, score in top_n:
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if movie_row.empty: continue
        movie_info = movie_row.iloc[0]
        
        poster_url = "https://via.placeholder.com/300x450?text=Sem+Link"
        if 'tmdbId' in movie_info:
            poster_url = fetch_poster_url(movie_info['tmdbId'])
            
        results.append({
            'movieId': movie_id,
            'Title': movie_info['title'],
            'Genres': str(movie_info['genres']).replace('|', ', '),
            'Score': score,
            'Poster': poster_url
        })
    return pd.DataFrame(results)


# def get_recommendations_new_user(selected_movie_ids, models, movies_df, ratings_df, n=10):
#     """
#     Usa o modelo KNN ITEM-BASED treinado para achar itens similares.
#     """

#     knn_item = models.get("KNN_ITEM")
    
#     # 1. Descobrir quais gêneros o usuário selecionou
#     input_movies = movies_df[movies_df['movieId'].isin(selected_movie_ids)]
#     input_genres = set(input_movies['genres'].str.split('|').explode())
#     # Removemos gêneros genéricos se quiser limpar mais
#     input_genres.discard('(no genres listed)')
    
#     # --- CENÁRIO A: TEMOS O MODELO TREINADO (Ideal) ---
#     if knn_item is not None:
#         candidates = {}
        
#         for raw_id in selected_movie_ids:
#             try:
#                 # Converte ID Real -> ID Interno do Surprise
#                 inner_id = knn_item.trainset.to_inner_iid(raw_id)
                
#                 # Pega os 10 vizinhos mais próximos na matriz treinada
#                 neighbors = knn_item.get_neighbors(inner_id, k=10)
                
#                 for i, neighbor_inner_id in enumerate(neighbors):
#                     # Converte ID Interno -> ID Real
#                     neighbor_raw_id = knn_item.trainset.to_raw_iid(neighbor_inner_id)

#                     cand_row = movies_df[movies_df['movieId'] == neighbor_raw_id]
#                     if not cand_row.empty:
#                         cand_genres = set(cand_row.iloc[0]['genres'].split('|'))
                        
#                         # Só aceita se tiver intersecção de gêneros (pelo menos 1 igual)
#                         if not input_genres.intersection(cand_genres):
#                             continue # Pula esse filme, ele não tem nada a ver com o gosto

#                     # Score baseado na ordem (1º vizinho ganha mais pontos que o 10º)
#                     score = (1.0 / (i + 1)) * 5.0 
                    
#                     if neighbor_raw_id not in selected_movie_ids:
#                         candidates[neighbor_raw_id] = candidates.get(neighbor_raw_id, 0) + score
#             except ValueError:
#                 continue # Filme selecionado não estava no treino (muito novo)

#         if candidates:
#             # Normaliza e formata
#             final_preds = list(candidates.items())
#             return _format_results(final_preds, movies_df, n)

#     # --- CENÁRIO B: FALLBACK (Se modelo falhar ou não achar vizinhos) ---
#     # Usa a lógica dinâmica de vizinhança de usuários (que criamos antes)
    
#     # 1. Acha usuários que gostaram do que você marcou
#     similar_users = ratings_df[
#         (ratings_df['movieId'].isin(selected_movie_ids)) & 
#         (ratings_df['rating'] >= 4.0)
#     ]['userId'].unique()

#     if len(similar_users) < 2:
#         similar_users = ratings_df[ratings_df['movieId'].isin(selected_movie_ids)]['userId'].unique()

#     if len(similar_users) == 0:
#         return pd.DataFrame() # Desiste se não achar ninguém

#     # 2. Vê o que eles viram
#     recs = ratings_df[
#         (ratings_df['userId'].isin(similar_users)) & 
#         (~ratings_df['movieId'].isin(selected_movie_ids))
#     ]
    
#     # 3. Score = Popularidade * Média
#     scores = recs.groupby('movieId').agg(
#         count=('rating', 'count'), 
#         mean=('rating', 'mean')
#     )
#     scores = scores[scores['count'] >= 2] # Filtro mínimo
#     scores['final_score'] = scores['mean'] # Simplificado
    
#     top_ids = scores.sort_values('final_score', ascending=False).head(n).index.tolist()
#     preds = [(mid, scores.loc[mid, 'final_score']) for mid in top_ids]
    
#     return _format_results(preds, movies_df, n)
