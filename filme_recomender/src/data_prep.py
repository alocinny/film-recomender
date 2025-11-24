import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from . import config

def load_data_brutos():

    """Carrega os dados brutos."""

    print("--> Carregando dados...")
    try:
        ratings = pd.read_csv(config.DATA_RAW / 'ratings.csv')
        movies = pd.read_csv(config.DATA_RAW / 'movies.csv')
        tags = pd.read_csv(config.DATA_RAW / 'tags.csv')
        return ratings, movies, tags
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivos não encontrados em {config.DATA_RAW}. Verifique se ratings.csv, movies.csv e tags.csv estão lá.")


# No arquivo src/training.py

def get_content_features():
    """
    Recupera features de conteúdo lendo o dataset de ratings enriquecido.
    """
    try:
        # MUDANÇA AQUI: Lemos 'ratings_enriched.parquet' em vez de 'movies_enriched'
        # Lemos apenas 1 linha para ser super rápido, pois só queremos os nomes das colunas
        path = config.DATA_PROCESSED / 'ratings_enriched.parquet'
        df = pd.read_parquet(path).head(1) 
    except FileNotFoundError:
        print(f"⚠️ Erro: '{path}' não encontrado. Rode a Etapa 1 do main.py.")
        return []

    # 1. Lista Fixa de Gêneros (Segurança)
    known_genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    features = []
    
    # 2. Adiciona 'year' (ou 'release_year')
    if 'year' in df.columns:
        features.append('year')
    elif 'release_year' in df.columns:
        features.append('release_year')
        
    # 3. Adiciona Gêneros (apenas os que existem nas colunas)
    features += [col for col in known_genres if col in df.columns]
    
    # 4. Adiciona Tags (Dinamicamente: tudo que começa com 'tag_')
    features += [col for col in df.columns if col.startswith('tag_')]
    
    print(f"Features detectadas no dataset: {len(features)}")
    
    if len(features) == 0:
        print("ALERTA: Nenhuma feature encontrada. Verifique as colunas do seu parquet.")
        print(f"Colunas disponíveis: {df.columns.tolist()[:15]}...")
        
    return features

def load_data_processados():

    CONTENT_FEATURES = get_content_features()

    print("--> Carrega os dados processados....")
    try:
        df = pd.read_csv(config.DATA_PROCESSED / 'ratings_enriched.csv')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivos não encontrados em {config.DATA_PROCESSED}. Verifique se ratings_enriched.csv estão lá.")


def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return np.nan

def process_movies_features(movies, tags):
    
    # Extrai Ano, One-Hot Encoding de Gêneros, TF-IDF das Tags
    
    print("--> Processando Features de Filmes (Ano, Gêneros, Tags)...")
    
    # Ano
    movies['year'] = movies['title'].apply(extract_year)
    movies['year'] = movies['year'].fillna(movies['year'].median()).astype(int)
    
    # Gêneros (One-Hot)
    genres_expanded = movies['genres'].str.get_dummies(sep='|')
    movies_enriched = pd.concat([movies, genres_expanded], axis=1)
    
    # Limpeza de colunas desnecessárias
    cols_to_drop = ['genres', '(no genres listed)']
    movies_enriched = movies_enriched.drop(columns=[c for c in cols_to_drop if c in movies_enriched.columns])
    
    # Tags (TF-IDF)
    print("    Calculando TF-IDF das Tags...")
    tags['tag'] = tags['tag'].astype(str).fillna('')
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    
    tfidf = TfidfVectorizer(stop_words=config.STOP_WORDS, max_features=config.TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(movie_tags['tag'])
    
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tag_{i}" for i in tfidf.get_feature_names_out()])
    tfidf_df['movieId'] = movie_tags['movieId']
    
    # Merge Tags com Filmes
    movies_enriched = pd.merge(movies_enriched, tfidf_df, on='movieId', how='left')
    
    # Preencher NaNs das tags com 0
    tag_cols = [c for c in movies_enriched.columns if c.startswith('tag_')]
    movies_enriched[tag_cols] = movies_enriched[tag_cols].fillna(0)
    
    return movies_enriched

def process_ratings_and_popularity(ratings, movies_enriched):
   
    # Trata Timestamps, Calcula Popularidade (n_ratings, avg_rating), Aplica Corte (Filtro < 5 ratings), Calcula Score Ponderado (IMDB), Calcula Rating Centered (User Bias)
    print("--> Processando Ratings e Métricas de Popularidade...")
    
    # Temporal
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['rating_year'] = ratings['timestamp'].dt.year
    
    # Popularidade Básica
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    movie_stats.columns = ['movieId', 'n_ratings', 'avg_rating']
    
    # Merge preliminar para filtrar
    movies_temp = pd.merge(movies_enriched, movie_stats, on='movieId', how='left').fillna(0)
    
    # Filtro de Corte (< 5 ratings)
    print(f"    Aplicando corte de filmes com menos de {config.MIN_RATINGS_CUTOFF} avaliações...")
    movies_to_keep = movies_temp[movies_temp['n_ratings'] >= config.MIN_RATINGS_CUTOFF]['movieId']
    
    ratings_filtered = ratings[ratings['movieId'].isin(movies_to_keep)].copy()
    movies_final = movies_temp[movies_temp['movieId'].isin(movies_to_keep)].copy()
    
    # Score Ponderado (Weighted Rating)
    C = movies_final['avg_rating'].mean()
    m = movies_final['n_ratings'].quantile(0.90)
    
    def weighted_rating(x, m=m, C=C):
        v = x['n_ratings']
        R = x['avg_rating']
        return (v/(v+m) * R) + (m/(v+m) * C)
    
    movies_final['score_ponderado'] = movies_final.apply(weighted_rating, axis=1)
    
    # User Bias (Rating Centered)
    print("    Calculando Viés do Usuário (Centering)...")
    user_means = ratings_filtered.groupby('userId')['rating'].mean().reset_index()
    user_means.columns = ['userId', 'user_mean']
    
    ratings_filtered = pd.merge(ratings_filtered, user_means, on='userId', how='left')
    ratings_filtered['rating_centered'] = ratings_filtered['rating'] - ratings_filtered['user_mean']
    
    return ratings_filtered, movies_final

def merge_and_normalize(ratings, movies):
    
    # Junta tudo (Ratings + Movies Enriched), gera versão normalizada
    
    print("--> Consolidando Dataset Final...")
    
    # Merge Final
    # Removemos colunas duplicadas se existirem
    cols_to_drop = ['n_ratings', 'avg_rating']
    ratings_clean = ratings.drop(columns=[c for c in cols_to_drop if c in ratings.columns], errors='ignore')
    
    dataset_final = pd.merge(ratings_clean, movies, on='movieId', how='inner')
    
    # Reordenar colunas
    cols = list(dataset_final.columns)
    priority_cols = ['userId', 'movieId', 'rating', 'n_ratings', 'avg_rating', 'score_ponderado']
    remaining = [c for c in cols if c not in priority_cols]
    dataset_final = dataset_final[priority_cols + remaining]
    
    # Normalização
    print("--> Criando versão Normalizada (MinMax)...")
    df_norm = dataset_final.copy()
    
    cols_to_normalize = [
        'n_ratings', 'avg_rating', 'rating_year', 'user_mean', 
        'rating_centered', 'year', 'rating', 'score_ponderado'
    ]
    # Filtra apenas as que existem
    cols_to_normalize = [c for c in cols_to_normalize if c in df_norm.columns]
    
    scaler = MinMaxScaler()
    df_norm[cols_to_normalize] = scaler.fit_transform(df_norm[cols_to_normalize])
    
    return dataset_final, df_norm

def run_pipeline():
    # Load
    ratings_raw, movies_raw, tags_raw = load_data_brutos()
    
    # Process Movies (Content Features)
    movies_enriched = process_movies_features(movies_raw, tags_raw)
    
    # Process Ratings & Popularity & Filter
    ratings_filtered, movies_final = process_ratings_and_popularity(ratings_raw, movies_enriched)
    
    # Final Merge & Normalize
    df_final, df_norm = merge_and_normalize(ratings_filtered, movies_final)
    
    # Save
    print("--> Salvando arquivos...")

    # Versão Original Enriquecida
    df_final.to_parquet(config.DATA_PROCESSED / 'ratings_enriched.parquet', index=False)
    df_final.to_csv(config.DATA_PROCESSED / 'ratings_enriched.csv', index=False)
    
    # Versão Normalizada
    df_norm.to_parquet(config.DATA_PROCESSED / 'ratings_enriched_normalized.parquet', index=False)
    df_norm.to_csv(config.DATA_PROCESSED / 'ratings_enriched_normalized.csv', index=False)
    
    print("Pipeline de Preprocessamento concluído com sucesso!")

if __name__ == "__main__":
    run_pipeline()