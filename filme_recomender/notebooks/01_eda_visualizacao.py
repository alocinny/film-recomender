import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os

# Adiciona a pasta raiz ao path para conseguir importar o src
sys.path.append('..') 
from src import config

# Configurações Visuais
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
PALETTE = "mako"

print("✅ Bibliotecas importadas e ambiente configurado.")

# 1. Carregar Dados Brutos (Para análises iniciais)
ratings = pd.read_csv(config.DATA_RAW / 'ratings.csv')
movies = pd.read_csv(config.DATA_RAW / 'movies.csv')
tags = pd.read_csv(config.DATA_RAW / 'tags.csv')

print(f"Dados Brutos Carregados:")
print(f"Ratings: {ratings.shape}")
print(f"Movies: {movies.shape}")

# 2. Carregar Dados Processados (Para analisar correlações e features criadas)
# Só vai funcionar se você já rodou o 'main.py' ou 'data_prep.py'
try:
    df_enriched = pd.read_parquet(config.DATA_PROCESSED / 'ratings_enriched.parquet')
    print(f"Dados Enriquecidos Carregados: {df_enriched.shape}")
except FileNotFoundError:
    print("⚠️ Aviso: 'ratings_enriched.parquet' não encontrado. Rode o pipeline de dados primeiro para ver correlações.")
    df_enriched = None

# Cálculo de Sparsity
n_users = ratings['userId'].nunique()
n_items = ratings['movieId'].nunique()
n_ratings = len(ratings)
total_possible = n_users * n_items
sparsity = 1 - (n_ratings / total_possible)

print(f"Sparsity do Dataset: {sparsity:.4%}")

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1. Distribuição Global
sns.countplot(ax=axes[0], x='rating', data=ratings, palette=PALETTE)
axes[0].set_title('Distribuição Global dos Ratings')
axes[0].set_xlabel('Nota')

# 2. Heatmap de Esparsidade (Zoom Top 100)
user_counts = ratings['userId'].value_counts()
top_100_users = user_counts.head(100).index
movie_counts = ratings['movieId'].value_counts()
top_100_movies = movie_counts.head(100).index

sample_df = ratings[
    ratings['userId'].isin(top_100_users) & 
    ratings['movieId'].isin(top_100_movies)
]
sample_matrix = sample_df.pivot(index='userId', columns='movieId', values='rating')

sns.heatmap(sample_matrix.notna(), cmap='Blues', cbar=False, xticklabels=False, yticklabels=False, ax=axes[1])
axes[1].set_title('Zoom de Esparsidade (Top 100 Users x Items)')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Ratings por Filme
movie_counts = ratings.groupby('movieId').size().sort_values(ascending=False)
sns.lineplot(x=range(len(movie_counts)), y=movie_counts.values, ax=axes[0], color='#2E8B57')
axes[0].axhline(y=5, color='red', linestyle='--', label='Corte (5 avaliações)')
axes[0].set_yscale('log')
axes[0].set_title('Cauda Longa: Popularidade dos Filmes')
axes[0].set_ylabel('Qtd Avaliações (Log)')
axes[0].legend()

# 2. Ratings por Usuário
user_activity = ratings.groupby('userId').size().sort_values(ascending=False)
sns.lineplot(x=range(len(user_activity)), y=user_activity.values, ax=axes[1], color='#4682B4')
axes[1].set_yscale('log')
axes[1].set_title('Atividade dos Usuários')
axes[1].set_ylabel('Qtd Avaliações (Log)')

plt.tight_layout()
plt.show()

# 1. WordCloud das Tags
if not tags.empty:
    tag_text = ' '.join(tags['tag'].dropna().astype(str).values)
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(tag_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Principais Tags Atribuídas pelos Usuários')
    plt.show()

# 2. Distribuição de Gêneros
genres_expanded = movies['genres'].str.get_dummies(sep='|')
genre_counts = genres_expanded.sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette=PALETTE)
plt.title('Quantidade de Filmes por Gênero')
plt.show()

# 2. Seleção Automática de Colunas Numéricas
# Seleciona apenas numeros (float/int)
numeric_df = df_enriched.select_dtypes(include=[np.number])

# 3. Limpeza: Remover colunas que não são features reais (IDs e Timestamps)
# Não queremos saber se o 'userId' tem correlação com 'rating' (pois é só um identificador)
cols_to_exclude = ['userId', 'movieId', 'timestamp']
cols_for_corr = [c for c in numeric_df.columns if c not in cols_to_exclude]

print(f"Calculando correlação de {len(cols_for_corr)} features...")
print(f"Colunas incluídas: {cols_for_corr[:5]} ...")

# 4. Calcular a Matriz de Correlação
# Isso pode levar alguns segundos dependendo do tamanho do dataset
corr_matrix = numeric_df[cols_for_corr].corr()

# 5. Plotar o Heatmap Gigante
plt.figure(figsize=(24, 20)) # Tamanho bem grande para ler os nomes

sns.heatmap(
    corr_matrix, 
    annot=False,       # False para não poluir visualmente (muitos números)
    cmap='RdBu_r',     # Vermelho (Negativo) <-> Azul (Positivo)
    center=0,          # Garante que o branco seja correlação zero
    vmin=-1, vmax=1,   # Trava a escala entre -1 e 1
    linewidths=0.1,    # Linhas finas para separar
    cbar_kws={"shrink": 0.8} # Barra de cores menorzinha
)

plt.title('Matriz de Correlação Global: Target + Métricas + Gêneros + Tags', fontsize=18)
plt.xticks(rotation=90, fontsize=10) # Rotaciona nomes embaixo
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# 6. Extra: Listar as Top Correlações com o Rating (O que mais impacta a nota?)
print("\n--- O que mais influencia o 'rating'? (Top 10 Correlações) ---")
target_corr = corr_matrix['rating'].drop('rating') # Remove a correlação dele com ele mesmo
top_positive = target_corr.sort_values(ascending=False).head(5)
top_negative = target_corr.sort_values(ascending=True).head(5)

print("📈 Top 5 Correlações POSITIVAS (Aumentam a nota):")
print(top_positive)
print("\n📉 Top 5 Correlações NEGATIVAS (Diminuem a nota):")
print(top_negative)