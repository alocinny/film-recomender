import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader
import os
import json
from collections import Counter

DATA_DIR = 'data'
IMAGES_DIR = 'images'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# carregar dados da EDA

ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
movies_path = os.path.join(DATA_DIR, 'movies.csv')

try: 
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    raise

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

n_users = ratings_df['userId'].nunique()
n_items = ratings_df['movieId'].nunique()
n_ratings = len(ratings_df)

sparsity = 1.0 - (n_ratings / (n_users * n_items))

user_stats = ratings_df.groupby('userId').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()
user_stats.columns = ['user_id', 'n_ratings', 'mean_rating', 'std_rating']

item_stats = ratings_df.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()
item_stats.columns = ['item_id', 'n_ratings', 'mean_rating', 'std_rating']

eda_stats = {
    'n_users': n_users,
    'n_items': n_items,
    'n_ratings': n_ratings,
    'sparsity': sparsity,
    'ratings per user (mean)': f'{n_ratings/n_users}',
    'ratings per film (mean)': f'{n_ratings/n_items}',
    'mean rating': f'{ratings_df["rating"].mean()}',
    'standard deviation': f'{ratings_df["rating"].std()}',
    'ratings distribution': f'{ratings_df["rating"].describe()}',
    'unique ratings': f'{sorted(ratings_df["rating"].unique())}',
    'more active user': f'{user_stats["n_ratings"].max()}',
    'less active user': f'{user_stats["n_ratings"].min()}',
    'high rated movie': f'{item_stats["n_ratings"].max()}',
    'less rated movie': f'{item_stats["n_ratings"].min()}',
}

with open(os.path.join(RESULTS_DIR, 'eda_stats.json'), 'w') as f:
    json.dump(eda_stats, f, indent=4)

# histograma de notas
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=ratings_df, palette='viridis')
plt.title('Rating Distribution')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
hist_path = os.path.join(IMAGES_DIR, 'rating_distribution.png')
plt.savefig(hist_path)
plt.close()

# top 10 filmes mais avaliados
movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
top_10_movies = movie_ratings['title'].value_counts().head(10)

plt.figure(figsize=(12, 8))
sns.barplot(y=top_10_movies.index, x=top_10_movies.values, orient='h', palette='mako')
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Movie Title', fontsize=12)
top10_path = os.path.join(IMAGES_DIR, 'top_10_most_rated_movies.png')
plt.savefig(top10_path, bbox_inches='tight')
plt.close()

# distribuição de avaliações por usuário
user_rating_counts = ratings_df['userId'].value_counts()

plt.figure(figsize=(12, 6))
sns.histplot(user_rating_counts, bins=50, kde=False, log_scale=(False, True))
plt.title('Distribution of Number of Ratings per User', fontsize=16)
plt.xlabel('Number of Ratings', fontsize=12)
plt.ylabel('Number of Users (log scale)', fontsize=12)
plt.axvline(user_rating_counts.mean(), color='red', linestyle='dashed', label='Mean')
plt.legend()
user_hist_path = os.path.join(IMAGES_DIR, 'user_rating_distribution1.png')
plt.savefig(user_hist_path, bbox_inches='tight')
plt.close()

# distribuição de avaliações por filme (cauda longa)
movie_ratings_counts = ratings_df['movieId'].value_counts()

plt.figure(figsize=(12, 6))
sns.histplot(movie_ratings_counts, bins=50, kde=False, log_scale=(True, True))
plt.title('Movie rating dsitribution', fontsize=16)
plt.xlabel('Number of Ratings (log scale)', fontsize=12)
plt.ylabel('Number of Movies (log scale)', fontsize=12)
movie_hist_path = os.path.join(IMAGES_DIR, 'movie_rating_distribution.png')
plt.savefig(movie_hist_path, bbox_inches='tight')
plt.close()

# popularidade dos gêneros
all_genres = Counter()
movies_df['genres'].str.split('|').apply(all_genres.update)

if '(no genres listed)' in all_genres:
    del all_genres['(no genres listed)']

genres_df = pd.DataFrame(all_genres.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

plt.figure(figsize=(14, 10))
sns.barplot(y='Genre', x='Count', data=genres_df, palette='viridis')
plt.title('Movie Genre Popularity', fontsize=16)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Genre', fontsize=12)
genre_bar_path = os.path.join(IMAGES_DIR, 'movie_genre_popularity.png')
plt.savefig(genre_bar_path, bbox_inches='tight')
plt.close()

# esparsidade
user_counts = ratings_df['userId'].value_counts()
top_100_users = user_counts.head(100).index

movie_counts = ratings_df['movieId'].value_counts()
top_100_movies = movie_counts.head(100).index

sample_df = ratings_df[
    ratings_df['userId'].isin(top_100_users) & 
    ratings_df['movieId'].isin(top_100_movies)
]

sample_matrix = sample_df.pivot(
    index='userId', 
    columns='movieId', 
    values='rating'
)

sample_matrix = sample_matrix.reindex(index=top_100_users, columns=top_100_movies)

plt.figure(figsize=(16, 14))
sns.heatmap(
    sample_matrix.notna(),
    cmap='Greys',
    cbar=False,
    xticklabels=True,
    yticklabels=True
)
plt.title('Sparsity Heatmap of Top 100 Users and Movies', fontsize=16)
plt.ylabel('User ID', fontsize=12)
plt.xlabel('Movie ID', fontsize=12)
sparsity_heatmap_path = os.path.join(IMAGES_DIR, 'sparsity_heatmap.png')
plt.savefig(sparsity_heatmap_path, bbox_inches='tight')
plt.close()

# analise temporal
ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
ratings_df['year'] = ratings_df['date'].dt.year
temporal = ratings_df.groupby('year').size().reset_index(name='count')

plt.figure(figsize=(14, 5))
plt.plot(temporal['year'], temporal['count'], marker='o', linewidth=2, markersize=6, color='steelblue')
plt.xlabel('year')
plt.ylabel('ratings')
plt.title('temporal analysis')
temporal_analysis_path = os.path.join(IMAGES_DIR, 'temporal_analysis.png')
plt.savefig(temporal_analysis_path, bbox_inches='tight')
plt.close()





