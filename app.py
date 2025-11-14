import streamlit as st
import pandas as pd
import pickle
import os

DATA_DIR = 'data'
MODELS_DIR = 'models'

# modelo
MODEL_NAME = 'final_svd_model.pkl'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')

@st.cache_data
def laod_data():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return None, None
    
    try:
        movies_df = pd.read_csv(MOVIES_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None
    
    return model, movies_df

@st.cache_data
def load_ratings():
    try:
        ratings_df = pd.read_csv(RATINGS_PATH)
        return ratings_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

# funcao principal de recomendacao
def get_top_n_recomendations(user_id, model, movies_df, ratings_df, n=10):
    all_movies_id = movies_df['movieId'].unique()

    try:
        user_id_int = int(user_id)
        seen_movie_ids = ratings_df[ratings_df['userId'] == user_id_int]['movieId'].unique()

    except Exception as e:
        print(f"Error converting user_id to integer: {e}")
        seen_movie_ids = []
    
    unseen_movie_ids = list(set(all_movies_id) - set(seen_movie_ids))

    if not unseen_movie_ids:
        return pd.DataFrame()
    
    # calcular a predicao pra cada filme não visto
    predictions = []
    for movie_id in unseen_movie_ids:
        pred = model.predict(uid=user_id_int, iid=movie_id)
        predictions.append((movie_id, pred.est))
    
    # ordenar pela nota estimada (maior p menor)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # fazer o top-n movies_ids
    top_n_preds = predictions[:n]
    top_n_movie_ids = [pred[0] for pred in top_n_preds]

    # criar dataframe final com os titulos e frames
    top_n_df = movies_df[movies_df['movieId'].isin(top_n_movie_ids)].copy()

    predicted_ratings_map = {movie_id: est for movie_id, est in top_n_preds}
    top_n_df['predicted_rating'] = top_n_df['movieId'].map(predicted_ratings_map)

    top_n_df['movieId_cat'] = pd.Categorical(top_n_df['movieId'], categories=top_n_movie_ids, ordered=True)
    final_df = top_n_df.sort_values(by='movieId_cat')

    final_df = final_df[['title', 'genres', 'predicted_rating']]
    final_df = final_df.rename(columns={
        'title': 'Title',
        'genres': 'Genres',
    })

    return final_df.reset_index(drop=True)

# interface do streamlit (UI)
st.set_page_config(layout="wide", page_title="Sistema de Recomendação")

st.title("Sistema de Recomendação de Filmes")
st.markdown(f"Desenvolvido com 'scikit-surprise' (modelo: **{MODEL_NAME}**) e 'streamlit'.")

# carregar dados
model, movies_df = laod_data()
ratings_df = load_ratings()

if model is not None and movies_df is not None and ratings_df is not None:
    min_user = int(ratings_df['userId'].min())
    max_user = int(ratings_df['userId'].max())

    st.header("Gerar Recomendações")

    user_id_input = st.number_input(
        label="ID do Usuário",
        min_value=min_user,
        max_value=max_user,
        value=196,
        step=1
    )

    if st.button("Gerar Recomendações", type="primary"):
        with st.spinner("Gerando recomendações..."):
            recommendations_df = get_top_n_recomendations(
                user_id=user_id_input,
                model=model,
                movies_df=movies_df,
                ratings_df=ratings_df,
                n=10
            )

        if recommendations_df.empty:
            st.warning("Nenhuma recomendação disponível para este usuário.")
        else:
            st.subheader(f"Recomendações para o Usuário {user_id_input}")

            st.dataframe(
                recommendations_df.style.format({'nota estimada': '{:.2f}'}),
                use_container_width=True
            )
else:
    st.error("Erro ao carregar os dados")

st.sidebar.header("Sobre")
st.sidebar.info("Esse projeto usa filtragem colaborativa (SVD)"
                "no dataset MovieLens Latest Small para gerar previsões de notas"
                "e recomendar filmes ue o usuário provavelmente vai gostar"
)
st.sidebar.subheader(f"Modelo: {MODEL_NAME}")