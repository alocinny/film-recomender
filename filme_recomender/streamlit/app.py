import streamlit as st
import backend_app as bk
import pandas as pd

# --- Configuração da Página ---
st.set_page_config(
    page_title="CineAI Hub",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Moderno ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fff; }
    h1, h2, h3 { color: #e50914 !important; }
    
    /* Box do Filme */
    .movie-container {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        height: 100%;
    }
    .movie-title {
        font-weight: bold;
        font-size: 0.9rem;
        margin-top: 8px;
        min-height: 40px; /* Alinha titulos */
    }
    
    /* Métricas do Perfil */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #e50914;
    }
</style>
""", unsafe_allow_html=True)

def display_movies_grid(df, show_score=True):
    """
    Exibe grade horizontal larga.
    Args:
        show_score (bool): Se False, esconde a nota (útil para Novo Usuário).
    """
    if not df.empty:
        rows = [df.iloc[i:i+5] for i in range(0, len(df), 5)]
        for row in rows:
            cols = st.columns(5)
            for idx, (_, movie) in enumerate(row.iterrows()):
                with cols[idx]:
                    # Tratamento seguro de dados
                    poster = movie.get('Poster')
                    if pd.isna(poster): poster = "https://via.placeholder.com/300x450?text=Img+N/A"
                    title = movie.get('Title', 'Sem Título')
                    
                    # Lógica para mostrar ou esconder a nota HTML
                    score_html = ""
                    if show_score:
                        score_val = movie.get('Score', 0)
                        score_html = f'<div style="font-weight:bold; color:#e50914; margin-top:5px;">⭐ {score_val:.1f}</div>'

                    st.markdown(f"""
                    <div class="movie-container">
                        <img src="{poster}" width="100%" style="border-radius:4px;">
                        <div class="movie-title">{title}</div>
                        {score_html}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Sem recomendações.")
# --- Load Data ---
model_hb, model_knn_item, model_knn , movies_df, ratings_df = bk.load_data()

# ==============================================================================
# SIDEBAR - CONTROLES
# ==============================================================================
with st.sidebar:
    st.title("CineAI")
    
    st.write("### Quem é você?")
    user_mode = st.radio("", ["Usuário Existente", "Sou Novo Aqui"], index=0)
    
    st.markdown("---")
    

    st.markdown("---")

    model_choice = None
    selected_user_id = None
    new_user_movies = []
    
    if user_mode == "Usuário Existente":

        st.write("### Modelo de IA")
        model_choice = st.selectbox(
        "Escolha o algoritmo:",
        ["Híbrido (SVD+RF)", "KNN (Simples)"]
    )
        if ratings_df is not None:
            min_u, max_u = int(ratings_df['userId'].min()), int(ratings_df['userId'].max())
            selected_user_id = st.number_input("Seu ID", min_u, max_u, value=196)
    
    num_recs = st.slider("Quantos filmes?", 5, 20, 10)
    btn_recommend = st.button("Gerar Recomendações", type="primary")

# ==============================================================================
# CORPO PRINCIPAL
# ==============================================================================

st.title("Recomendação Inteligente de Filmes")

if model_hb is None or model_knn is None:
    st.error("Erro ao carregar sistema. Verifique os arquivos.")
    st.stop()

# ------------------------------------------------------------------------------
# MODO 1: USUÁRIO EXISTENTE
# ------------------------------------------------------------------------------
if user_mode == "Usuário Existente":
    
    # Layout: Perfil à esquerda, infos à direita (ou topo)
    st.subheader(f"Análise do Usuário #{selected_user_id}")
    
    # Pega dados brutos (contagem)
    user_stats = bk.get_user_profile(selected_user_id, ratings_df, movies_df)
    
    if not user_stats.empty:
        st.caption("Baseado no seu histórico de avaliações:")
        
        # Cria 5 colunas para mostrar os gêneros como "KPIs"
        cols = st.columns(5)
        
        # Pega o total para calcular porcentagem
        total_actions = user_stats.sum()
        
        for idx, (genre, count) in enumerate(user_stats.items()):
            col = cols[idx % 5]
            with col:
                # Mostra o Nome e o Número grande
                st.metric(label=genre, value=f"{count} filmes")
                # Barra de progresso relativa ao total
                percent = min(count / total_actions * 2.0, 1.0) # *2.0 para dar mais visual
                st.progress(percent)
    else:
        st.info("Este usuário ainda não avaliou filmes suficientes para traçar um perfil.")

    st.divider()

    if btn_recommend:
        with st.spinner(f"Rodando modelo {model_choice}..."):
            recs = bk.get_recommendations_existing(selected_user_id, model_hb, model_knn, movies_df, ratings_df, model_choice, n=num_recs)
            
            if not recs.empty:
                st.success(f"Top {num_recs} recomendações para você:")
                
                # Grid de Filmes
                rows = [recs.iloc[i:i+5] for i in range(0, len(recs), 5)]
                for row in rows:
                    cols = st.columns(5)
                    for _, (col, movie) in enumerate(zip(cols, row.iterrows())):
                        m = movie[1]
                        with col:
                            st.image(m['Poster'], use_container_width=True)
                            st.markdown(f"<div class='movie-title'>{m['Title']}</div>", unsafe_allow_html=True)
                            st.caption(f"{m['Genres'][:30]}...")
                            st.write(f"⭐ **{m['Score']:.1f}**")
            else:
                st.warning("Não encontramos recomendações novas para este perfil.")

# ------------------------------------------------------------------------------
# MODO 2: NOVO USUÁRIO (Cold Start)
# ------------------------------------------------------------------------------

else:
    st.subheader("Bem-vindo! Vamos calibrar o sistema.")
    st.info("ℹ️ Usando algoritmo **KNN Item-Based** (Inteligência de Itens Similares).")
    
    st.markdown("#### 1. Quais gêneros você prefere?")
    all_genres = bk.get_all_genres(movies_df)
    selected_genres = st.multiselect("Selecione:", options=all_genres)

    st.markdown("#### 2. Marque filmes que você GOSTOU:")
   
    popular_filtered = bk.get_popular_movies_filtered(movies_df, ratings_df, genres=selected_genres, n=50)
    
    # Lista para o selectbox
    movie_options = {f"{row.title}": row.movieId for row in popular_filtered.itertuples()}
    selected_titles = st.multiselect("Filmes:", options=list(movie_options.keys()))
    selected_ids = [movie_options[t] for t in selected_titles]
    
    st.divider()
    
    if btn_recommend:
        if len(selected_ids) < 1:
            st.warning("⚠️ Selecione pelo menos 3 filme para melhores recomendações.")
        else:
            with st.spinner("Buscando itens similares na matriz (Item-Based)..."):
                
               
                models_dict = {"KNN_ITEM": model_knn_item}
                
                recs = bk.get_recommendations_new_user(
                    selected_ids, 
                    models_dict, 
                    movies_df, 
                    ratings_df, 
                    n=num_recs
                )
                
                if not recs.empty:
                        st.success("Recomendações baseadas nos itens que você selecionou:")
                        # AQUI ESTAVA FALTANDO: Usar a função visual pronta em vez do loop manual
                        display_movies_grid(recs, show_score=False)
                else:
                    st.warning("Não encontramos correlações suficientes. Tente selecionar filmes mais populares.")
