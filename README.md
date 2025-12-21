<div align="center">
  <h1>CineAI Hub</h1>
  <p>
    <b>Sistema de Recomendação Híbrido (SVD + KNN + Random Forest)</b>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Python_3.11-0d1117?style=flat-square&logo=python&logoColor=39d353" />
    <img src="https://img.shields.io/badge/Streamlit-0d1117?style=flat-square&logo=streamlit&logoColor=39d353" />
    <img src="https://img.shields.io/badge/Scikit_Learn-0d1117?style=flat-square&logo=scikitlearn&logoColor=39d353" />
    <img src="https://img.shields.io/badge/Surprise_Lib-0d1117?style=flat-square&logo=python&logoColor=39d353" />
  </p>
</div>

---

## Sobre o Projeto

O **CineAI Hub** é um motor de recomendação avançado desenvolvido sobre o dataset *MovieLens Latest Small*. Diferente de abordagens simples, este projeto implementa uma **Arquitetura Híbrida em Camadas**:

1.  **Filtragem Colaborativa:** Uso de **SVD** (Singular Value Decomposition) para capturar vetores latentes de usuários e itens, e **KNN** para vizinhança.
2.  **Conteúdo & Contexto:** Processamento de Tags com **TF-IDF**, Gêneros (One-Hot) e métricas temporais.
3.  **Meta-Modelagem:** Um **Random Forest Regressor** que aprende a ponderar os vetores latentes do SVD junto com as features de conteúdo para prever a nota final.

O resultado é servido via uma aplicação web interativa em **Streamlit**, capaz de resolver o problema de *Cold Start* para novos usuários e gerar recomendações personalizadas para usuários existentes.

---

## Estrutura Modular

Abaixo, a arquitetura dos scripts principais do pipeline:

| Componente | Arquivo | Descrição Técnica |
| :--- | :--- | :--- |
| **ETL & Features** | `src/data_prep.py` | Limpeza, One-Hot Encoding de gêneros, TF-IDF de tags e normalização MinMax. |
| **Pipeline** | `main.py` | Orquestrador CLI que gerencia o fluxo de pré-processamento e retreinamento. |
| **Modelagem KNN** | `notebooks/train_knn.py` | Treina e serializa modelos KNN (User-Based & Item-Based) com GridSearch. |
| **Modelagem Híbrida** | `notebooks/SVDeRF/*.py` | Treina o SVD, extrai vetores latentes e alimenta o Random Forest (`HybridRecommenderSystem`). |
| **Frontend** | `streamlit/app.py` | Interface Web Dark Mode para interação com o usuário (Cold Start & Perfil). |
| **Backend App** | `streamlit/backend_app.py` | Camada de inferência que carrega os `.pkl` e conecta com a API do TMDB (Posters). |

---

## Tech Stack

<div align="center">

| Categoria | Tecnologias |
| :--- | :--- |
| **Core** | `Python` `Pandas` `Numpy` |
| **ML & Stats** | `Scikit-Learn` `Scikit-Surprise` `RandomForest` |
| **Viz & Web** | `Streamlit` `Seaborn` `Matplotlib` `WordCloud` |
| **Dados** | `Parquet` `TMDB API` |

</div>

---

## Como Executar

### 1. Preparação do Ambiente
Certifique-se de ter o Python 3.8+ instalado.

```bash
# Clone o repositório
git clone [https://github.com/alocinny/film-recomender.git](https://github.com/alocinny/film-recomender.git)
cd film-recomender

# Crie e ative o ambiente virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Pipeline de Dados (Treinamento)
Se for a primeira vez, é necessário processar os dados e treinar os modelos. Os arquivos serão salvos em models/.
```bash
# Execute o orquestrador e siga as instruções no terminal (s/n)
python main.py
```

### 3. Executando a Aplicação Web
Com os modelos treinados (.pkl gerados), suba o servidor Streamlit:
```bash
streamlit run streamlit/app.py
``` 

<div align="center"> <sub>Built with 💀 by alocinny, michellydarquia, Danielle-sn</sub> </div>
