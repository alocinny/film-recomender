# Sistema de Recomendação de Filmes (MovieLens)

Este projeto implementa um sistema de recomendação de filmes utilizando **Filtragem Colaborativa** com o dataset "MovieLens Latest Small". O objetivo é comparar o desempenho dos algoritmos SVD e KNN, otimizá-los com Grid Search, e apresentar as recomendações em uma aplicação web interativa construída com Streamlit.

---

## Sobre o Projeto

O sistema é capaz de:
* Analisar o comportamento de 610 usuários e 9.742 filmes.
* Treinar modelos de recomendação (SVD e KNN) para prever notas.
* Avaliar os modelos usando métricas de erro (RMSE e MAE).
* Gerar recomendações "Top-N" personalizadas para um usuário específico.
* Disponibilizar uma interface web (Streamlit) para a demonstração ao vivo.

---

## Preparação do Ambiente

Siga os passos abaixo para configurar o ambiente de desenvolvimento local.

### 1. Pré-requisitos
* [Python 3.8+](https://www.python.org/downloads/)
* `git` (para clonar o repositório)

### 2. Clonar o Repositório
```bash
# Clone este repositório
git clone https://github.com/alocinny/film-recomender.git

# Entre na pasta do projeto
cd film-recomender
```
### 3. Baixar os Dados (MovieLens Latest Small)
1. Baixe o dataset "ml-latest-small.zip" diretamente do GroupLens: https://grouplens.org/datasets/movielens/latest/
2. Descompacte o arquivo
3. Adicione os arquivos para a pasta data/ do projeto

### 4. Criar e Ativar o Ambiente Virtual
```bash
# Criar um ambiente virtual (ex: 'venv')
py -3.11 -m venv venv

# Ativar o ambiente
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

### 5. Instalar as Dependências
```bash
# Instalar a partir do requirements.txt
pip install -r requirements.txt
```

## Executando o Projeto

### 1. Treinamento (obrigatório na primeira vez)
Os modelos treinados (.pkl) são gerados localmente e não são versionados. Você deve executar os scripts da pasta notebooks/ na ordem correta:
```bash
# 1. Gera EDA, modelos baseline e salva em 'models/'
python notebooks/eda.py
python notebooks/baseline_models.py

# 2. Roda o Grid Search, salva os melhores modelos e os resultados
python notebooks/grid_search.py
```

### 2. Aplicação Streamlit
Com os modelos gerados, inicie o servidor do Streamlit:
```bash
streamlit run app.py
```
O terminal exibirá um URL local. Abra este endereço no seu navegador para interagir com o sistema de recomendação.