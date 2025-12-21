import numpy as np

class HybridRecommenderSystem:
    def __init__(self, svd_model, rf_model, trainset, catalog_vectors, catalog_content):
        self.svd = svd_model
        self.rf = rf_model
        self.trainset = trainset
        self.catalog_vectors = catalog_vectors # Dict {movieId: np.array}
        self.catalog_content = catalog_content # DataFrame
        self.n_factors = svd_model.n_factors

    def predict(self, uid, iid):
        """
        Método mágico: Recebe ID do Usuário e ID do Filme e retorna a nota prevista.
        Simula o comportamento do .predict() que você queria.
        """
        # 1. Verifica se o filme existe no catálogo
        if iid not in self.catalog_content.index:
            return 0.0 # Filme desconhecido

        # 2. Pega o vetor do Usuário (SVD)
        try:
            inner_uid = self.trainset.to_inner_uid(uid)
            u_vec = self.svd.pu[inner_uid]
        except ValueError:
            # Usuário novo (Cold Start)
            u_vec = np.zeros(self.n_factors)

        # 3. Pega o vetor do Item (Cacheado)
        i_vec = self.catalog_vectors.get(iid, np.zeros(self.n_factors))

        # 4. Pega Features de Conteúdo
        content_feats = self.catalog_content.loc[iid].values

        # 5. Junta tudo num vetor só (User + Item + Content)
        # O reshape(1, -1) é necessário porque o Random Forest espera uma matriz 2D
        final_vec = np.concatenate([u_vec, i_vec, content_feats]).reshape(1, -1)

        # 6. Previsão final
        prediction = self.rf.predict(final_vec)[0]
        return prediction
    
