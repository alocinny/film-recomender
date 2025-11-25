from src import data_prep
from notebooks import train_knn
import notebooks.SVDeRF.train_svd as train_svd 


def main():

    print("========================================")
    print(" PIPELINE DE SISTEMA DE RECOMENDAÇÃO")
    print("========================================")
    
    # 1. Processamento (Sua Etapa 1 Refatorada)
    print("\n[ETAPA 1] EDA e Pré-processamento")
    step1 = input("Deseja rodar o Pré-processamento (gerar parquets)? (s/n): ").lower()
    if step1 == 's':
        data_prep.run_pipeline()
    
    # 2. Treinamento
    print("\n[ETAPA 2] Treinamento de Modelos")
    step2 = input("Deseja rodar o Treinamento? (s/n): ").lower()
    if step2 == 's':
        
        train_knn.run()
        train_svd.run()

    print("\n✅ Processo finalizado. Para ver os gráficos, abra 'notebooks/01_eda.ipynb'.")
    print("Para ver o app, rode 'streamlit run app.py'")
    
    

if __name__ == "__main__":
    main()
