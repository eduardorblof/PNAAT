import pandas as pd
import numpy as np

def gerar_dados():
    num_amostras = 200

    velocidade_rpm = np.random.uniform(1000, 5000, num_amostras)

    idade_anos = np.random.uniform(0, 10, num_amostras)

    consumo_energia = 20 + (velocidade_rpm * 0.05) + (idade_anos * 2.5) + np.random.normal(0, 5, num_amostras)

    df_maquinas = pd.DataFrame({
        'Velocidade_RPM': velocidade_rpm,
        'Idade_Anos': idade_anos,
        'Consumo_Energia_kWh': consumo_energia
    })

    df_maquinas.to_csv('data/dados_maquinas.csv', index=False)

    print("Arquivo 'dados_maquinas.csv' gerado com sucesso.")
    print(df_maquinas.head())


if __name__ == "__main__":
    gerar_dados()