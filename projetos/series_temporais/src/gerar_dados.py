import pandas as pd
import numpy as np

def gerar_dados():
    
    date_rng = pd.date_range(start='2024-08-01', end='2025-08-31', freq='h')

    trend = np.linspace(30, 35, len(date_rng))
    seasonality = np.sin(np.arange(len(date_rng)) * 2 * np.pi / 24) * 2
    noise = np.random.normal(0, 0.5, len(date_rng))

    temperatura = trend + seasonality + noise

    df_temp = pd.DataFrame({'Temperatura (°C)': temperatura}, index=date_rng)
    df_temp.index.name = 'Data e Hora'

    df_temp.to_csv('data/leituras_sensores.csv')

    print("CSV gerado com sucesso")

if __name__ == "__main__":
    gerar_dados()