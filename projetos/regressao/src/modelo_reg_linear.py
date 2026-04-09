import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def analisar():
    df_loaded = pd.read_csv('data/dados_maquinas.csv')

    X = df_loaded[['Velocidade_RPM', 'Idade_Anos']]
    y = df_loaded['Consumo_Energia_kWh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nErro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2:.2f}")

    nova_maquina = [[4000, 5]]
    consumo_previsto = model.predict(nova_maquina)
    print(f"\nPrevisão de consumo para a nova máquina: {consumo_previsto[0]:.2f} kWh")

    coeficientes = model.coef_
    intercepto = model.intercept_

    print("\nCoeficientes:")
    for nome, valor in zip(['Velocidade_RPM', 'Idade_Anos'], coeficientes):
        print(f"{nome}: {valor:.4f}")
    print(f"Intercepto: {intercepto:.4f}")

    print(f"\nFórmula aproximada do modelo:")
    print(f"Consumo_Energia_kWh = {intercepto:.2f} + ({coeficientes[0]:.2f} * Velocidade_RPM) + ({coeficientes[1]:.2f} * Idade_Anos)")


if __name__ == "__main__":
    analisar()
    