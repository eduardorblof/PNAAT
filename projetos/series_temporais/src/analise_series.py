import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing


def analisar():

    df = pd.read_csv(
        'data/leituras_sensores.csv',
        index_col='Data e Hora',
        parse_dates=True
    )

    df = df.asfreq('h')

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Temperatura (°C)'])
    plt.title('Leituras de Temperatura do Motor')
    plt.xlabel('Data')
    plt.ylabel('Temperatura (°C)')
    plt.grid(True)
    plt.show()

    decomposition = seasonal_decompose(
        df['Temperatura (°C)'],
        model='additive',
        period=24
    )

    fig = decomposition.plot()
    fig.set_size_inches(12,8)
    plt.show()

    model = ExponentialSmoothing(
        df['Temperatura (°C)'],
        trend='add',
        seasonal='add',
        seasonal_periods=24
    )

    fit_model = model.fit()

    forecast = fit_model.forecast(48)

    plt.figure(figsize=(12,6))
    plt.plot(df.tail(5*24)['Temperatura (°C)'], label='Histórico')
    plt.plot(forecast, label='Previsão', linestyle='--')

    plt.legend()
    plt.grid(True)
    plt.show()

    print("Previsões:")
    print(forecast)


if __name__ == "__main__":
    analisar()