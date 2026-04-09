import numpy as np
import pandas as pd

print("\n--- 2.1:Criando um DataFrame a partir de dados de sensores ---")
data = {
    'Timestamp': pd.to_datetime(pd.Series(range(10)), unit='s', origin='2025-08-07'),
    'Temperatura_C': np.random.uniform(25, 30, 10),
    'Umidade_Relativa': np.random.uniform(40, 60, 10),
    'Vibracao_Hz': np.random.uniform(0.1, 1.5, 10)
}

df_fabrica = pd.DataFrame(data)
print(df_fabrica)

print("\n--- 2.2: Filtrando e Analisando Dados ---")

temperatura_filtrada = df_fabrica['Temperatura_C']
print("\nTemperaturas Registradas:")
print(temperatura_filtrada)

anomalias_vibracao = df_fabrica[df_fabrica['Vibracao_Hz'] > 1.0]
print("\nAnomalias de Vibração (Vibração > 1.0 Hz):")
if not anomalias_vibracao.empty:
    print(anomalias_vibracao)
else:
    print("Nenhuma anomalia de vibração encontrada.")
if not anomalias_vibracao.empty:
    media_temp_anomalias = anomalias_vibracao['Temperatura_C'].mean()
    print(f"\nMédia de Temperatura nas Anomalias de Vibração: {media_temp_anomalias:.2f} °C")

novos_dados_dicionario = {
    'Maquina_ID': ['M_A1', 'M_A2', 'M_A3', 'M_A4', 'M_A5'],
    'Tempeatura_C': [28.5, 31.0, 27.8, 32.5, 29.1],
    'Corrente_A': [5.1, 8.9, 4.5, 9.2, 6.7]
}

df_novas_maquinas = pd.DataFrame(novos_dados_dicionario)
print("\nNovos Dados de Máquinas:")
print(df_novas_maquinas)

amplitude_temp = df_novas_maquinas['Tempeatura_C'].max() - df_novas_maquinas['Tempeatura_C'].min()
print(f"\nAmplitude de Temperatura nas Novas Máquinas: {amplitude_temp:.2f} °C")  

print("\nResultados de um novo filtro (Umidade < 50)")
df_umidade_baixa = df_fabrica[df_fabrica['Umidade_Relativa'] < 50]
print(df_umidade_baixa)

dados_personalizados = {
    'Pressao_hPa': [1010, 1012, 1009, 1015, 1011],
    'Ruido_dB': [55, 62, 58, 70, 60]
}
df_personalizado = pd.DataFrame(dados_personalizados)
print("\nDados Personalizados:")
print(df_personalizado)

