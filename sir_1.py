


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Cargar y preparar los datos de S&P 500
activo = "^GSPC"
inicio = "2022-01-01"
final = "2023-12-30"
df = yf.download(activo, start=inicio, end=final)

# Verificar si los datos se cargaron correctamente
print(df.head())

# Calcular las diferencias de cierre
df['diff'] = df['Close'].diff()

# Identificar picos y valles
df['peak'] = (df['diff'] > 0) & (df['diff'].shift(-1) < 0)
df['trough'] = (df['diff'] < 0) & (df['diff'].shift(-1) > 0)

# Calcular On-Balance Volume (OBV)
df['OBV'] = 0
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
        df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] + df['Volume'].iloc[i]
    elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
        df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV'] - df['Volume'].iloc[i]
    else:
        df.loc[df.index[i], 'OBV'] = df.loc[df.index[i - 1], 'OBV']

# Obtener los picos y valles
picos = df[df['peak']]
valles = df[df['trough']]

# Visualizar precios y picos/valles
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Precio de cierre')
plt.scatter(picos.index, picos['Close'], color='red', label='Picos', alpha=0.6)
plt.scatter(valles.index, valles['Close'], color='green', label='Valles', alpha=0.6)
plt.title('Detección de Picos y Valles en el S&P 500')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()

# Visualizar OBV
plt.figure(figsize=(14, 7))
plt.plot(df['OBV'], label='On-Balance Volume', color='purple')
plt.title('On-Balance Volume (OBV) del S&P 500')
plt.xlabel('Fecha')
plt.ylabel('OBV')
plt.legend()
plt.show()

# Verificar resultados
print("Picos detectados:")
print(picos[['Close']])
print("Valles detectados:")
print(valles[['Close']])
print("Datos de volumen y OBV:")
print(df[['Volume', 'OBV']].tail())
