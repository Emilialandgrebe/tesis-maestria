import yfinance as yf
import pandas as pd
import os

# Descargar Chicago
ticker = "ZS=F"
data = yf.download(ticker, start="2005-01-01")

# Crear carpeta Data si no existe
os.makedirs("Data", exist_ok=True)

# Convertir a USD por tonelada
BUSHELS_TO_TON = 36.74
data["Price_USD_Ton"] = data["Close"] * BUSHELS_TO_TON

# Supongamos retención actual 33%
retencion = 0.33
data["Precio_Interno_Teorico_USD"] = data["Price_USD_Ton"] * (1 - retencion)

# Guardar
data.to_csv("Data/soybean_internal_price_teorico.csv")

print("Descarga y transformación completa")
print(data[["Close", "Price_USD_Ton", "Precio_Interno_Teorico_USD"]].head())

