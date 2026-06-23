# Tesis de Maestría — Simulación de Rendimientos de Pistacho
### Jocolí, Lavalle, Mendoza · Horizonte 20 años · 25 hectáreas

Modela la incertidumbre en los rendimientos de un cultivo de pistacho (variedad Kerman)
combinando técnicas de simulación probabilística calibradas con datos climáticos históricos.

---

## Notebooks

| Módulo | Descripción | Colab |
|--------|-------------|-------|
| 00 — Clima histórico | Descarga y análisis de 35 años de datos climáticos de Jocolí (Open-Meteo). Calibra horas de frío, heladas tardías, calor estival y déficit hídrico. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/00_clima_historico.ipynb) |
| 01 — Monte Carlo | Simulación de 10.000 escenarios de rendimiento e ingresos a 20 años. Variables: vecería, horas de frío, precio de mercado, falla de plantas. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/01_monte_carlo.ipynb) |

---

## Estructura del proyecto

```
tesis-maestria/
├── notebooks/
│   ├── 00_clima_historico.ipynb
│   └── 01_monte_carlo.ipynb
├── src/
│   ├── data/
│   │   ├── climate_fetcher.py    # descarga Open-Meteo API
│   │   └── climate_features.py  # horas de frío, heladas, déficit hídrico
│   └── monte_carlo.py           # simulación de rendimientos e ingresos
├── data/
│   └── raw/                     # datos descargados (generados al correr)
└── requirements.txt
```

## Stack tecnológico

Python 3.11 · NumPy · Pandas · SciPy · Matplotlib · Requests · PyArrow
