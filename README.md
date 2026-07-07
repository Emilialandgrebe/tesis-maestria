# Simulación de Rendimientos de Pistacho — Jocolí, Mendoza

Tesis de Maestría en Ciencia de Datos — Universidad Austral
**Autora:** Emilia Landgrebe · **Director:** Ezequiel Nuske

Modela la incertidumbre en el rendimiento y la rentabilidad de un cultivo de
pistacho (variedad Kerman) en Jocolí, Lavalle, Mendoza, combinando análisis
agroclimático histórico con simulación de Monte Carlo. Horizonte de 20 años,
superficie configurable.

---

## Notebooks

| Módulo | Descripción | Colab |
|--------|-------------|-------|
| 00 — Clima histórico | Descarga y análisis de 35 años de datos climáticos de Jocolí (Open-Meteo / ERA5-Land). Calibra horas de frío, calor estival, déficit hídrico y heladas tardías. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/00_clima_historico.ipynb) |
| 01 — Monte Carlo | Simulación de 10.000 escenarios de rendimiento, ingresos y VAN neto a 20 años. Variables: vecería, horas de frío, déficit de calor, precio de mercado, falla de plantas, estructura de costos real. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/01_monte_carlo.ipynb) |

## Estado actual

**Implementado:** análisis climático histórico (Módulo 0), simulación de
rendimientos con 5 fuentes de variabilidad, estructura de costos (CAPEX/OPEX)
calibrada con el plan de negocio real, VAN neto, TIR y año de recupero.

**Pendiente:** reducción de varianza (variables antitéticas), análisis de
sensibilidad (Sobol), inferencia bayesiana de parámetros climáticos. El detalle
completo está en `notas/PLAN_TESIS.md`.

## Estructura del proyecto

```
tesis-maestria/
├── notebooks/
│   ├── 00_clima_historico.ipynb
│   └── 01_monte_carlo.ipynb
├── src/
│   ├── data/
│   │   ├── climate_fetcher.py    # descarga y cachea datos de Open-Meteo
│   │   └── climate_features.py   # horas de frío, heladas, calor, déficit hídrico
│   ├── monte_carlo.py            # simulación de rendimientos, ingresos y VAN
│   └── costos.py                 # estructura de CAPEX/OPEX
├── data/
│   ├── raw/                      # datos climáticos descargados (generados al correr)
│   └── external/                 # datos del plan de negocio (CAPEX, OPEX, ver README propio)
├── notas/                        # bitácora de trabajo, no forma parte de la tesis en sí
└── requirements.txt
```

## Stack tecnológico

Python 3.11 · NumPy · Pandas · SciPy · Matplotlib · Requests · PyArrow
