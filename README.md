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
| 01 — Monte Carlo | Simulación de 10.000 escenarios de rendimiento, ingresos y VAN neto a 20 años. Variables: vecería, horas de frío, déficit de calor, precio de mercado, falla de plantas, estructura de costos real. Reducción de varianza por variables antitéticas. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/01_monte_carlo.ipynb) |
| 03 — Sensibilidad Sobol | Análisis de sensibilidad global de Sobol sobre el simulador real, corrido por separado para cada escenario de precio (pesimista/base/optimista). Identifica qué variable explica más la varianza del VAN. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/03_sensibilidad_sobol.ipynb) |
| 04 — Calibración de precio | Diagnóstico de autocorrelación del precio del pistacho (FRED, serie WPU01190106, PPI mensual 1991-2026) y calibración del modelo AR(1) sobre retornos log que reemplaza el precio triangular independiente. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emilialandgrebe/tesis-maestria/blob/main/notebooks/04_calibracion_precio.ipynb) |

## Estado actual

**Implementado:** análisis climático histórico (Módulo 0); simulación de
rendimientos con 5 fuentes de variabilidad, recalibrada contra el rendimiento
validado real (rendimiento medio ~2.990 kg/ha en plena producción, calibrado
contra el dato validado en `data/external/produccion_ingresos_plan.csv`, ver
`RENDIMIENTO_VALIDADO_KG_HA`/`RENDIMIENTO_IDEAL_KG_HA` en `src/monte_carlo.py`);
estructura de costos (CAPEX/OPEX) calibrada con el plan de negocio real; VAN
neto, TIR y año de recupero; reducción de varianza por variables antitéticas
(`run_monte_carlo_antitetico()`, ~69% de reducción de varianza del estimador
verificada); análisis de sensibilidad global de Sobol sobre el simulador real
(`src/sensibilidad.py`, notebook 03); modelo de precio con memoria calibrado
con datos reales de FRED (AR(1) sobre retornos log, `src/precio_estocastico.py`,
notebook 04); dataset sintético para ML vía Latin Hypercube Sampling sobre 8
parámetros, con targets de VAN medio y probabilidad de VAN negativo
(`src/dataset_ml.py`, `data/processed/dataset_ml_train.parquet`).

**Pendiente:** inferencia bayesiana de parámetros climáticos (PyMC); entrenar
el primer modelo de ML sobre el dataset sintético ya generado. El detalle
completo está en `notas/PLAN_TESIS.md`.

## Estructura del proyecto

```
tesis-maestria/
├── notebooks/
│   ├── 00_clima_historico.ipynb
│   ├── 01_monte_carlo.ipynb
│   ├── 03_sensibilidad_sobol.ipynb
│   └── 04_calibracion_precio.ipynb
├── src/
│   ├── data/
│   │   ├── climate_fetcher.py    # descarga y cachea datos de Open-Meteo
│   │   ├── climate_features.py   # horas de frío, heladas, calor, déficit hídrico
│   │   ├── price_fetcher.py      # descarga y cachea el índice de precio de FRED
│   │   └── price_features.py     # serie anual, retornos log y diagnóstico de autocorrelación
│   ├── monte_carlo.py            # simulación de rendimientos, ingresos y VAN
│   ├── costos.py                 # estructura de CAPEX/OPEX
│   ├── sensibilidad.py           # análisis de sensibilidad de Sobol (SALib)
│   ├── precio_estocastico.py     # generador de precio con memoria (AR(1) sobre retornos)
│   └── dataset_ml.py             # dataset sintético para ML vía Latin Hypercube Sampling
├── data/
│   ├── raw/                      # datos climáticos descargados (generados al correr)
│   ├── external/                 # datos del plan de negocio (CAPEX, OPEX, ver README propio)
│   └── processed/                # índices de Sobol y otros derivados (generados al correr)
├── notas/                        # bitácora de trabajo, no forma parte de la tesis en sí
└── requirements.txt
```

## Stack tecnológico

Python 3.11 · NumPy · Pandas · SciPy · Matplotlib · Requests · PyArrow · SALib
