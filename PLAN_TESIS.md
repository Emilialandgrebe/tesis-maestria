# Plan de implementación — Tesis de Maestría
## Aptitud agroclimática y modelo de negocio del pistacho en Jocolí, Mendoza
**Universidad Austral · Maestría en Ciencia de Datos**

---

## (guia para seguir y  no olvidarme ideas!)

### Módulo 0 — Análisis climático histórico (`notebooks/00_clima_historico.ipynb`)

| Análisis | Estado | Notas |
|---|---|---|
| Descarga y caché ERA5-Land (Open-Meteo) | ✅ | `src/data/climate_fetcher.py` |
| Temperatura máxima media verano (ene–feb) | ✅ | Mann-Kendall + Sen's Slope + LOWESS + IC 95% |
| Frecuencia días > 35/38/40 °C | ✅ | |
| Boxplots de Tmax por década | ✅ | |
| Anomalías respecto al período base 1990–2020 | ✅ | Sen's Slope + OMM |
| GDD agosto–octubre (brotación) | ✅ | Base 10 °C, temperatura media horaria |
| Déficit hídrico anual (ET₀ − P) | ✅ | Mann-Kendall + Sen's Slope |
| Índice de aridez de De Martonne | ✅ | Clasificación automática |
| Horas de frío (mayo–sep, umbral < 7 °C) | ✅ | Tendencia + umbral crítico 800 hs |
| Validación distribución horas de frío | ✅ | Shapiro-Wilk + AIC (Normal / Log-normal / Gamma) |
| Rachas máximas de calor > 38/40 °C | ✅ | |
| Feature engineering consolidado (`df_features`) | ✅ | 11 variables anuales exportadas a parquet |
| Matriz de correlación entre variables | ✅ | |
| Síntesis e interpretación agronómica | ✅ | Texto + limitaciones explícitas |
| Parámetros calibrados → Módulo 1 | ✅ | Distribución seleccionada por AIC |

### Módulo 1 — Simulación de Monte Carlo (`notebooks/01_monte_carlo.ipynb`)

| Análisis | Estado | Notas |
|---|---|---|
| Monte Carlo básico (10 000 iteraciones, 20 años) | ✅ | |
| Variabilidad por horas de frío (función de transferencia) | ✅ | Pendiente: calibrar con Bayes (ver abajo) |
| Vecería por cadena de Markov (alto/bajo) | ✅ | Pendiente: calibrar parámetros |
| Falla de plantas (distribución Beta) | ✅ | |
| Precio por distribución triangular (3 escenarios) | ✅ | |
| Gráficos percentiles P10/P50/P90 | ✅ | |
| Distribución del VAN al año 20 | ✅ | **⚠ VAN solo de ingresos brutos — sin costos** |
| Comparación de escenarios de precio | ✅ | |

---

## ⚠ Problemas críticos a resolver ANTES de la defensa

### 1. El VAN no es real — falta el módulo de costos
**Problema:** El modelo calcula `VAN = ingresos_brutos × factor_descuento`. Esto sobreestima el retorno real entre 3× y 5×. No es un modelo de negocio, es un modelo de ingresos.

**Lo que hay que agregar en `src/monte_carlo.py`:**
```
Inversión inicial (año 0):  ~USD 18.000/ha
  - Preparación del terreno: ~4.000/ha
  - Plantines + plantación:  ~8.000/ha
  - Sistema de riego:        ~6.000/ha

Costos fijos anuales:       ~USD 1.500/ha
  - Mano de obra permanente
  - Mantenimiento riego
  - Seguros y administración

Costos variables (desde año 6): ~USD 800/ha
  - Cosecha y procesamiento: ~400/ha
  - Insumos (fertilizantes, fitosanitarios): ~400/ha
```

**Resultado esperado:** VAN del flujo de caja neto, tasa de retorno interna (TIR), período de recupero.

---

### 2. Sin análisis de sensibilidad
**Problema:** No se sabe qué variable mueve más el resultado. ¿Es el precio? ¿Las horas de frío? ¿La vecería?

**Lo que hay que agregar:**
- **Tornado chart:** varía cada parámetro ±20% mientras los demás se mantienen fijos. Muestra cuál tiene mayor impacto en el VAN.
- **Índices de Sobol (primer orden + total):** estándar en análisis de incertidumbre para simulación. Cuantifica qué fracción de la varianza del VAN se explica por cada variable.
- Librería: `SALib` (ya disponible en pip).

---

### 3. Sin reducción de varianza
**Requerido por el programa de la materia.**

**Variables antitéticas:** en vez de generar `U ~ Uniform(0,1)`, generar pares `(U, 1−U)`. El estimador de Monte Carlo con variables antitéticas tiene menor varianza con la misma cantidad de simulaciones.

```python
# Implementación conceptual en simulate_yields():
u = rng.random((n // 2, T))
u_antitetica = 1 - u
u_total = np.vstack([u, u_antitetica])
horas = stats.norm.ppf(u_total, loc=media, scale=std)
```

---

## Plan de implementación — próximos pasos

### Paso 1 — Módulo de costos (urgente)
**Archivo:** `src/costos.py` (nuevo) + actualizar `src/monte_carlo.py`

```python
@dataclass
class ParametrosCostos:
    inversion_inicial_ha: float = 18_000   # USD/ha
    costo_fijo_ha: float = 1_500           # USD/ha/año
    costo_variable_ha: float = 800         # USD/ha/año (desde año 6)
    año_inicio_costos_variables: int = 6
```

**Resultado:** reemplazar `van_acumulado_usd` por `van_neto_usd` en el DataFrame de salida.

---

### Paso 2 — Variables antitéticas (materia)
**Archivo:** `src/monte_carlo.py` → nueva función `run_monte_carlo_antitetico()`

Comparar la varianza del estimador con y sin variables antitéticas para el mismo `n_simulaciones`.

---

### Paso 3 — Análisis de sensibilidad (materia + tesis)
**Archivo:** nuevo notebook `notebooks/02_sensibilidad.ipynb`

- Tornado chart (variación local, un parámetro a la vez)
- Índices de Sobol S1 y ST (variación global, librería `SALib`)
- Gráfico estándar en análisis de riesgo financiero

---

### Paso 4 — Inferencia Bayesiana con PyMC (materia)
**Archivo:** nuevo notebook `notebooks/03_inferencia_bayesiana.ipynb`

**Objetivo:** calibrar los parámetros más inciertos del modelo usando MCMC (Metropolis-Hastings o NUTS).

**Parámetros a inferir:**
- Los factores de la función de transferencia de frío (`f(800hs)`, `f(1000hs)`)
- Los parámetros de la cadena de Markov de vecería (`p_bajo_si_alto`, `p_alto_si_bajo`)
- Si se consiguen datos de otras fincas: modelo jerárquico (pooling)

**Librería:** `pymc` (≥ 5.0)

---

### Paso 5 — SDE para trayectoria de biomasa (opcional/avanzado)
**Archivo:** `src/sde_biomasa.py` (nuevo)

Modelar el peso del fruto como proceso de Wiener geométrico:
```
dX(t) = μ·X(t)·dt + σ·X(t)·dW(t)
```
- `μ` determinista: depende de GDD (conecta con Módulo 0)
- `σ` estocástico: variabilidad climática

Resolver numéricamente con el esquema de Euler-Maruyama.

**Valor para la tesis:** conecta el análisis climático (GDD) con el rendimiento de manera continua y dinámica, no puntual.

---

### Paso 6 — Break-even y análisis de riesgo (tesis)
- P(VAN < 0) por escenario
- Período de recupero: distribución del año en que el VAN acumulado cruza cero
- Probabilidad de recuperar la inversión antes del año 12 / 15 / 20
- Curvas de nivel: precio mínimo de venta para que el proyecto sea viable dado cada escenario climático

---

## Técnicas del programa de la materia — mapeo

| Técnica del programa | Dónde aplicar | Prioridad |
|---|---|---|
| Monte Carlo + estimación E[g(X)] | `01_monte_carlo.ipynb` ✅ | — |
| Variables antitéticas | Paso 2 | Alta |
| Metropolis-Hastings / MCMC | Paso 4 (`03_inferencia_bayesiana.ipynb`) | Alta |
| Modelo Bayesiano Jerárquico | Paso 4 (si hay datos de otras fincas) | Media |
| SDE + Euler-Maruyama | Paso 5 (`src/sde_biomasa.py`) | Media |
| Optimización Bayesiana (Procesos Gaussianos) | Paso 3 extendido (riego óptimo) | Baja |
| Reinforcement Learning | No recomendado* | — |

*RL requiere un entorno de simulación completo y datos de validación que no existen para este caso.
El riesgo metodológico supera el beneficio para una tesis de Maestría.

---

## Estructura sugerida de la tesis

```
Capítulo 1 — Introducción
  1.1 Contexto del cultivo de pistacho en Argentina
  1.2 Justificación del modelo de negocio
  1.3 Objetivos y alcance

Capítulo 2 — Análisis agroclimático histórico  ← Módulo 0 ✅
  2.1 Datos y fuente (ERA5-Land)
  2.2 Régimen térmico estival
  2.3 Acumulación de frío invernal
  2.4 Balance hídrico y aridez
  2.5 Ingeniería de variables e indicadores derivados
  2.6 Síntesis agronómica

Capítulo 3 — Modelo de simulación probabilística  ← Módulo 1 (extender)
  3.1 Estructura del modelo de Monte Carlo
  3.2 Variables estocásticas y distribuciones
  3.3 Reducción de varianza: variables antitéticas
  3.4 Módulo de costos y VAN neto
  3.5 Análisis de sensibilidad (Sobol)
  3.6 Break-even y análisis de riesgo

Capítulo 4 — Inferencia Bayesiana  ← Módulo 3 (nuevo)
  4.1 Calibración de la función de transferencia de frío
  4.2 Estimación de parámetros de vecería
  4.3 (Opcional) Modelo jerárquico multibloque

Capítulo 5 — (Opcional) SDE para trayectoria de biomasa  ← Módulo avanzado
  5.1 Formulación del proceso estocástico
  5.2 Conexión con datos climáticos (GDD)
  5.3 Distribución de trayectorias de producción

Capítulo 6 — Conclusiones y trabajo futuro
```

---

## Notas de referencia bibliográfica pendiente

Estos números están en el código sin cita — hay que agregarlas antes de la defensa:

| Parámetro | Valor usado | Referencia a buscar |
|---|---|---|
| Umbral horas de frío crítico | 800 hs | Goldhammer (1995), Crane & Takeda (1979) |
| Umbral horas de frío sin penalidad | 1 000 hs | Idem |
| Factores función de transferencia (0.40, 0.70, 1.00) | Ad hoc | Buscar en Ferguson (2006) o Ruiz et al. (2018) |
| Parámetros vecería (0.65, 0.80) | Ad hoc | Polito & Pinney (1999) o datos INTA |
| Temperatura base GDD pistacho | 10 °C | Crane & Takeda (1979) |
| Umbral rango óptimo estival (35–38 °C) | — | Crane & Takeda (1979), Ferguson (2006) |
| Inversión inicial 18 000 USD/ha | Estimación | Plan de negocios INTA o cotizaciones locales |
| Umbral déficit hídrico 600 mm | Plan de negocios | Necesita cita o aclarar que es supuesto del modelo |

---

## Librerías a agregar a `requirements.txt`

```
pymc>=5.0          # Inferencia Bayesiana (Paso 4)
SALib>=1.4         # Análisis de sensibilidad Sobol (Paso 3)
arviz>=0.17        # Visualización de distribuciones posterior (Paso 4)
```

---


