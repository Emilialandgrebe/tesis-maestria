# Notas de tesis — pistacho Jocolí
Universidad Austral · Maestría en Ciencia de Datos

---

## Qué hay hecho

### Módulo 0 — análisis climático histórico

- Descarga de datos ERA5-Land desde Open-Meteo, caché en parquet local
- Temperatura máxima media de verano (enero-febrero), con Mann-Kendall + Sen's Slope + LOWESS + IC 95%
- Frecuencia de días extremos: > 35, > 38 y > 40 °C
- Boxplots de Tmax por décadas y anomalías respecto al período base 1990-2020
- GDD de agosto a octubre (período de brotación), base 10 °C
- Déficit hídrico anual (ET0 - precipitación), con tendencia Mann-Kendall
- Índice de aridez de De Martonne con clasificación automática
- Horas de frío mayo-septiembre, umbral < 7 °C
- Validación de la distribución de horas de frío: Shapiro-Wilk + comparación por AIC entre Normal, Log-normal y Gamma
- Rachas máximas de días consecutivos con Tmax > 38 °C y > 40 °C
- DataFrame `df_features` con 11 indicadores anuales, exportado a parquet
- Matriz de correlación entre todas las variables
- Síntesis agronómica con texto descriptivo en cada sección

### Módulo 1 — simulación de Monte Carlo

- 10.000 iteraciones, horizonte de 20 años
- Cuatro fuentes de variabilidad combinadas:
  - horas de frío con función de transferencia (penalidad si < 800 hs, sin penalidad si >= 1.000 hs)
  - vecería modelada como cadena de Markov binaria (año alto / año bajo)
  - falla de plantas con distribución Beta
  - precio con distribución triangular, tres escenarios (pesimista / base / optimista)
- Gráficos de percentiles P10 / P50 / P90 por año
- Distribución del VAN al año 20 por escenario

---

## Resuelto

### El VAN era solo de ingresos brutos, no un modelo de negocio

Resuelto el 2026-07-04. `src/costos.py` tiene `ParametrosCostos` (CAPEX/OPEX
calibrados por hectárea contra el plan de negocio real, ver `data/external/`).
`run_monte_carlo()` en `src/monte_carlo.py` calcula el flujo de caja neto
(ingresos - OPEX - CAPEX inicial) en vez de ingresos brutos, y `resumen_financiero()`
devuelve VAN neto, TIR y año de recupero por simulación.

### No había análisis de sensibilidad

Resuelto el 2026-07-07. `src/sensibilidad.py` corre índices de Sobol (S1 y ST)
con SALib directamente sobre el simulador real (`run_monte_carlo_antitetico`
+ `resumen_financiero`), por separado para cada escenario de precio (el
parámetro `escenario` es categórico, Sobol necesita variables continuas).
Notebook: `notebooks/03_sensibilidad_sobol.ipynb`. Hallazgo principal:
`tasa_descuento` domina la varianza del VAN en los tres escenarios; el
segundo lugar (`capex_extra_pct` vs. `hectareas`) depende del escenario de
precio.

### No había reducción de varianza

Resuelto el 2026-07-07. `run_monte_carlo_antitetico()` en `src/monte_carlo.py`
genera cada fuente de variabilidad (vecería, horas de frío, calor,
supervivencia, precio) transformando uniformes antitéticos vía la PPF de la
distribución correspondiente. Verificado: 69,3% de reducción de varianza del
estimador (no de la varianza "pooled" de las muestras, que no es la métrica
correcta — ver la nota metodológica en `notebooks/01_monte_carlo.ipynb`).

### El precio se simulaba triangular e independiente por año, sin memoria

Resuelto el 2026-07-11/12. `simulate_prices()` asumía un sorteo independiente
por año (sin autocorrelación), lo que descarta cualquier persistencia real
del precio de un commodity angosto como el pistacho. Se calibró un AR(1)
sobre los retornos logarítmicos (no sobre el nivel de precio) con datos reales
de FRED (serie `WPU01190106`, PPI mensual, BLS, 1991-2026):

- `src/data/price_fetcher.py` — descarga y cachea el índice mensual (`data/raw/precio_pistacho_fred.parquet`).
- `src/data/price_features.py` — serie anual, retornos log y diagnóstico de autocorrelación.
- `notebooks/04_calibracion_precio.ipynb` — diagnóstico documentado, citable en la tesis.
- `src/precio_estocastico.py` — generador `simulate_prices_ar1()` / `simulate_prices_ar1_antitetico()`.
- `run_monte_carlo_precio_historico()` en `src/monte_carlo.py` — integra el AR(1) al simulador real (`simulate_prices()`/`ESCENARIOS_PRECIO` quedan intactos como referencia/comparación).
- `src/dataset_ml.py` — barre la incertidumbre del drift (`precio_drift_anual`) como parámetro de entrada del dataset ML.

Calibración (serie completa 1991-2026, n_obs=35 retornos anuales): ADF sobre
el nivel no rechaza raíz unitaria (estadístico=-1,0715, p=0,726) — no hay
sustento para modelar reversión a un nivel de precio de largo plazo (se
descartó un Ornstein-Uhlenbeck sobre log(precio)). El AR(1) sobre retornos da
`phi=-0,265423` (p=0,131, no significativo al 5% pero económicamente
relevante por el efecto de composición a 20 años) y `sigma_eps=0,137358`. El
drift (`c`) se dejó en `0,008` (no la estimación puntual de la serie completa,
`0,031127`): el IC95% del drift es amplio (aprox. [-2%, +7%]), así que se usó
una calibración más conservadora sobre la ventana reciente 2010-2026 en vez de
reclamar la precisión que la estimación puntual no tiene.

Se había descartado antes un GBM puro (retornos i.i.d.): el ADF tampoco lo
rechaza, pero componer 20 años de volatilidad sin corrección da colas
económicamente implausibles (P90 del precio año 20 ≈ USD 53/kg en el
escenario optimista). El AR(1) achica esa cola a ≈USD 28/kg con el drift
conservador, manteniendo el ancla por escenario (`pesimista=7.0`,
`base=9.5`, `optimista=13.0` USD/kg — misma moda que `ESCENARIOS_PRECIO`).

Validado además que `P(VAN<0)` sube de 0,64% (precio triangular) a ~17% con
el AR(1): no es una regresión, es la corrección de un sesgo del modelo viejo,
que trataba el riesgo de precio como idiosincrático (se diluye en 15 años de
producción) cuando en realidad es sistemático/persistente.

## Problemas abiertos

### Los parámetros clave no tienen soporte bibliográfico

Algunos números que están en el código son supuestos que hay que citar o justificar:
- Los factores de la función de transferencia de frío (0.40, 0.70, 1.00) — buscar en Ferguson (2006) o Ruiz et al. (2018)
- Los parámetros de la cadena de Markov de vecería (p_bajo_si_alto = 0.65, p_alto_si_bajo = 0.80) — buscar en Polito & Pinney (1999) o datos INTA
- La temperatura base GDD = 10 °C — Crane & Takeda (1979)
- El umbral de 800 hs como crítico y 1.000 hs como óptimo — Goldhammer (1995)
- La inversión inicial de USD 18.000/ha — necesita cotización real o fuente de INTA / plan de negocios

### Inferencia bayesiana pendiente

Ver Paso 4 más abajo — reemplazar los supuestos ad hoc de frío/vecería por estimaciones con MCMC (PyMC).

### Dataset sintético para ML pendiente

Barrer el espacio de parámetros con `simulate_yields()` para generar un dataset sintético de entrenamiento (capa 3 de la arquitectura de datos del proyecto). Todavía no existe ningún archivo para esto en el repo.

---

## Plan de implementación

### Paso 1 — módulo de costos (empezar por acá)

**Estado: completado (2026-07-04).**

Crear `src/costos.py` con una dataclass `ParametrosCostos` y actualizar `src/monte_carlo.py` para que el VAN se calcule sobre el flujo de caja neto, no sobre los ingresos brutos.

El flujo de cada año sería:
- año 0: egresa la inversión inicial
- años 1 a 5: solo costos fijos (no hay producción todavía)
- años 6 a 20: ingresos - costos fijos - costos variables

El resultado esperado es tener `van_neto_usd` en lugar de `van_acumulado_usd`, más la TIR y el año de recupero.

### Paso 2 — variables antitéticas

**Estado: completado (2026-07-07).**

En `src/monte_carlo.py` agregar una función `run_monte_carlo_antitetico()`. La idea es simple: generar la mitad de las simulaciones con U y la otra mitad con 1-U, apilarlas y comparar la varianza del estimador resultante contra el Monte Carlo estándar con el mismo N.

### Paso 3 — análisis de sensibilidad

**Estado: completado (2026-07-07).** El notebook terminó llamándose
`notebooks/03_sensibilidad_sobol.ipynb` (no `02_sensibilidad.ipynb` como decía
originalmente acá) y por ahora solo tiene los índices de Sobol, sin tornado
chart local de un parámetro a la vez — el tornado chart que sí se hizo es de
los índices ST de Sobol ordenados, que cumple el mismo propósito visual.

Nuevo notebook `notebooks/02_sensibilidad.ipynb`:
- primero el tornado chart (análisis local, un parámetro a la vez)
- después los índices de Sobol con SALib (análisis global, captura interacciones entre variables)

### Paso 4 — inferencia Bayesiana con PyMC

Nuevo notebook `notebooks/03_inferencia_bayesiana.ipynb`.

El objetivo es reemplazar los supuestos ad hoc de los parámetros más inciertos por estimaciones con MCMC. Las candidatas:
- los tres puntos de la función de transferencia de frío
- los parámetros de transición de la cadena de Markov de vecería

Si se consiguen datos de más de una finca, se puede hacer un modelo jerárquico (pooling parcial). Eso sería el punto más sólido de la tesis desde el lado de Ciencia de Datos. Librería: pymc >= 5.0

### Paso 5 — SDE para trayectoria de biomasa (opcional, si da el tiempo)

Modelar el crecimiento del fruto como un proceso de Wiener geométrico donde la deriva depende del GDD calculado en el Módulo 0. Se resuelve numéricamente con el esquema de Euler-Maruyama. Conecta el análisis climático con el rendimiento de forma continua en lugar de puntual.

### Paso 6 — break-even y análisis de riesgo

- Probabilidad de VAN < 0 por escenario
- Distribución del año en que el VAN acumulado cruza cero
- Probabilidad de recuperar la inversión antes del año 12, 15 y 20
- Precio mínimo de venta para que el proyecto sea viable dado un escenario climático específico

---

## Estructura de capítulos (borrador)

- Capítulo 1: introducción, contexto del cultivo en Argentina, justificación del modelo de negocio, objetivos
- Capítulo 2: análisis agroclimático histórico (Módulo 0, ya hecho)
  - datos y fuente ERA5-Land
  - régimen térmico estival
  - acumulación de frío invernal
  - balance hídrico y aridez
  - ingeniería de variables
  - síntesis agronómica
- Capítulo 3: modelo de simulación probabilística (Módulo 1, extender)
  - estructura del Monte Carlo
  - variables estocásticas y distribuciones
  - modelo de precio con memoria: AR(1) sobre retornos log, datos y fuente FRED (serie `WPU01190106`, PPI mensual, BLS)
  - reducción de varianza con variables antitéticas
  - módulo de costos y VAN neto
  - análisis de sensibilidad con índices de Sobol
  - break-even y análisis de riesgo
- Capítulo 4: inferencia Bayesiana (Módulo 3, nuevo)
  - calibración de la función de transferencia de frío
  - estimación de parámetros de vecería
  - modelo jerárquico si se tienen datos de otras fincas
- Capítulo 5 (opcional): SDE para trayectoria de biomasa
- Capítulo 6: conclusiones y trabajo futuro

---

## Técnicas del programa de la materia y dónde van

- Monte Carlo + estimación E[g(X)] — ya está en el Módulo 1
- Variables antitéticas — Paso 2
- Metropolis-Hastings / MCMC — Paso 4
- Modelo Bayesiano Jerárquico — Paso 4, si hay datos de otras fincas
- SDE + Euler-Maruyama — Paso 5
- Optimización Bayesiana con Procesos Gaussianos — no prioritario, si sobra tiempo
- Reinforcement Learning — no recomendado para esta tesis, el riesgo metodológico es alto sin datos de validación

---

## Librerías que hay que agregar a requirements.txt

- pymc >= 5.0 — inferencia Bayesiana
- arviz >= 0.17 — visualización de distribuciones posterior

SALib ya está agregado (`SALib==1.4.8`, fijado a esa versión porque 1.5.x exige `numpy>=2.0`).

---

## Cosas que no hay que olvidar

- Agregar el .gitignore (hay un archivo sin commitear)
- Citar todas las referencias bibliográficas de los parámetros antes de la defensa (ver sección de problemas)
- Aclarar en el texto de la tesis que el VAN no incluye impuestos ni financiamiento
- La curva de producción por año (del 1 al 20) también necesita una referencia o aclarar que es un supuesto del modelo
- CAPEX parcialmente cotizado: de 30 ítems en data/external/capex.csv, 16
  tienen fuente real (columna costeado=SI, suman ~USD 1.129.013) y 14 siguen
  sin cotizar (costeado=NO). Los ítems faltantes se completan con estimaciones
  web de distinta confianza en data/external/capex_estimaciones_web.csv
  (algunas "media", otras "baja"/"muy baja", una directamente "sin_dato" -- la
  represa/cisterna). Los VAN citados en este documento y en el repo son
  PROVISORIOS hasta cerrar la cotización completa. `capex_extra_pct` (barrido
  en Sobol y en el dataset ML) existe justamente para representar esta
  incertidumbre, pero no reemplaza tener el dato real. Antes de escribir
  números finales en el documento de tesis (no solo en el repo), correr de
  nuevo el pipeline completo (dataset_ml.py + entrenar_modelo.py) con el CAPEX
  cerrado.

---

Última actualización: 2026-07-12
