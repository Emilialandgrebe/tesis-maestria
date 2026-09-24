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

### No había dataset sintético para ML

Resuelto el 2026-07-11 (commit `06b7472`). `src/dataset_ml.py` barre el
espacio de parámetros con Latin Hypercube Sampling (8 parámetros continuos
de `ESPACIO_PARAMETROS` × 3 escenarios de precio) y evalúa el simulador real
en cada punto (`run_monte_carlo_precio_historico` + `resumen_financiero`),
generando un dataset para *emular* el simulador (surrogate modeling), no para
predecir un rendimiento agroclimático observado. Salida en
`data/processed/dataset_ml_{train,test}.parquet` (2400 y 600 filas × 19
columnas; train y test son diseños LHS independientes, semillas 42 y 2026).
Targets: `van_neto_medio_usd` (regresión) y `prob_van_negativo` (riesgo).
`src/entrenar_modelo.py` entrena Random Forest y LightGBM sobre él (commit
`d2d2474`). Última regeneración: 2026-08-28 (Fase 3, ver la entrada de esa
fecha más abajo) — sin cambios numéricos.

### Superficie por defecto seguía en 25 ha y el CAPEX/OPEX eran 100% determinísticos dentro del Monte Carlo

Resuelto el 2026-08-26. Dos cambios relacionados sobre `src/monte_carlo.py` y
`src/costos.py`:

**1. Hectáreas por defecto: 25 → 50 ha.** `HECTAREAS` en `src/monte_carlo.py`
y el default de `hectareas` en `ParametrosCostos` (`src/costos.py`) pasan a
50.0, como parámetro libre de punta a punta (`ParametrosMC.hectareas` sigue
sincronizando automáticamente a `ParametrosCostos.hectareas` vía
`_resolver_params_costos()`, sin cambios ahí). Se confirmó que no queda
ningún otro número hardcodeado a 25 ha en el motor: el único otro `25` en
`src/` es el rango inferior del barrido LHS de `hectareas` en
`dataset_ml.py` (`(25.0, 100.0)`), que es un límite de sweep, no un default,
y queda fuera del alcance de este cambio. `notebooks/01_monte_carlo.ipynb`
ahora imprime `params.hectareas` explícitamente y aclara en la intro que la
escala por defecto es 50 ha.

**2. CAPEX y OPEX ahora tienen una fuente de variabilidad estocástica dentro
de cada iteración del Monte Carlo** (antes, la única variabilidad de costos
en todo el repo era el barrido `capex_extra_pct` de `dataset_ml.py`, fijo
por punto del diseño LHS, no ruido dentro de las 10.000 iteraciones que
arman la distribución del VAN):

- CAPEX: se separó `data/external/capex.csv` en un componente FIJO
  (`capex_inicial_ha`, ítems con `costeado=SI`, sin cambios) y uno
  ESTOCÁSTICO nuevo (`simulate_capex_extra()`/`simulate_capex_extra_antitetico()`
  en `src/monte_carlo.py`): dos triangulares independientes, riego
  (USD 2.800/3.250/3.500 por ha) y pozo de agua (USD 800/1.320/1.600 por
  ha), calibradas con los rangos de mercado investigados en
  `data/external/capex_estimaciones_web.csv` (confianza "media" y "baja"
  respectivamente). Se simulan por separado y se suman, en vez de aproximar
  la suma con una triangular equivalente por momentos — más simple de
  mantener y sin el error de encajar esa suma en una forma que no le
  corresponde. Desmonte/nivelación/subsolado (solo estimación puntual, sin
  rango, confianza "muy baja"), represa/cisterna y paneles solares
  (confianza "sin_dato"/sin estimar) quedan explícitamente FUERA de esta
  distribución: son riesgo real no cuantificado, no ruido modelable —
  ponerles un rango inventado habría disfrazado de precisión modelada algo
  que en realidad es falta de dato. El CAPEX total simulado sigue siendo un
  piso, no el costo total esperado del proyecto.
- OPEX: no hay ningún rango investigado todavía para OPEX pre-productivo ni
  productivo (a diferencia del CAPEX, donde sí hay cotizaciones de mercado).
  Se agregó `simulate_opex_multiplicador()`/`simulate_opex_multiplicador_antitetico()`:
  una triangular simétrica ±15% (`ParametrosCostos.opex_variacion_pct`),
  con el mismo criterio que `precision_factor_frio`/`precision_factor_calor`
  en `ParametrosMC` — un punto de partida razonable y declarado como tal,
  pendiente de calibrar con datos reales de costos operativos de otras
  fincas, no una afirmación empírica.
- Implementación: un solo draw de CAPEX extra por simulación (el CAPEX
  ocurre una vez) y un solo multiplicador de OPEX por simulación (aplicado
  a todos los años — la incertidumbre es "este proyecto resultó más/menos
  caro de lo presupuestado", no ruido año a año). Ambas fuentes tienen
  versión antitética siguiendo el mismo patrón de
  `_generar_uniformes_antiteticos()` que ya usan frío/calor/precio, así que
  `run_monte_carlo_antitetico()` y `run_monte_carlo_precio_historico()`
  siguen teniendo reducción de varianza también en esta fuente nueva
  (verificado: ~70,8% de reducción de varianza del estimador con la nueva
  fuente incluida, vs. ~69,3% documentado antes solo con frío/calor/precio).
  No se rompió la firma pública de `run_monte_carlo()` /
  `run_monte_carlo_antitetico()` / `run_monte_carlo_precio_historico()`; se
  agregó una columna nueva (`capex_extra_estocastico_usd`), sin tocar las
  existentes. `resumen_financiero()` se actualizó para sumar
  `costos.capex_inicial` (fijo) + esa columna nueva (estocástico) al
  calcular TIR y año de recupero — antes solo usaba el fijo, lo cual habría
  quedado inconsistente con el `van_neto_usd` ya calculado con ambos
  componentes.

Resultado a 50 ha (10.000 simulaciones, escenario base, semilla 42), sobre
`van_neto_usd` al año 20:

| | Antes (25 ha, costos deterministas) | Después (50 ha, costos deterministas) | Después (50 ha, CAPEX/OPEX estocásticos) |
|---|---|---|---|
| Media | USD 747.916 | USD 1.267.333 | USD 1.273.871 |
| P10 | USD 393.790 | USD 559.079 | USD 539.401 |
| P50 | USD 758.003 | USD 1.287.506 | USD 1.295.835 |
| P90 | USD 1.090.515 | USD 1.952.529 | USD 1.977.604 |
| P(VAN<0) | 0,48% | 0,65% | 1,72% |

La columna "50 ha, costos deterministas" aísla el efecto de escala puro
(duplicar hectáreas duplica aproximadamente ingresos, CAPEX y OPEX, por eso
el salto grande en todas las métricas). La comparación relevante para esta
tarea es esa columna contra la de la derecha: agregar la variabilidad de
CAPEX/OPEX no mueve mucho la media ni la mediana (como se esperaba: las
triangulares nuevas están centradas en sus valores deterministas previos),
pero SÍ ensancha la cola izquierda — `P(VAN<0)` pasa de 0,65% a 1,72%,
principalmente por la cola alta de riego+pozo (el CAPEX estocástico extra
tiene media USD 221.112 y desvío USD 10.959 a 50 ha) combinada con años de
OPEX más caro de lo presupuestado. Pendiente para el próximo paso: revisar
si `sensibilidad.py`/`dataset_ml.py` deben incorporar estos nuevos
parámetros al barrido de Sobol/LHS (hoy los ignoran, ya que usan
`ParametrosCostos()` con los defaults nuevos pero no los barren como
variables de entrada).

### No había forma de desactivar el CAPEX/OPEX estocástico sin romper `rng.triangular`

Resuelto el 2026-08-26. La entrada anterior (misma fecha, arriba) agregó
CAPEX/OPEX estocásticos pero sin ninguna forma real de compararlos contra el
modelo determinístico: el intento obvio (`opex_variacion_pct=0.0` y
triangulares de CAPEX con `low=mode=high`) fallaba con
`ValueError: left == right`, porque `rng.triangular` de NumPy no acepta
ancho cero.

Se agregó `capex_opex_estocastico: bool = True` a `ParametrosMC`
(`src/monte_carlo.py`) y se propaga a las cuatro funciones de simulación
(`simulate_capex_extra`, `simulate_capex_extra_antitetico`,
`simulate_opex_multiplicador`, `simulate_opex_multiplicador_antitetico`) vía
un parámetro `estocastico`. En `False`, cada función devuelve directamente
`capex_extra=0` / `opex_multiplicador=1.0` para las N iteraciones, sin
llamar a `rng.triangular` ni a la PPF de `triang_dist`. Independientemente
del flag, también se agregaron los helpers `_triangular_o_constante()` /
`_triangular_ppf_o_constante()`: si algún `low`/`high` pasado explícito da
ancho cero (`np.isclose`), devuelven un array constante en `mode` en vez de
fallar — defensivo para el caso en que alguien arme un `ParametrosCostos`
con rangos angostos a mano, no solo para el flag.

**Importante:** `capex_opex_estocastico=False` NO es equivalente al
workaround con épsilon usado antes de esta entrada. El workaround dejaba el
CAPEX de riego+pozo en su valor MODAL (~USD 221.112 a 50 ha) con varianza
casi nula; el flag en `False` saca el CAPEX de riego+pozo del todo
(`capex_extra=0`), reproduciendo el comportamiento previo a que existiera
esta fuente de variabilidad. Por eso los números de "determinista" de esta
tabla no coinciden con los de la entrada anterior — no es un error, son dos
nociones distintas de "determinista".

Resultado a 50 ha, 10.000 simulaciones, semilla 42, **escenario de precio
`base`** (el default de `ParametrosMC.escenario`), sobre `van_neto_usd` al
año 20, usando el flag (sin ningún workaround):

| Métrica | `capex_opex_estocastico=False` | `capex_opex_estocastico=True` |
|---|---|---|
| VAN medio | USD 1.495.833 | USD 1.273.871 |
| VAN P10 | USD 787.579 | USD 539.401 |
| VAN P50 | USD 1.516.006 | USD 1.295.835 |
| VAN P90 | USD 2.181.029 | USD 1.977.604 |
| P(VAN<0) | 0,48% | 1,72% |
| TIR media | 0,1141 | 0,1077 |

La diferencia entre columnas acá es más grande que la de la tabla anterior
porque combina dos efectos: (a) agregar el CAPEX de riego+pozo (que en
`False` es cero, no su modo) y (b) la variabilidad alrededor de ese CAPEX y
del OPEX. `P(VAN<0)` en `False` (0,48%) coincide con el número histórico de
"25 ha, costos deterministas" de la entrada anterior porque en ambos casos
no hay CAPEX de riego/pozo ni variación de OPEX — la única diferencia entre
esos dos es la escala (25 vs. 50 ha), que no cambia `P(VAN<0)` en este
modelo (ver el diagnóstico de `hectareas` en `notebooks/03_sensibilidad_sobol.ipynb`:
CAPEX, OPEX e ingresos escalan linealmente con `hectareas`, y `rendimiento`/
`precio` no dependen de `hectareas` en absoluto). Con la misma semilla, la
secuencia de sorteos de rendimiento y precio es idéntica en ambas corridas
(el consumo de `rng` para esas dos fuentes ocurre antes de tocar
`capex_opex_estocastico`), así que `VAN_50ha = 2 × VAN_25ha` simulación por
simulación cuando no hay CAPEX/OPEX estocástico -- el signo de esa igualdad
no cambia, por eso el 0,48% es exactamente igual, no una coincidencia de
redondeo.

**Validación agregada tras `/code-review ultra`:** `ParametrosCostos.__post_init__()`
ahora rechaza con `ValueError` (mensaje explícito) un `opex_variacion_pct`
fuera de `[0, 1)` (fuera de ese rango la triangular del multiplicador de
OPEX puede sortear un valor <= 0, invirtiendo el signo del OPEX en
`flujo_caja_neto`) y cualquier triangular de CAPEX riego/pozo con
`low > mode` o `mode > high`. Confirmado que los valores actuales (0.15 y
los rangos calibrados desde `capex.csv`/`capex_estimaciones_web.csv`) pasan
la validación sin cambiar ningún resultado de la tabla de arriba.

### Correlación empírica frío/calor: modelada pero apagada por defecto

Resuelto el 2026-08-27. Módulo 0 (`data/processed/features_climaticas.parquet`,
35 años ERA5-Land Jocolí 1990-2024) mostró una correlación de Pearson entre
horas de frío y `tmax_media_verano_c` de **r = -0,2894** (p = 0,0918, n = 35).

**Fundamentación estadística de por qué NO es significativa al 5%:**
- Test t sobre el coeficiente: t(33) = -1,737, p = 0,092 (dos colas) — no
  rechaza H0: r=0 al nivel convencional.
- IC 95% de Fisher (transformación z, arctanh): **[-0,568; 0,049]** — incluye
  el cero con margen amplio, así que el signo del efecto poblacional no está
  determinado por estos datos.
- Potencia con n=35 para detectar un r de esta magnitud (alpha=0,05, dos
  colas): **~39%** — muy por debajo del 80% convencional. Con esta muestra,
  el test tiene más probabilidad de NO detectar el efecto aunque exista que
  de detectarlo.

**Decisión:** se agregó `ParametrosMC.correlacionar_frio_calor: bool = False`
(`src/monte_carlo.py`) con el mismo criterio que `capex_opex_estocastico`:
apagado por defecto porque la evidencia es marginal, disponible para
análisis de robustez explícito.

**BUG DE IMPLEMENTACIÓN CORREGIDO (2026-08-27, detectado por `/code-review
ultra`):** la primera versión de este mecanismo aplicaba la cópula gaussiana
sobre los FACTORES finales `factor_frio`/`factor_calor` — es decir, sobre el
ruido Beta residual que se agrega DESPUÉS de la función de transferencia
climática — en vez de sobre las variables climáticas crudas (`horas_frio`,
`tmax_verano`), que es donde efectivamente se midió `r=-0,2894`. Como la
mayor parte de la varianza de cada factor viene del driver climático y no
del ruido residual, esa versión diluía la correlación lograda a **≈-0,03**
en la práctica (verificado numéricamente, n=50.000), no los -0,29
documentados — un desajuste de capas, no un error de signo ni de fórmula.
Esta sección fue **reescrita**, no solo corregida: el mecanismo cambió de
raíz y toda la validación de robustez de abajo se volvió a correr desde
cero, porque los números de la versión anterior correspondían a un efecto
~10x más débil que el real.

**Mecanismo corregido:** la correlación (`CORR_HORAS_FRIO_TMAX_VERANO =
-0,2894`) se aplica entre las variables climáticas crudas `horas_frio` y
`tmax_verano` vía **cópula gaussiana** (`_muestrear_clima_correlacionado()`):
como ambas tienen marginal Normal, la cópula se reduce a una combinación
lineal de dos normales estándar por Cholesky (`Z1`, `rho·Z1 +
sqrt(1-rho²)·Z2`) escaladas y centradas en su propia media/desvío — no hace
falta pasar por `U=Phi(Z)` ni por una PPF, a diferencia de cuando el
marginal es una Beta. El ruido Beta que se aplica después sobre cada factor
(`_muestrear_beta_por_media`) queda **sin cambios y sin correlacionar en
ambas ramas del flag** — recibe como entrada valores de horas/tmax_verano ya
correlacionados cuando el flag está en `True`, pero el muestreo del ruido en
sí es siempre independiente y usa el mismo `rng.beta` en las dos ramas (esto
además resuelve, por diseño, un hallazgo separado del mismo review: antes
las ramas True/False muestreaban la Beta con dos algoritmos distintos —
`rng.beta` vs. `beta.ppf(norm.cdf(z),...)` — lo cual mezclaba "efecto de la
correlación" con "efecto de cambiar de algoritmo de muestreo" en la
comparación con semilla fija; ahora ambas ramas usan exactamente el mismo
muestreo de ruido). `_muestrear_clima_correlacionado_antitetico()` es la
versión con variables antitéticas, siguiendo el mismo patrón que el resto
del motor. Con el flag en `False`, el orden de consumo de `rng` en
`simulate_yields()`/`simulate_yields_antitetico()` sigue siendo idéntico al
de antes de agregar esta fuente (verificado: la fila
`correlacionar_frio_calor=False, semilla=42` de la tabla de abajo reproduce
exacto el mismo valor que antes del fix y que la columna "CAPEX/OPEX
estocásticos" de la entrada anterior — el bug y su corrección solo tocaron
la rama `True`).

Sigue habiendo una limitación inherente en la comparación True vs. False con
semilla fija, documentada en el docstring de `simulate_yields()`: activar el
flag cambia el orden en que Cholesky consume el stream de `rng`, así que una
comparación realización-por-realización no aísla `100%` el efecto de la
correlación — por eso la validación de abajo compara distribuciones
agregadas (10.000 simulaciones, 3 semillas), no pares de corridas individuales.

**Validación de robustez** — 50 ha, 10.000 simulaciones, escenario base,
`capex_opex_estocastico=True`, `run_monte_carlo()`, sobre `van_neto_usd` al
año 20, con 3 semillas por valor del flag para poder comparar el efecto del
flag contra el ruido de muestreo entre semillas del propio Monte Carlo.
**Tabla completa re-corrida con el mecanismo corregido** (script:
`notas/robustez_correlacion_frio_calor.csv`; los valores de la fila `False`
son idénticos a los de la versión anterior de esta tabla, como corresponde
ya que el bug solo afectaba la rama `True`):

| `correlacionar_frio_calor` | semilla | VAN medio | VAN P10 | VAN P50 | VAN P90 | P(VAN<0) | TIR media |
|---|---|---|---|---|---|---|---|
| False | 42   | USD 1.273.871 | USD 539.401 | USD 1.295.835 | USD 1.977.604 | 1,72% | 0,10773 |
| False | 123  | USD 1.284.467 | USD 560.541 | USD 1.307.785 | USD 1.978.445 | 1,79% | 0,10792 |
| False | 2026 | USD 1.284.341 | USD 538.747 | USD 1.311.510 | USD 1.998.751 | 1,84% | 0,10793 |
| True  | 42   | USD 1.268.817 | USD 552.352 | USD 1.283.303 | USD 1.964.856 | 1,44% | 0,10766 |
| True  | 123  | USD 1.275.711 | USD 551.527 | USD 1.297.385 | USD 1.964.454 | 1,55% | 0,10776 |
| True  | 2026 | USD 1.277.296 | USD 544.006 | USD 1.308.047 | USD 1.967.939 | 1,66% | 0,10780 |
| **False, media±d.e. entre semillas** | | **1.280.893 ± 6.082** | **546.230 ± 12.398** | | | **1,78% ± 0,06 pp** | 0,10786 |
| **True, media±d.e. entre semillas** | | **1.273.941 ± 4.508** | **549.295 ± 4.599** | | | **1,55% ± 0,11 pp** | 0,10774 |

**Chequeo de robustez (diferencia entre flags vs. ruido entre semillas):**
- VAN medio: la diferencia entre flags (-USD 6.952, -0,54%) ahora es del
  mismo orden que el desvío entre semillas (USD 4.500–6.100) — más grande
  que la diferencia de -USD 860 que arrojaba la versión con el bug, pero
  sigue sin ser claramente distinguible del ruido de muestreo con solo 3
  semillas por grupo.
- TIR media: diferencia entre flags ≈-0,00012 (recalculada desde los valores
  exactos de la tabla; la versión anterior de esta línea decía "~0,00003",
  que también estaba mal — el valor correcto de la versión *anterior* del
  mecanismo era ≈0,0000026, no 0,00003, un segundo error aritmético en esta
  sección que `accc6fa` no llegó a corregir). Con el mecanismo nuevo la
  diferencia real (-0,00012) es del mismo orden que el desvío entre semillas
  (0,00008–0,00011).
- P(VAN<0): diferencia entre flags (**-0,23 pp**, 1,78%→1,55%) — esta es la
  señal más clara de las cuatro métricas: 2–4 veces el desvío entre semillas
  (0,06–0,11 pp), en la dirección que predice la hipótesis de cobertura
  natural (activar la correlación negativa REDUCE la probabilidad de VAN
  negativo).
- VAN P10: diferencia entre flags (+USD 3.065, 546.230→549.295) — con el
  mecanismo corregido esta es ahora la señal MÁS débil de las cuatro, muy
  por debajo del desvío entre semillas (USD 4.599–12.398). La versión con el
  bug había mostrado acá la señal más fuerte (+USD 13.651) — ese resultado
  no se sostiene con la correlación real y correctamente ubicada; fue en
  buena medida un artefacto de qué seeds tocaron el grupo `True` diluido,
  no evidencia genuina de cobertura natural vía P10.

**Conclusión:** con el mecanismo corregido, el efecto de
`correlacionar_frio_calor` sobre el resultado financiero es más grande que
lo que sugería la versión con el bug (la diferencia en VAN medio pasó de
-USD 860 a -USD 6.952, casi 8x), pero para VAN medio y TIR media sigue
siendo del mismo orden que el ruido de muestreo entre semillas — no
concluyente con solo 3 semillas por grupo. La señal más clara y consistente
con la hipótesis de cobertura natural (frío malo y calor malo tienden a no
coincidir) aparece ahora en `P(VAN<0)`, no en el P10 como sugería
erróneamente la versión con el bug. Esto sigue siendo coherente con que
r=-0,2894 no sea estadísticamente significativo al 5% (IC95% cruza el cero,
potencia ~39%): el efecto financiero es real y algo más grande de lo
estimado originalmente, pero no lo bastante fuerte ni consistente entre
métricas como para justificar modelar esta dependencia como parte del motor
"oficial". Se mantiene la decisión de dejar el flag apagado por defecto,
disponible como análisis de sensibilidad explícito — con la salvedad de que,
a diferencia de la conclusión anterior, ahora hay evidencia algo más sólida
(P(VAN<0) separado 2–4x del ruido de semillas) de que el efecto no es
enteramente despreciable, y valdría la pena repetir esta validación con más
de 3 semillas si en algún momento se necesita una conclusión firme al
respecto.

### `dataset_ml.py` y `sensibilidad.py` fijan CAPEX/OPEX estocástico y correlación frío/calor en `False`

Resuelto el 2026-08-28 (Fase 3). Cierra el pendiente detectado por
`/code-review ultra` el 2026-08-26 (detalle del problema al final de esta
entrada).

**Decisión:** `src/dataset_ml.py` (`evaluar_punto()`) y `src/sensibilidad.py`
(`_evaluar_metrica()`) ahora construyen `ParametrosMC` con
`capex_opex_estocastico=False` y `correlacionar_frio_calor=False` **explícitos**
(antes heredaban el default del motor `capex_opex_estocastico=True` en
silencio). Fundamentación: el barrido LHS mide sensibilidad a los parámetros
de `ESPACIO_PARAMETROS`, y Sobol descompone la varianza atribuible a las 6
variables de `PROBLEMA_SOBOL`. El ruido de CAPEX/OPEX estocástico y la
correlación frío/calor son fuentes de variabilidad intra-simulación que las
Fases 1-2 agregaron al motor y que **no son inputs controlados** de ninguno de
los dos diseños: dejarlas activas mete varianza no atribuible a ningún factor
del barrido/Sobol. El NIVEL de CAPEX se sigue barriendo, vía el parámetro
`capex_extra_pct` que multiplica `capex_inicial_ha` — ese sí es un input
controlado del diseño y no cambia. De las dos alternativas que estaban
planteadas (fijar los flags en `False`, o sumar `capex_riego_*_ha` /
`capex_pozo_*_ha` / `opex_variacion_pct` como variables del barrido) se tomó
la primera, más simple.

**Regeneración desde cero, comparada contra las salidas anteriores** (motor en
`4bbc634..3bac94e`, semillas sin cambios):

| Salida | Resultado |
|---|---|
| `dataset_ml_train.parquet` (2400×19) / `_test.parquet` (600×19) | **byte-idénticos** a la versión commiteada (`DataFrame.equals` = True sobre las 3000 filas). `van_neto_medio_usd` y `prob_van_negativo` no se movieron ni un centésimo. Regeneración: train 529s + test 118s. |
| `sobol_indices.parquet` (VAN medio) | **byte-idéntico** (max \|ΔS1\| = max \|ΔST\| = 0). Ranking ST (promedio entre escenarios) sin cambios: `tasa_descuento` (≈0,893) ≫ `hectareas` (0,076) > `capex_extra_pct` (0,038) > `p_bajo_si_alto` (0,005) ≫ `precision_factor_frio`/`_calor` (≈1e-7). |
| `sobol_indices_riesgo.parquet` (`prob_van_negativo`) | **byte-idéntico**. Ranking ST sin cambios: `tasa_descuento` (≈0,934) ≫ `capex_extra_pct` (0,158) > `p_bajo_si_alto` (0,020) ≫ `precision_factor_frio`/`_calor` (≈3e-4 / 1e-4) > `hectareas` (0,000). |

El resultado es idéntico —no "parecido"— porque el dataset y las corridas de
Sobol anteriores (julio 2026) se generaron **antes** de que existieran
`capex_opex_estocastico` y `correlacionar_frio_calor` (Fases 1-2), es decir, ya
reflejaban el comportamiento `False` de ambos flags. Las ramas `False` del
motor preservan exactamente el orden de consumo de `rng` previo a esas fases
(verificado en la revisión de Fase 2), así que fijar los flags no altera
ningún sorteo. El valor de este paso es dejar la elección **explícita en el
código y documentada**, no un cambio numérico.

**Modelos surrogate re-entrenados** (`src/entrenar_modelo.py` sobre el dataset
regenerado; sin cambios de código, solo re-ejecución; R² en el test set, un
diseño LHS independiente del de train):

| Target | Random Forest R² | LightGBM R² |
|---|---|---|
| `van_neto_medio_usd` | 0,9861 | 0,9926 |
| `prob_van_negativo` | 0,9913 | 0,9953 |

Los cuatro siguen por encima de 0,98 (el umbral que ya se tenía). Como el
dataset es idéntico y la semilla de entrenamiento es fija, estos R² coinciden
con los de la corrida anterior — el re-entrenamiento confirma reproducibilidad,
no aporta números nuevos.

**El problema que esto cierra** (detectado por `/code-review ultra` el
2026-08-26, sobre el commit que agregó `capex_opex_estocastico`):
`evaluar_punto()` y `_evaluar_metrica()` armaban `ParametrosMC` sin pasar
`capex_opex_estocastico`, heredando el default `True`: cada punto del barrido
LHS / cada evaluación de Sobol arrastraba ruido de CAPEX/OPEX que no era
feature de `ESPACIO_PARAMETROS` ni variable de `PROBLEMA_SOBOL`, contaminando
`van_neto_medio_usd` / `prob_van_negativo` con varianza no atribuible a ningún
input. Con `correlacionar_frio_calor` no había problema práctico (su default ya
era `False`), pero se fijó explícito por el mismo criterio.

### Generador de precio consolidado en un solo flag (`modo_precio`), en vez de tres funciones paralelas

Resuelto el 2026-09-23, a pedido del feedback de Eze (tutor): existían tres
funciones `run_monte_carlo*()` con un generador de precio distinto cada una
y sin que nadie lo hubiera decidido a propósito por función —
`run_monte_carlo()` y `run_monte_carlo_antitetico()` usaban el triangular
simple (`simulate_prices`/`simulate_prices_antitetico`), mientras que
`run_monte_carlo_precio_historico()` usaba el AR(1) calibrado con datos
reales de FRED (`src/precio_estocastico.py`, ver más arriba en esta misma
sección). Consecuencia real, no hipotética: `app.py` y `src/sensibilidad.py`
quedaron en el triangular y `src/dataset_ml.py` en el AR(1) sin decisión
explícita de por medio.

**Cambio** (`src/monte_carlo.py`, sin tocar `app.py`, `src/sensibilidad.py`
ni `src/dataset_ml.py` — ver "Problemas abiertos" para ese próximo paso):

- `ParametrosMC.modo_precio: Literal["ar1", "triangular"] = "ar1"` — nuevo
  campo, default AR(1) porque es el modelo elegido (calibrado con datos
  reales, ver notas/plan_precio_historico.md). El triangular queda
  disponible como modo explícito de comparación contra el modelo naive, ya
  no por default en ningún lado.
- `run_monte_carlo()` y `run_monte_carlo_antitetico()` despachan
  internamente según `params.modo_precio` (funciones privadas
  `_despachar_precios()`/`_despachar_precios_antitetico()`) — mismo patrón
  que ya usan `capex_opex_estocastico`/`correlacionar_frio_calor`: el flag
  decide DENTRO de la función compartida, no en funciones top-level
  paralelas. Ambas aceptan ahora un `precio_params: ParametrosPrecioAR1`
  opcional (solo tiene efecto con `modo_precio="ar1"`), necesario porque
  `src/dataset_ml.py` barre el drift (`precio_params.c`) por LHS.
- `run_monte_carlo_precio_historico()` queda como **alias fino** de
  `run_monte_carlo_antitetico()` con `modo_precio="ar1"` forzado (no se
  elimina: es el único caller de `src/dataset_ml.py`, que este cambio
  explícitamente no toca todavía).

**Validación** (50 ha, 10.000 simulaciones, escenario base, semilla 42,
`capex_opex_estocastico=True` default, sobre `van_neto_usd` al año 20):

| Función | `modo_precio` | VAN medio | P(VAN<0) |
|---|---|---:|---:|
| `run_monte_carlo()` | triangular | USD 1.273.871 | 1,72% |
| `run_monte_carlo()` | ar1 (nuevo default) | USD 2.257.098 | 19,60% |
| `run_monte_carlo_antitetico()` | triangular | USD 1.279.569 | 1,64% |
| `run_monte_carlo_antitetico()` | ar1 | USD 2.244.566 | 20,50% |
| `run_monte_carlo_precio_historico()` (alias) | — | USD 2.244.566 | 20,50% |

`triangular` en ambas funciones reproduce EXACTO (mismo orden de consumo de
`rng`, código sin cambios de comportamiento) los valores ya documentados
más arriba en esta sección (1,72% para `run_monte_carlo()`, la fila
`correlacionar_frio_calor=False, semilla=42` de la tabla de robustez). El
alias `run_monte_carlo_precio_historico()` da bit-idéntico a
`run_monte_carlo_antitetico(modo_precio="ar1")`, como corresponde a un
alias sin lógica propia. Migración confirmada sin sorpresas.

**Efecto en los callers existentes, sin tocar su código**: como `"ar1"` es
ahora el default de `ParametrosMC`, `app.py` (`run_monte_carlo()`) y
`src/sensibilidad.py` (`run_monte_carlo_antitetico()`) pasan a usar AR(1)
automáticamente la próxima vez que corran — sin editar esos archivos, que
es el punto de centralizar la decisión en el dataclass. `src/dataset_ml.py`
sigue en AR(1) sin cambios (vía el alias). Falta como próximo paso (ver
"Problemas abiertos"): confirmarlo corriendo la interfaz y regenerar
Sobol/dataset/modelos si corresponde, una vez que Emilia dé el visto bueno
a este cambio en el motor.

## Problemas abiertos

### `st.markdown` rompe el formato de P10/P50/P90 cuando P10 es negativo (app.py)

Detectado el 2026-09-23 probando la UI en el navegador tras el cambio de
`modo_precio` (más común de ver ahora que con AR(1) el P10 da negativo más
seguido, pero el bug es preexistente, no introducido por ese cambio). La
línea `st.markdown(f"**P10:** US$ {p10:,.0f}  ·  **P50:** US$ {p50:,.0f}  ·
**P90:** US$ {p90:,.0f}")` en la pestaña "Distribución del VAN": cuando
`p10` es negativo, el texto resultante contiene dos `$` (de "US$") con un
número negativo en el medio, y Streamlit interpreta ese tramo como
delimitador de LaTeX en vez de texto literal — rompe el formato en pantalla
(muestra fórmula en vez de "US$ -667.568"). Arreglo simple: no usar el
símbolo `$` literal (reemplazar "US$" por "USD" en ese f-string, o escapar
con `\$`). No se corrigió en este cambio por estar fuera de su alcance.

### Los parámetros clave no tienen soporte bibliográfico

Algunos números que están en el código son supuestos que hay que citar o justificar:
- Los factores de la función de transferencia de frío (0.40, 0.70, 1.00) — buscar en Ferguson (2006) o Ruiz et al. (2018)
- Los parámetros de la cadena de Markov de vecería (p_bajo_si_alto = 0.65, p_alto_si_bajo = 0.80) — buscar en Polito & Pinney (1999) o datos INTA
- La temperatura base GDD = 10 °C — Crane & Takeda (1979)
- El umbral de 800 hs como crítico y 1.000 hs como óptimo — Goldhammer (1995)
- La inversión inicial de USD 18.000/ha — necesita cotización real o fuente de INTA / plan de negocios

### Inferencia bayesiana pendiente

Ver Paso 4 más abajo — reemplazar los supuestos ad hoc de frío/vecería por estimaciones con MCMC (PyMC).

### Deuda técnica menor en `src/costos.py`/`src/monte_carlo.py`

También detectado por `/code-review ultra` el 2026-08-26, sin corregir (no
son bugs, son limpieza pendiente):

- `costo_operativo_anual()` se recalcula dos veces por corrida: una vez
  dentro de `flujo_caja_neto()` y otra vez en `_orquestar_resultado()` solo
  para poblar la columna `opex_usd`. Trabajo duplicado, no afecta el
  resultado.
- El `if not estocastico: return ...` (capex/opex en cero/uno) está
  copy-pasteado idéntico en las cuatro funciones `simulate_capex_extra`,
  `simulate_capex_extra_antitetico`, `simulate_opex_multiplicador` y
  `simulate_opex_multiplicador_antitetico` en vez de resolverse una sola
  vez -- cualquier cambio futuro a ese caso especial requiere tocar las
  cuatro en simultáneo.

### La interfaz Streamlit permite simular hasta 300 ha, fuera del rango barrido para ML/Sobol

Anotado el 2026-08-30 (Fase 4, 2ª iteración). El slider de `hectareas` de
`app.py` llega a 300 ha. El **motor** lo soporta sin problema: se verificó
que `run_monte_carlo()` con `hectareas=300` corre sin error y que el VAN
medio escala exactamente 6x respecto de 50 ha (CAPEX, OPEX e ingresos son
lineales en superficie; `P(VAN<0)` y TIR quedan invariantes, como se
espera). La simulación de la interfaz llama directo al motor, no al
surrogate, así que esto es correcto.

Lo que **no** cubre 300 ha:
- `ESPACIO_PARAMETROS` en `src/dataset_ml.py`: `hectareas` en `(25.0, 100.0)`
  — el dataset sintético de train/test y por lo tanto el modelo ML surrogate
  sólo vieron hasta 100 ha.
- `PROBLEMA_SOBOL` en `src/sensibilidad.py`: `hectareas` en `[50.0, 100.0]`
  — los índices de `data/processed/sobol_indices.parquet` se calcularon en
  ese rango.

Si en algún momento se usa el modelo ML o se interpretan los índices de
Sobol para escenarios de >100 ha, es **extrapolación fuera del rango de
entrenamiento** y hay que decirlo explícito (o re-barrer con el piso/techo
ampliado). No es urgente: hoy la interfaz sólo usa el motor.

### `modo_precio="ar1"` propagado a app.py y sensibilidad.py; Sobol/dataset/modelos regenerados

Resuelto el 2026-09-23 (Fase A, paso 2 — cierra el punto 3 del feedback de
Eze: que la elección del generador de precio no quedara implícita/heredada
en ningún caller). Sobre la consolidación de `ParametrosMC.modo_precio` (ver
la entrada "Resuelto" anterior), quedaban dos lugares que heredaban el
triangular viejo en silencio: `app.py` (vía `run_monte_carlo()`) y
`src/sensibilidad.py` (vía `run_monte_carlo_antitetico()` en
`_evaluar_metrica()`). Ninguno de los dos tenía en su docstring una
justificación puntual para preservar el triangular a propósito — era
inercia del código anterior a que existiera el flag, así que se corrigieron
ambos.

**Cambios:**

- `app.py`: nuevo selector "Modelo de precio" en el sidebar (`st.selectbox`,
  opciones AR(1)/triangular, default AR(1)). Se corrigió el texto de ayuda
  de "Escenario de precio" (ya no afirma "distribución triangular" como si
  fuera el único modelo) y se agregaron menciones al modo de precio activo
  en la pestaña "Comparar escenarios" y un aviso en la pestaña "Sensibilidad
  (Sobol)" cuando el modo seleccionado no coincide con el usado para
  pre-calcular esos índices (siempre AR(1) ahora).
- `src/sensibilidad.py`: `_evaluar_metrica()` fija `modo_precio="ar1"`
  explícito al construir `ParametrosMC` -- mismo patrón que ya usa para
  `capex_opex_estocastico=False`/`correlacionar_frio_calor=False`. Se
  eligió `"ar1"` (no `"triangular"`) porque Sobol debe analizar el
  simulador "oficial" (el que corre `app.py` por default), no el modelo
  naive.
- `src/dataset_ml.py` **sin cambios** (ya usaba AR(1) vía
  `run_monte_carlo_precio_historico()`).

**Dataset LHS (`src/dataset_ml.py`) regenerado para confirmar**: 685s total
(train 521s + test 164s). **Byte-idéntico** al commiteado
(`DataFrame.equals` = True, 2400×19 train y 600×19 test) — no cambió, como
se esperaba, porque ya corría en AR(1) antes de este cambio.

**Sobol regenerado** (`n_base=128`, mismos parámetros que la corrida
anterior; 2055s total): acá SÍ cambian los números, porque antes corría con
el triangular heredado y ahora con AR(1) explícito.

VAN medio (ST, promedio entre escenarios):

| parámetro | ST viejo (triangular) | ST nuevo (AR1) |
|---|---:|---:|
| tasa_descuento | 0,8934 | 0,8856 |
| hectareas | 0,0763 | 0,0930 |
| capex_extra_pct | 0,0378 | 0,0248 |
| p_bajo_si_alto | 0,00518 | 0,00459 |
| precision_factor_frio | ~0 | ~0 |
| precision_factor_calor | ~0 | ~0 |

Riesgo (`prob_van_negativo`, ST):

| parámetro | ST viejo (triangular) | ST nuevo (AR1) |
|---|---:|---:|
| tasa_descuento | 0,9340 | 0,9345 |
| capex_extra_pct | 0,1584 | 0,0663 |
| p_bajo_si_alto | 0,0195 | 0,0075 |
| precision_factor_frio | 0,0003 | 0,0001 |
| precision_factor_calor | 0,0001 | 0,0001 |
| hectareas | 0,0000 | 0,0000 |

**`tasa_descuento` sigue dominando lejos, en ambos targets y en las dos
corridas** — el ranking no cambió de orden. Lo que sí se mueve: con AR(1),
`hectareas` gana algo de peso en el VAN medio (0,076→0,093, la escala
importa un poco más cuando el precio ya no se "cancela" entre años vía el
triangular independiente) y `capex_extra_pct` pierde peso en ambos targets,
más marcado en riesgo (0,158→0,066) — con AR(1) el precio explica una
porción mayor de la varianza de `prob_van_negativo` que antes se le
atribuía en parte al ruido de CAPEX/OPEX vía correlación espuria con la
semilla compartida, así que `capex_extra_pct` queda con menos varianza
residual para explicar. `precision_factor_frio`/`_calor` siguen en ~0 en
ambas corridas (correcto: no mueven la media por construcción).

**Modelos ML reentrenados** (`src/entrenar_modelo.py`, mismo dataset →
mismos R² que antes, sin sorpresas): RF 0,9861 / LightGBM 0,9926
(`van_neto_medio_usd`); RF 0,9913 / LightGBM 0,9953 (`prob_van_negativo`).
La tabla de importancia de features del RF sigue liderada por
`tasa_descuento`/`hectareas`/`escenario`, consistente con el ranking de
Sobol nuevo.

**UI verificada en el navegador** (`streamlit run app.py`, Chrome vía
claude-in-chrome): el selector "Modelo de precio" cambia correctamente
entre AR(1) (default, VAN medio ≈USD 2,31M, P(VAN<0)≈18,8% a N=3.000) y
triangular (VAN medio ≈USD 1,29M, P(VAN<0)≈1,8% a N=3.000) -- ambos en
línea con los valores de la corrida de 10.000 simulaciones de la entrada
anterior. El histograma se ve visiblemente más angosto en triangular
(consistente con no tener memoria/persistencia de precio). El aviso de
"Sobol no refleja el modo triangular seleccionado" aparece correctamente
al cambiar de modo. Detectado de paso, **no corregido acá por estar fuera
de alcance** de este cambio: el markdown de P10/P50/P90 en la pestaña
"Distribución del VAN" (`st.markdown(f"**P10:** US$ {p10:,.0f} ...")`) se
renderiza mal cuando `p10` es negativo -- Streamlit interpreta el `$...$`
como delimitador de LaTeX y rompe el formato. Es un bug preexistente
(no introducido por este cambio), más visible ahora porque con AR(1) el
P10 da negativo más seguido.

### El motor escala TODO el CAPEX linealmente con hectáreas — probablemente incorrecto para varios ítems

Anotado el 2026-08-30 (Fase 4, auditoría de CAPEX pedida por Emilia).
**Hallazgo pendiente de resolver, NO verificado con datos reales del
proyecto.** La clasificación de abajo es mi mejor juicio técnico/agronómico
(Claude), no algo confirmado por Emilia ni por Facundo/Patricio. No se
cambió código todavía.

**Mecanismo actual (cómo `src/costos.py` llega a "USD/ha"):**

1. `CAPEX_INICIAL_USD_HA = 45_160.52` es una **constante hardcodeada**, no
   se lee `data/external/capex.csv` en runtime. El valor se calculó offline:
   suma de los `total_usd` de las 16 filas con `costeado=SI` de `capex.csv`
   = **USD 1.129.013,12**, dividida por **25 ha** (Fase I, la superficie que
   `data/external/README.md` fija como base del dataset financiero).
   1.129.013,12 / 25 = 45.160,52.
2. `ParametrosCostos.capex_inicial_ha` toma esa constante como default
   (src/costos.py:107).
3. La propiedad `ParametrosCostos.capex_inicial` (src/costos.py:159) hace
   `capex_inicial_ha * hectareas` — escala **lineal pura**, sin distinguir
   ítem por ítem. Comentario en el código: `# escala lineal con hectareas`.
4. `dataset_ml.py` y `sensibilidad.py` sólo multiplican `capex_inicial_ha`
   por `(1 + capex_extra_pct)` — un recargo porcentual, no cambia el
   supuesto de linealidad.

Consecuencia: 50 ha → 300 ha da exactamente 6x en `capex_inicial`. Y ya al
default de 50 ha, dividir por 25 la finca de 100 ha + la maquinaria "lumpy"
infla el CAPEX/ha.

**Clasificación tentativa de cada ítem de `capex.csv`** (`lineal` = crece
~proporcional a la superficie; `fijo` = no depende [mucho] de la superficie;
`escala` = crece con la superficie pero sub-proporcional / a saltos):

| Ítem (categoría) | total_usd | costeado | Clasif. | Razonamiento (1 línea) |
|---|---:|:---:|:---:|---|
| Finca (TIERRA) | 600.000 | SI | **fijo** ⚠ | Los 100 ha ya están comprados (README pto. 2): para cualquier proyecto ≤100 ha el costo de tierra es hundido y fijo; además hoy se divide por 25 una línea cotizada sobre 100 ha → USD/ha de tierra x4. *Requiere confirmar si los 100 ha están efectivamente pagados.* |
| Análisis de suelo (TIERRA) | 513,12 | SI | fijo | Un análisis compuesto de laboratorio cubre el campo; más ha suman algunos puntos de muestreo, no proporcional. |
| Análisis de agua (TIERRA) | 0 | NO | fijo | Un análisis por fuente de agua (pozo), no por ha plantada. |
| Desmonte / nivelación (TIERRA) | 0 | NO | lineal | Movimiento de suelo medido en USD/ha (la fuente usa `unidad=ha`); heterogeneidad del terreno agrega ruido. |
| Subsolado profundo (TIERRA) | 0 | NO | lineal | Labor de tractor a 80 cm, costo ≈ horas/ha × tarifa (`unidad=ha`). |
| Marcación (TIERRA) | 0 | NO | lineal | Marcar posiciones de planta escala con ha, pero la fuente dice que es inmaterial. |
| Pozo de agua perforación x2 (RIEGO) | 0 | NO | **escala** ⚠ | La capacidad (L/s) de un pozo cubre un rango de ha; se agrega un 2º pozo recién al cruzar ese umbral (la fuente ya prevé 2). *Requiere el estudio hidrogeológico real para saber ha/pozo.* |
| Bombas y tablero eléctrico x2 (RIEGO) | 0 | NO | escala ⚠ | Dimensionadas a la capacidad del pozo/bloque de riego; saltan con el nº de bombas. Atado al nº de pozos. |
| Filtros y fertilizadores (RIEGO) | 0 | NO | escala ⚠ | Un cabezal de filtrado tiene caudal máximo que cubre un rango de ha; se suman equipos a saltos. Parte proporcional si se sobredimensiona el cabezal. |
| Tuberías + laterales + goteros (RIEGO) | 0 | NO | lineal | Longitud de tubería y nº de goteros ≈ proporcionales a superficie y nº de plantas (`unidad=ha`, 2 goteros/planta); las primarias tienen leve economía de escala. |
| Represa / cisterna 5.000 m³ (RIEGO) | 0 | NO | escala ⚠ | El volumen necesario crece con ha, pero la obra civil de un reservorio escala sub-lineal (USD/m³ baja con el tamaño). *Sin fuente de precio.* |
| Plantas hembra Kerman (PLANTAS) | 310.500 | SI | **lineal** | Densidad de plantación (plantas/ha) fija → nº de plantas y costo proporcionales a la superficie. |
| Plantas macho Peters/Randy (PLANTAS) | 27.000 | SI | **lineal** | Proporción de polinizadores (~8-10 % de hembras) fija → escala con ha. |
| Tutores caña/fibra (PLANTAS) | 7.500 | SI | lineal | Un tutor por árbol joven → escala con el nº de plantas / ha. |
| Tractor 90 HP (MAQUINARIA) | 55.000 | SI | **fijo** en el rango ⚠ | Un tractor cubre ~100-150 ha de tareas de monte; recién se compra un 2º al superar su capacidad de trabajo. En 25-100 ha, 1 unidad. |
| Pulverizadora turbina 1000 L (MAQUINARIA) | 12.000 | SI | fijo en el rango ⚠ | Cubre muchas ha por ventana de pulverización; 2ª unidad sólo a gran escala. |
| Subsolador 3 cuerpos (MAQUINARIA) | 4.000 | SI | fijo | Un implemento sirve a toda la operación dentro de un rango amplio. |
| Rastra 16 discos (MAQUINARIA) | 3.800 | SI | fijo | Ídem: implemento único. |
| Desmalezadora 1.5 m (MAQUINARIA) | 4.200 | SI | fijo | Ídem: implemento único; quizá 2º a gran escala. |
| Arado de discos (MAQUINARIA) | 3.500 | SI | fijo | Ídem: implemento único. |
| Paneles solares (INFRA) | 0 | NO | escala ⚠ | kW instalados ≈ demanda de bombeo ≈ ha regadas → casi lineal, con economía de escala en instalación. *La familia lo está cotizando.* |
| Galpón herramientas 200 m² (INFRA) | 50.000 | SI | escala ⚠ | 200 m² sirven a un rango de ha; a más escala el área crece sub-proporcional y el USD/m² es ~plano. En 25-100 ha, 1 galpón. |
| Sistema eléctrico finca (INFRA) | 15.000 | SI | fijo en el rango ⚠ | El costo de la línea trifásica depende sobre todo de la distancia a la red, no de las ha; los tableros escalan algo con la carga. |
| Camino interno ripio 3 km (INFRA) | 12.000 | SI | escala | La longitud de caminos crece con el área como ~√área (se atraviesa el predio), no lineal. |
| Alambrado perimetral 8 km (INFRA) | 12.000 | SI | **escala** | El perímetro escala como ~√área: duplicar la superficie → perímetro ×1,41, no ×2. Sub-lineal clásico. |
| Red antigranizo (INFRA) | 0 | NO | lineal | Malla + postes son estructura por ha (`unidad=ha`). Opcional. |
| Oficina/vivienda encargado (INFRA) | 12.000 | SI | fijo | Un módulo prefabricado independiente de las ha en el rango; quizá 2ª vivienda a muy gran escala. |
| Software de gestión agrícola (TECNOLOGIA) | 0 | NO | fijo ⚠ | Licencia por finca/usuario, no por ha (algunos SaaS cobran por ha). "No indispensable". |
| Estación meteorológica (TECNOLOGIA) | 0 | NO | fijo | Una estación cubre la finca / microclima; 2ª sólo para sitios muy grandes o heterogéneos. |
| Sitio web e-commerce (TECNOLOGIA) | 0 | NO | fijo | Totalmente independiente de las ha; "se haría internamente". |

**Peso de cada clase dentro de los USD 1.129.013 costeados (los que hoy escalan lineal):**

| Clase | Ítems | USD | % del CAPEX costeado |
|---|---|---:|---:|
| Genuinamente lineal | plantas hembra + macho + tutores | 345.000 | ~30,6 % |
| Tierra (fijo/hundido para ≤100 ha, hoy mal basado a /25) | finca | 600.000 | ~53,1 % |
| Fijo / "lumpy" (maquinaria + edificios) | tractor, pulverizadora, 4 implementos, análisis suelo, sist. eléctrico, oficina | 110.013 | ~9,7 % |
| Economía de escala | galpón | 50.000 | ~4,4 % |
| Sub-lineal (~√área) | camino interno + alambrado | 24.000 | ~2,1 % |

Es decir: **sólo ~31 % del CAPEX costeado es realmente lineal con la
superficie.** El ~53 % es tierra (probablemente fija en el rango de ha del
proyecto) y el ~16 % restante es fijo/lumpy o sub-lineal. Escalar el total
×6 de 50 a 300 ha sobreestima fuerte el CAPEX a gran escala; y dividir por
25 la finca de 100 ha + la maquinaria lumpy ya sobreestima el CAPEX/ha al
default de 50 ha. Por debajo de 25 ha, la linealidad subestima (los costos
fijos no se achican).

**Qué haría falta para resolverlo (no hacerlo ahora, esperar decisión de Emilia):**
separar `capex.csv` en un componente fijo (USD absolutos, no USD/ha), uno
lineal (USD/ha × ha) y uno con función de escala (p. ej. escalones para
pozos/tractor, √área para perímetro), y decidir con Facundo/Patricio los
umbrales reales (ha por pozo, ha por tractor, si los 100 ha ya están
pagados). Mientras tanto, la simulación a ≠25 ha arrastra este sesgo.

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

Última actualización: 2026-08-26
