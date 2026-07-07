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

## Problemas que hay que resolver antes de la defensa

### El VAN es solo de ingresos brutos, no es un modelo de negocio

Esto es lo más urgente. Ahora el modelo calcula ingresos x precio x hectáreas y lo descuenta. No hay costos. Un jurado con formación financiera lo detecta de inmediato.

Lo que falta modelar:
- Inversión inicial en año 0 (plantines, sistema de riego, preparación del suelo) — alrededor de USD 18.000/ha
- Costos fijos anuales (mano de obra, mantenimiento, seguros) — alrededor de USD 1.500/ha/año
- Costos variables desde el año 6 en adelante (cosecha, procesamiento, insumos) — alrededor de USD 800/ha/año

Con eso se puede calcular VAN neto, TIR y período de recupero. Sin esto el número que sale no significa nada.

### No hay análisis de sensibilidad

No se sabe qué variable mueve más el VAN. ¿Es el precio? ¿Las horas de frío? ¿La vecería? Eso hay que demostrarlo, no suponerlo.

Lo que falta:
- Tornado chart: varía cada parámetro uno a la vez mientras los demás están fijos
- Índices de Sobol S1 y ST: mide qué fracción de la varianza total del VAN explica cada variable, incluyendo interacciones
- Librería a usar: SALib

### No hay reducción de varianza

El programa de la materia lo pide explícitamente. La técnica es variables antitéticas: en lugar de generar U ~ Uniforme(0,1), se generan pares (U, 1-U). El estimador resultante tiene menos varianza con el mismo número de simulaciones. Hay que implementarlo y comparar la varianza con y sin la técnica.

### Los parámetros clave no tienen soporte bibliográfico

Algunos números que están en el código son supuestos que hay que citar o justificar:
- Los factores de la función de transferencia de frío (0.40, 0.70, 1.00) — buscar en Ferguson (2006) o Ruiz et al. (2018)
- Los parámetros de la cadena de Markov de vecería (p_bajo_si_alto = 0.65, p_alto_si_bajo = 0.80) — buscar en Polito & Pinney (1999) o datos INTA
- La temperatura base GDD = 10 °C — Crane & Takeda (1979)
- El umbral de 800 hs como crítico y 1.000 hs como óptimo — Goldhammer (1995)
- La inversión inicial de USD 18.000/ha — necesita cotización real o fuente de INTA / plan de negocios

---

## Plan de implementación

### Paso 1 — módulo de costos (empezar por acá)

Crear `src/costos.py` con una dataclass `ParametrosCostos` y actualizar `src/monte_carlo.py` para que el VAN se calcule sobre el flujo de caja neto, no sobre los ingresos brutos.

El flujo de cada año sería:
- año 0: egresa la inversión inicial
- años 1 a 5: solo costos fijos (no hay producción todavía)
- años 6 a 20: ingresos - costos fijos - costos variables

El resultado esperado es tener `van_neto_usd` en lugar de `van_acumulado_usd`, más la TIR y el año de recupero.

### Paso 2 — variables antitéticas

En `src/monte_carlo.py` agregar una función `run_monte_carlo_antitetico()`. La idea es simple: generar la mitad de las simulaciones con U y la otra mitad con 1-U, apilarlas y comparar la varianza del estimador resultante contra el Monte Carlo estándar con el mismo N.

### Paso 3 — análisis de sensibilidad

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
- SALib >= 1.4 — análisis de sensibilidad Sobol
- arviz >= 0.17 — visualización de distribuciones posterior

---

## Cosas que no hay que olvidar

- Agregar el .gitignore (hay un archivo sin commitear)
- Citar todas las referencias bibliográficas de los parámetros antes de la defensa (ver sección de problemas)
- Aclarar en el texto de la tesis que el VAN no incluye impuestos ni financiamiento
- La curva de producción por año (del 1 al 20) también necesita una referencia o aclarar que es un supuesto del modelo

---

Última actualización: junio 2026
