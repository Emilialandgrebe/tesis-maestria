"""Simulación de Monte Carlo para rendimientos e ingresos del cultivo de pistacho."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from scipy.stats import triang as triang_dist

from src.costos import HECTAREAS_REFERENCIA, ParametrosCostos, flujo_caja_neto, opex_total
from src.modelo_v2 import cargar_modelo_v2
from src.precio_estocastico import (
    ParametrosPrecioAR1,
    simulate_prices_ar1,
    simulate_prices_ar1_antitetico,
)

# ---------------------------------------------------------------------------
# Constantes del plan de negocios y calibración climática
# ---------------------------------------------------------------------------

_MODELO = cargar_modelo_v2()

HECTAREAS = HECTAREAS_REFERENCIA  # 120 ha, Supuestos!B11 del Excel v2

# Rendimiento a plena producción (100% de la curva): Supuestos!B20 del Excel
# v2, 3.000 kg/ha de pistacho con cáscara. Es el rendimiento ESPERADO una
# vez aplicados frío, calor y supervivencia en sus valores medios
# calibrados (la vecería tiene media 1 desde la sincronización v2, ver
# `_media_veceria()`), no un valor que se observe todos los años.
RENDIMIENTO_VALIDADO_KG_HA = _MODELO.supuestos["rendimiento_plena_kg_ha"]

# Calibración climática y biológica real (Módulo 0, Jocolí 1990-2024,
# temperatura ERA5-Land de Open-Meteo con `models=era5_land`; recalibrado el
# 2026-11-09 desde la serie "Best Match", que desde 2017 mezcla ECMWF IFS y
# no es homogénea, ver src/data/climate_fetcher.py y notas/PLAN_TESIS.md).
# Se usan tanto como defaults de ParametrosMC como para derivar
# RENDIMIENTO_IDEAL_KG_HA más abajo — una sola fuente de verdad.
_HORAS_FRIO_MEDIA = 998.7
_HORAS_FRIO_STD = 220.4
_CALOR_VERANO_MEDIA = 29.42
_CALOR_VERANO_STD = 1.54
_P_BAJO_SI_ALTO = 0.65   # P(año bajo | año previo fue alto), vecería
_P_ALTO_SI_BAJO = 0.80   # P(año alto | año previo fue bajo), vecería
_VECERIA_FACTOR_MIN = 0.60
_VECERIA_FACTOR_MAX = 0.70
_PLANTAS_ALPHA = 2.0     # falla de plantas ~ Beta(alpha, beta), media ~9%
_PLANTAS_BETA = 20.0

# Correlación empírica horas_frio vs. tmax_media_verano_c (Módulo 0,
# data/processed/features_climaticas.parquet, 35 años ERA5-Land Jocolí
# 1990-2024): Pearson r=-0.2767, p=0.108 (con Best Match era r=-0.2894,
# p=0.092). NO significativa al 5% (n=35 da poca potencia para detectar un
# r de esta magnitud, y el IC95% de Fisher es aprox. [-0.56, +0.06] --
# incluye el cero con margen). Por eso
# NO se modela por defecto (`ParametrosMC.correlacionar_frio_calor=False`);
# ver notas/PLAN_TESIS.md, 2026-08-26, para la fundamentación completa y el
# chequeo de robustez contra el ruido de muestreo del propio Monte Carlo.
CORR_HORAS_FRIO_TMAX_VERANO = -0.2767

# Fracción del rendimiento pleno por año del proyecto (1-20).
#
# CURVA_EXCEL es la curva de Supuestos!D54:X54 tal cual. Ya trae la vecería
# ADENTRO (años 12-20: 0,90 / 0,80 / 0,75 / 0,95 / 0,80 / 0,75 / 0,85 / 1 /
# 1): se usa solo en el modo determinístico (`ParametrosMC.deterministico`),
# que reproduce el Excel sin ninguna fuente estocástica.
#
# CURVA_PRODUCCION es la curva de maduración para Monte Carlo, donde la
# vecería se simula aparte (`_simular_veceria()`): usar la curva del Excel
# la contaría dos veces. Primera cosecha en el año siguiente al último
# pre-productivo (Supuestos!B8 = 6 -> año 7), rampa del Excel en los años
# 7-11, y desde el año 12 una meseta igual al promedio de la curva del
# Excel en los años 12-20 (7,8 / 9 = 0,8667): así el rendimiento esperado
# acumulado de la meseta coincide con el del Excel.
ANIO_PRIMERA_COSECHA = int(_MODELO.supuestos["ultimo_anio_preproductivo"]) + 1
ANIO_INICIO_MESETA = 12

_CURVA_EXCEL = _MODELO.curva_excel(_MODELO.n_años)
CURVA_EXCEL: dict[int, float] = {
    año: float(_CURVA_EXCEL[año - 1]) for año in range(1, _MODELO.n_años + 1)
}
MESETA_PRODUCCION = float(_CURVA_EXCEL[ANIO_INICIO_MESETA - 1:].mean())
CURVA_PRODUCCION: dict[int, float] = {
    año: (
        0.0 if año < ANIO_PRIMERA_COSECHA
        else CURVA_EXCEL[año] if año < ANIO_INICIO_MESETA
        else MESETA_PRODUCCION
    )
    for año in range(1, _MODELO.n_años + 1)
}

# Modo de precio "triangular" (comparación contra el modelo naive, sin
# memoria): MULTIPLICADOR (min, moda, max) independiente por año sobre la
# trayectoria de precio del Excel v2 del escenario. Los anchos relativos son
# los de la triangular absoluta anterior al Excel v2 (pesimista 6/7/8,5,
# base 8/9,5/11,5, optimista 11/13/15 USD/kg), divididos por su moda.
ESCENARIOS_PRECIO: dict[str, tuple[float, float, float]] = {
    "pesimista": (6.0 / 7.0,  1.0,  8.5 / 7.0),
    "base":      (8.0 / 9.5,  1.0, 11.5 / 9.5),
    "optimista": (11.0 / 13.0, 1.0, 15.0 / 13.0),
}


# ---------------------------------------------------------------------------
# Funciones de transferencia climática
#
# Se definen antes de RENDIMIENTO_IDEAL_KG_HA porque ese valor se deriva
# integrándolas contra la calibración climática de arriba.
# ---------------------------------------------------------------------------

def _media_factor_frio(horas: np.ndarray) -> np.ndarray:
    """
    Función de transferencia: horas de frío acumuladas → factor de rendimiento
    ESPERADO (media determinística; el ruido biológico se agrega después con
    `_muestrear_beta_por_media`).

    El punto de referencia (factor = 1.00) es la MEDIA HISTÓRICA REAL del
    sitio (horas_frio_media = 998.7 hs, calibrada con datos ERA5-Land de
    Jocolí 1990-2024), no el óptimo agronómico teórico de 1.000 hs. Si se
    centrara en el óptimo teórico, el clima típico del sitio (que está por
    debajo de ese óptimo) se penalizaría dos veces: una vez implícitamente,
    porque RENDIMIENTO_VALIDADO_KG_HA (3.000 kg/ha) ya está validado contra
    el clima real de Jocolí, y otra vez explícitamente, acá, si el factor no
    llegara a 1.00 en un año climáticamente típico.

    Umbrales (mismo ancho de zona que la versión centrada en el óptimo teórico,
    solo desplazados a la media real):
    - >= 998.7 hs : año típico o mejor, sin penalidad (factor = 1.00)
    - 798.7–998.7 hs: penalidad lineal moderada (0.70–1.00)
    - < 798.7 hs  : año con déficit severo, penalidad fuerte (0.40–0.70)
    """
    CENTRO = _HORAS_FRIO_MEDIA
    ANCHO = 200.0  # mismo ancho de zona moderada que la función original
    UMBRAL_SEVERO = CENTRO - ANCHO  # 798.7
    return np.clip(
        np.where(
            horas >= CENTRO,
            1.0,
            np.where(
                horas >= UMBRAL_SEVERO,
                0.70 + 0.30 * (horas - UMBRAL_SEVERO) / ANCHO,
                0.40 + 0.30 * horas / UMBRAL_SEVERO,
            ),
        ),
        0.0,
        1.0,
    )


def _media_factor_calor(tmax_verano: np.ndarray) -> np.ndarray:
    """
    Función de transferencia: tmax media de verano (ene-feb) → factor de
    rendimiento ESPERADO (media determinística; el ruido biológico se agrega
    después con `_muestrear_beta_por_media`).

    El punto de referencia (factor = 1.00) es la MEDIA HISTÓRICA REAL del
    sitio (calor_verano_media = 29.42 °C, calibrada con datos ERA5-Land de
    Jocolí 1990-2024), no el rango óptimo agronómico teórico (35-38 °C,
    Crane y Takeda, 1979; Ferguson, 2006). Mismo razonamiento que en
    `_media_factor_frio`: centrar en el óptimo teórico penalizaría dos veces
    el déficit de calor que ya está implícito en RENDIMIENTO_VALIDADO_KG_HA
    (3.000 kg/ha) para este sitio.

    No se premia con factor > 1.00 un año más cálido que la media: eso
    requeriría que el factor pudiera superar 1.0, lo cual es incompatible
    con `_muestrear_beta_por_media` (la Beta está acotada en [0, 1] por
    definición). "Mejor que la media" se traduce en "sin penalidad adicional"
    (factor = 1.00), igual que en la función de frío.

    Umbrales (mismo ancho de zona que la versión centrada en el óptimo
    teórico, solo desplazados a la media real):
    - >= 29.42 °C  : año típico o mejor, sin penalidad (factor = 1.00)
    - 22.12–29.42 °C: penalidad lineal moderada (0.70–1.00)
    - < 22.12 °C   : año con déficit severo, penalidad fuerte (0.40–0.70)
    """
    CENTRO = _CALOR_VERANO_MEDIA
    ANCHO = 7.3  # mismo ancho de zona moderada que la función original (35 - 27.7)
    UMBRAL_SEVERO = CENTRO - ANCHO  # 22.12
    return np.clip(
        np.where(
            tmax_verano >= CENTRO,
            1.0,
            np.where(
                tmax_verano >= UMBRAL_SEVERO,
                0.70 + 0.30 * (tmax_verano - UMBRAL_SEVERO) / ANCHO,
                0.40 + 0.30 * tmax_verano / UMBRAL_SEVERO,
            ),
        ),
        0.0,
        1.0,
    )


def _alpha_beta_por_media(
    media: np.ndarray, precision: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte una media objetivo y una precisión (kappa) a los parámetros
    (alpha, beta) de una Beta reparametrizada por media. Usado tanto por
    `_muestrear_beta_por_media` (muestreo directo con rng.beta) como por
    `simulate_yields_antitetico` (muestreo vía beta.ppf de uniformes
    antitéticos) — una sola fórmula, sin duplicar el clip ni el cálculo.
    """
    media = np.clip(media, 1e-4, 1 - 1e-4)
    alpha = media * precision
    beta = (1 - media) * precision
    return alpha, beta


def _muestrear_beta_por_media(
    media: np.ndarray, precision: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Muestrea una Beta(alpha, beta) reparametrizada por media y precisión.

    precision (kappa) controla la variabilidad residual alrededor de `media`:
    kappa alto -> el factor se acerca al valor determinístico de `media`;
    kappa bajo -> más ruido biológico no explicado por el clima.
    """
    alpha, beta = _alpha_beta_por_media(media, precision)
    return rng.beta(alpha, beta)


# ---------------------------------------------------------------------------
# Correlación empírica frío/calor vía cópula gaussiana (PLAN_TESIS.md,
# 2026-08-26; ver CORR_HORAS_FRIO_TMAX_VERANO más arriba para la
# fundamentación estadística de por qué está apagada por defecto)
#
# BUG CORREGIDO 2026-08-27 (detectado por /code-review ultra): una versión
# anterior de este mecanismo aplicaba la cópula sobre los FACTORES finales
# (factor_frio, factor_calor) -- es decir, sobre el ruido Beta residual que
# se agrega DESPUÉS de la función de transferencia climática -- en vez de
# sobre las variables climáticas crudas (horas_frio, tmax_verano), que es
# donde efectivamente se midió r=-0,2894 (Módulo 0). Como la mayor parte de
# la varianza de cada factor viene del driver climático (no del ruido
# residual), esa versión diluía la correlación lograda a ~-0,03 en vez de
# ~-0,29 -- un desajuste de capas, no un error de signo ni de fórmula. Ver
# PLAN_TESIS.md para el detalle y la validación de robustez re-corrida.
#
# Se correlacionan HORAS_FRIO y TMAX_VERANO (las variables climáticas
# crudas) vía cópula gaussiana. Como ambas tienen marginal Normal, la
# cópula se reduce a una combinación lineal de dos normales estándar por
# Cholesky (Z1, rho*Z1 + sqrt(1-rho²)*Z2) escaladas y centradas en su
# propia media/desvío -- no hace falta pasar por U=Phi(Z) ni por una PPF,
# a diferencia de cuando el marginal es una Beta. El ruido Beta que se
# aplica después sobre cada factor (`_muestrear_beta_por_media`) queda
# SIN CAMBIOS y SIN CORRELACIONAR en ambas ramas del flag -- es el mismo
# mecanismo independiente de siempre, solo que ahora recibe como entrada
# valores de horas/tmax_verano que ya vienen correlacionados cuando el
# flag está en True.
# ---------------------------------------------------------------------------

def _muestrear_clima_correlacionado(
    horas_frio_media: float, horas_frio_std: float,
    calor_verano_media: float, calor_verano_std: float,
    rho: float, shape: tuple[int, ...], rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Muestrea horas de frío y tmax de verano CORRELACIONADAS (r=`rho`) vía
    cópula gaussiana. Retorna (horas, tmax_verano), cada una de forma
    `shape`.
    """
    z1 = rng.standard_normal(shape)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(shape)
    horas = horas_frio_media + horas_frio_std * z1
    tmax_verano = calor_verano_media + calor_verano_std * z2
    return horas, tmax_verano


def _muestrear_clima_correlacionado_antitetico(
    horas_frio_media: float, horas_frio_std: float,
    calor_verano_media: float, calor_verano_std: float,
    rho: float, n: int, T: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Versión de `_muestrear_clima_correlacionado()` con reducción de
    varianza por variables antitéticas: `z1` y `z2_indep` se arman cada uno
    desde su propio par de uniformes antitéticos
    (`_generar_uniformes_antiteticos`) vía `norm.ppf`, ANTES de combinarlos
    por Cholesky -- mismo patrón que el resto del motor. Una combinación
    lineal de dos normales antitéticas (segunda mitad = -primera mitad) es
    a su vez antitética, así que `z2` conserva la propiedad aunque sea una
    mezcla de `z1` y `z2_indep`.
    """
    u1 = _generar_uniformes_antiteticos(n, (T,), rng)
    u2 = _generar_uniformes_antiteticos(n, (T,), rng)
    z1 = norm.ppf(u1)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * norm.ppf(u2)
    horas = horas_frio_media + horas_frio_std * z1
    tmax_verano = calor_verano_media + calor_verano_std * z2
    return horas, tmax_verano


def _esperanza_factor_frio(media: float, std: float) -> float:
    """E[factor de frío] con horas ~ Normal(media, std): integración numérica
    de la función de transferencia (el ruido Beta posterior de
    `_muestrear_beta_por_media` no cambia la media, por construcción:
    E[Beta(m*k, (1-m)*k)] = m)."""
    return integrate.quad(
        lambda h: float(_media_factor_frio(np.array([h]))[0]) * norm.pdf(h, media, std),
        media - 6 * std,
        media + 6 * std,
    )[0]


def _esperanza_factor_calor(media: float, std: float) -> float:
    """E[factor de calor] con tmax de verano ~ Normal(media, std). Mismo
    criterio que `_esperanza_factor_frio`."""
    return integrate.quad(
        lambda t: float(_media_factor_calor(np.array([t]))[0]) * norm.pdf(t, media, std),
        media - 6 * std,
        media + 6 * std,
    )[0]


def _producto_esperado_factores() -> float:
    """
    Producto de los valores esperados de los factores estocásticos de
    `simulate_yields()` que NO están normalizados a media 1 (frío, calor,
    supervivencia), bajo la calibración por defecto. La vecería no entra:
    desde la sincronización v2 su factor tiene media 1 por construcción
    (ver `_media_veceria()`). Se usa para derivar RENDIMIENTO_IDEAL_KG_HA a
    partir de RENDIMIENTO_VALIDADO_KG_HA — ver el docstring de esa constante.

    Supervivencia: media de 1 - Beta(alpha, beta) = 1 - alpha / (alpha + beta).
    """
    e_frio = _esperanza_factor_frio(_HORAS_FRIO_MEDIA, _HORAS_FRIO_STD)
    e_calor = _esperanza_factor_calor(_CALOR_VERANO_MEDIA, _CALOR_VERANO_STD)
    e_supervivencia = 1.0 - _PLANTAS_ALPHA / (_PLANTAS_ALPHA + _PLANTAS_BETA)

    return e_frio * e_calor * e_supervivencia


# Rendimiento IDEAL: constante de NORMALIZACIÓN derivada, no una afirmación
# agronómica directa. Es el valor que hay que poner como techo de la curva
# de maduración (curva_base) para que, después de multiplicar por los
# factores estocásticos (frío, calor, supervivencia; la vecería tiene media
# 1) en sus valores esperados, el rendimiento resultante converja al dato
# del Excel RENDIMIENTO_VALIDADO_KG_HA (3.000 kg/ha, ver esa constante más
# arriba). RENDIMIENTO_IDEAL_KG_HA nunca se observa en la práctica:
# requeriría que todos los factores dieran 1.0 simultáneamente (la
# supervivencia es estructuralmente siempre < 1).
RENDIMIENTO_IDEAL_KG_HA = RENDIMIENTO_VALIDADO_KG_HA / _producto_esperado_factores()


# ---------------------------------------------------------------------------
# Configuración del modelo
# ---------------------------------------------------------------------------

@dataclass
class ParametrosMC:
    """Parámetros configurables del modelo de Monte Carlo."""

    n_simulaciones: int = 10_000
    n_años: int = 20
    hectareas: float = HECTAREAS
    rendimiento_plena: float = RENDIMIENTO_IDEAL_KG_HA

    # Horas de frío — calibrado desde PARAMS_CLIMA_JOCOLI (Módulo 0)
    horas_frio_media: float = _HORAS_FRIO_MEDIA
    horas_frio_std: float = _HORAS_FRIO_STD

    # Calor estival (tmax media diaria, enero-febrero) — calibrado con datos
    # reales ERA5-Land (models=era5_land) 1990-2024 (Módulo 0, calcular_calor_verano)
    calor_verano_media: float = _CALOR_VERANO_MEDIA
    calor_verano_std: float = _CALOR_VERANO_STD

    # Precisión (kappa) del ruido Beta sobre los factores de frío/calor:
    # valores altos acercan el factor muestreado a su media determinística,
    # valores bajos agregan más variabilidad biológica no explicada por el
    # clima (microclima, estado sanitario, timing de heladas). Punto de
    # partida razonable, pendiente de calibrar con datos reales o inferencia
    # bayesiana (ver notas/PLAN_TESIS.md, Paso 4).
    precision_factor_frio: float = 15.0
    precision_factor_calor: float = 15.0

    # Correlación empírica frío/calor (r=-0.2894, no significativa al 5%,
    # ver CORR_HORAS_FRIO_TMAX_VERANO más arriba) -- default apagado, mismo
    # criterio que `capex_opex_estocastico`: el motor "oficial" no asume
    # dependencia entre frío y calor salvo que se pida explícito, dado que
    # la evidencia estadística es marginal. Pensado como análisis de
    # robustez (correr True vs. False y comparar), no como calibración.
    correlacionar_frio_calor: bool = False

    # Vecería: cadena de Markov sobre estado alto/bajo
    p_bajo_si_alto: float = _P_BAJO_SI_ALTO
    p_alto_si_bajo: float = _P_ALTO_SI_BAJO
    veceria_factor_min: float = _VECERIA_FACTOR_MIN
    veceria_factor_max: float = _VECERIA_FACTOR_MAX

    # Tasa de falla de plantas — Beta(alpha, beta); media ~9%
    plantas_alpha: float = _PLANTAS_ALPHA
    plantas_beta: float = _PLANTAS_BETA

    # Escenario de precio y tasa de descuento para VAN. La tasa es REAL en
    # USD (hoja Tasa del Excel v2: CAPM + riesgo país de Damodaran = 17,09%
    # nominal, pasado a real con Fisher contra 2,5% de inflación de EE.UU.),
    # coherente con flujos en USD constantes de 2026.
    escenario: Literal["pesimista", "base", "optimista"] = "base"
    tasa_descuento: float = _MODELO.supuestos["tasa_descuento_real"]

    # Generador de precio: "ar1" (AR(1) sobre retornos log, calibrado con
    # datos reales de FRED -- src/precio_estocastico.py) o "triangular"
    # (ESCENARIOS_PRECIO, independiente por año, sin memoria). Default "ar1"
    # porque es el modelo elegido -- calibrado con datos reales, ver
    # notas/plan_precio_historico.md -- desde que hasta ahora convivían tres
    # funciones run_monte_carlo*() con generadores distintos y sin que nadie
    # lo hubiera decidido a propósito por función (feedback de Eze,
    # notas/PLAN_TESIS.md). "triangular" queda disponible como modo
    # explícito de comparación contra el modelo naive, no como default en
    # ningún caller.
    modo_precio: Literal["ar1", "triangular"] = "ar1"

    # Activa la variabilidad estocástica de OPEX (multiplicador de OPEX --
    # ver simulate_opex_multiplicador() más abajo). El nombre conserva
    # "capex" por compatibilidad: desde Fase B el componente de CAPEX
    # (riego/pozo, simulate_capex_extra()) está retirado y devuelve cero
    # con o sin este flag -- ver src/costos.py. En False,
    # opex_multiplicador=1.0 para las 10.000 iteraciones, sin invocar
    # rng.triangular -- para comparar contra el modelo determinístico sin
    # recurrir a anchos "casi cero" que rng.triangular no acepta.
    capex_opex_estocastico: bool = True

    # Modo determinístico COMPLETO: reproduce el Excel v2 sin ninguna fuente
    # estocástica. Rendimiento = CURVA_EXCEL (vecería del Excel incluida) *
    # RENDIMIENTO_VALIDADO_KG_HA, sin clima, vecería simulada ni falla de
    # plantas; precio = trayectoria del Excel del `escenario` (ignora
    # `modo_precio` y `precio_params`); multiplicador de OPEX = 1. Las
    # `n_simulaciones` filas salen idénticas. Pensado para verificar el
    # motor contra el Excel (tests/test_consistencia_excel.py), no para
    # análisis de riesgo.
    deterministico: bool = False

    # Fuentes estocásticas reemplazadas por su valor esperado, para
    # descomponer la varianza del VAN por fuente (scripts/descomponer_varianza.py).
    # Valores: "precio" (trayectoria del Excel, que es E[precio] con c=0),
    # "clima" (factores de frío y calor en su media), "veceria" (factor 1) y
    # "plantas" (supervivencia media). Los sorteos se hacen igual y después
    # se pisan: el consumo de `rng` no cambia, así que las fuentes que quedan
    # activas reciben exactamente los mismos números en todas las corridas
    # (números aleatorios comunes). El OPEX se apaga con
    # `capex_opex_estocastico=False`.
    fuentes_fijas: frozenset[str] = frozenset()

    semilla: int = 42


# ---------------------------------------------------------------------------
# Funciones internas restantes
# ---------------------------------------------------------------------------

def _pi_bajo_estacionaria(params: ParametrosMC) -> float:
    """Probabilidad estacionaria del estado "bajo" de la cadena de vecería."""
    return params.p_bajo_si_alto / (params.p_bajo_si_alto + params.p_alto_si_bajo)


def _media_veceria(params: ParametrosMC) -> float:
    """
    E[factor de vecería crudo] bajo la distribución estacionaria: los años
    altos valen 1 y los bajos Uniforme(min, max). Se divide el factor crudo
    por este valor para que la vecería tenga media 1 (desvío de media cero):
    redistribuye la producción entre años sin cambiar el rendimiento
    esperado, que ya fija la curva (la meseta de CURVA_PRODUCCION es el
    promedio del Excel, que ya incluye años malos). La normalización usa los
    `params` de la corrida, así que sigue valiendo cuando Sobol o el dataset
    barren `p_bajo_si_alto` / `p_alto_si_bajo`: esos parámetros mueven la
    VARIANZA del rendimiento, no su media.
    """
    pi_bajo = _pi_bajo_estacionaria(params)
    factor_bajo_medio = (params.veceria_factor_min + params.veceria_factor_max) / 2
    return pi_bajo * factor_bajo_medio + (1 - pi_bajo)


def _simular_veceria(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Cadena de Markov binaria (año alto / año bajo) para modelar la alternancia.

    El estado del año 1 se sortea con la distribución ESTACIONARIA de la
    cadena (no 50/50): así el factor tiene la misma media en todos los años
    y la normalización por `_media_veceria()` da media 1 exacta en cada año.

    Retorna array (n_simulaciones, n_años) con factores multiplicadores de
    media 1.
    """
    n, T = params.n_simulaciones, params.n_años
    es_bajo = np.zeros((n, T), dtype=bool)
    es_bajo[:, 0] = rng.random(n) < _pi_bajo_estacionaria(params)

    for t in range(1, T):
        p_bajo = np.where(
            ~es_bajo[:, t - 1],
            params.p_bajo_si_alto,
            1.0 - params.p_alto_si_bajo,
        )
        es_bajo[:, t] = rng.random(n) < p_bajo

    factores_bajos = rng.uniform(
        params.veceria_factor_min,
        params.veceria_factor_max,
        (n, T),
    )
    return np.where(es_bajo, factores_bajos, 1.0) / _media_veceria(params)


FUENTES_ESTOCASTICAS = ("precio", "clima", "veceria", "plantas")


def _validar_fuentes_fijas(params: ParametrosMC) -> None:
    desconocidas = set(params.fuentes_fijas) - set(FUENTES_ESTOCASTICAS)
    if desconocidas:
        raise ValueError(
            f"fuentes_fijas admite {FUENTES_ESTOCASTICAS}, recibido {sorted(desconocidas)}"
        )


def _combinar_factores(
    params: ParametrosMC,
    curva_base: np.ndarray,
    factor_veceria: np.ndarray,
    factor_frio: np.ndarray,
    factor_calor: np.ndarray,
    supervivencia: np.ndarray,
) -> np.ndarray:
    """Producto de la curva por los factores, con las fuentes de
    `params.fuentes_fijas` reemplazadas por su valor esperado."""
    fijas = params.fuentes_fijas
    if "veceria" in fijas:
        factor_veceria = np.ones_like(factor_veceria)
    if "clima" in fijas:
        factor_frio = np.full_like(
            factor_frio, _esperanza_factor_frio(params.horas_frio_media, params.horas_frio_std)
        )
        factor_calor = np.full_like(
            factor_calor, _esperanza_factor_calor(params.calor_verano_media, params.calor_verano_std)
        )
    if "plantas" in fijas:
        supervivencia = np.full_like(
            supervivencia, 1.0 - params.plantas_alpha / (params.plantas_alpha + params.plantas_beta)
        )
    return np.maximum(
        curva_base * factor_veceria * factor_frio * factor_calor * supervivencia, 0.0
    )


def _fijar_precio(
    params: ParametrosMC, prices: np.ndarray, precio_params: ParametrosPrecioAR1 | None
) -> np.ndarray:
    """Si "precio" está en `params.fuentes_fijas`, reemplaza los precios
    simulados por la trayectoria base: la de `precio_params` si la trae, si
    no la del Excel del escenario."""
    if "precio" not in params.fuentes_fijas:
        return prices
    if precio_params is not None and params.modo_precio == "ar1":
        trayectoria = precio_params.trayectoria_base(params.n_años)
    else:
        trayectoria = _MODELO.trayectoria_precio(params.escenario, params.n_años)
    return np.broadcast_to(trayectoria, prices.shape).copy()


# ---------------------------------------------------------------------------
# Reducción de varianza — variables antitéticas (PLAN_TESIS.md, Paso 2)
# ---------------------------------------------------------------------------

def _generar_uniformes_antiteticos(
    n: int, shape_por_muestra: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """
    Genera `n` uniformes(0,1) usando variables antitéticas: la primera mitad
    (filas 0 a n//2 - 1) son uniformes frescos U, la segunda mitad (filas
    n//2 a n-1) son sus complementos 1-U, fila a fila emparejadas
    (fila i <-> fila i + n//2). Esto induce correlación negativa entre cada
    simulación y su par antitético, reduciendo la varianza del estimador
    (la media) para el mismo N total de evaluaciones, sin cambiar el valor
    esperado.

    Si `n` es impar, la última fila queda como un uniforme fresco sin
    pareja: no participa de la reducción de varianza, pero mantiene la
    forma de salida en `n` filas.

    Parámetros
    ----------
    n : int
        Cantidad total de muestras a generar (pares + resto si es impar).
    shape_por_muestra : tuple[int, ...]
        Forma de cada muestra individual, p. ej. `(n_años,)` o `(1,)`.
    rng : np.random.Generator

    Retorna
    -------
    np.ndarray de forma (n, *shape_por_muestra), valores en (0, 1).
    """
    n_mitad = n // 2
    u = rng.random((n_mitad, *shape_por_muestra))
    pares = np.concatenate([u, 1.0 - u], axis=0)
    if n % 2 == 1:
        extra = rng.random((1, *shape_por_muestra))
        pares = np.concatenate([pares, extra], axis=0)
    return pares


def simulate_yields_antitetico(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Versión de `simulate_yields()` con reducción de varianza por variables
    antitéticas. Mismas cinco fuentes de variabilidad y mismas distribuciones,
    pero cada una se arma transformando uniformes antitéticos
    (`_generar_uniformes_antiteticos`) vía la función inversa (PPF) de la
    distribución correspondiente, en vez de muestrear directo con los
    métodos de `np.random.Generator`.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años) en kg/ha.
    """
    n, T = params.n_simulaciones, params.n_años

    curva_base = np.array(
        [CURVA_PRODUCCION[a] * params.rendimiento_plena for a in range(1, T + 1)]
    ).reshape(1, T)

    # --- Vecería: cadena de Markov, uniformes antitéticos aplicados en cada paso ---
    u_estado = _generar_uniformes_antiteticos(n, (T,), rng)
    u_magnitud = _generar_uniformes_antiteticos(n, (T,), rng)

    es_bajo = np.zeros((n, T), dtype=bool)
    es_bajo[:, 0] = u_estado[:, 0] < _pi_bajo_estacionaria(params)
    for t in range(1, T):
        p_bajo = np.where(
            ~es_bajo[:, t - 1],
            params.p_bajo_si_alto,
            1.0 - params.p_alto_si_bajo,
        )
        es_bajo[:, t] = u_estado[:, t] < p_bajo

    # Uniforme(min, max) = min + (max - min) * U -- PPF trivial, lineal
    factores_bajos = (
        params.veceria_factor_min
        + (params.veceria_factor_max - params.veceria_factor_min) * u_magnitud
    )
    factor_veceria = np.where(es_bajo, factores_bajos, 1.0) / _media_veceria(params)

    # --- Frío y calor: variables climáticas correlacionadas (cópula
    # gaussiana) o independientes; el ruido Beta sobre cada factor es
    # SIEMPRE independiente (ver comentario más arriba, junto a
    # `_muestrear_clima_correlacionado_antitetico`). La rama `False`
    # preserva EXACTAMENTE el orden de consumo de `rng` de antes de agregar
    # esta fuente, para no cambiar en silencio ningún resultado ya
    # documentado con el flag apagado (default).
    if params.correlacionar_frio_calor:
        horas, tmax_verano = _muestrear_clima_correlacionado_antitetico(
            params.horas_frio_media, params.horas_frio_std,
            params.calor_verano_media, params.calor_verano_std,
            CORR_HORAS_FRIO_TMAX_VERANO, n, T, rng,
        )
        media_ff = _media_factor_frio(horas)
        alpha_ff, beta_ff = _alpha_beta_por_media(media_ff, params.precision_factor_frio)
        u_frio = _generar_uniformes_antiteticos(n, (T,), rng)
        factor_frio = beta_dist.ppf(u_frio, alpha_ff, beta_ff)

        media_fc = _media_factor_calor(tmax_verano)
        alpha_fc, beta_fc = _alpha_beta_por_media(media_fc, params.precision_factor_calor)
        u_calor = _generar_uniformes_antiteticos(n, (T,), rng)
        factor_calor = beta_dist.ppf(u_calor, alpha_fc, beta_fc)
    else:
        u_horas = _generar_uniformes_antiteticos(n, (T,), rng)
        horas = norm.ppf(u_horas, loc=params.horas_frio_media, scale=params.horas_frio_std)
        media_ff = _media_factor_frio(horas)

        alpha_ff, beta_ff = _alpha_beta_por_media(media_ff, params.precision_factor_frio)
        u_frio = _generar_uniformes_antiteticos(n, (T,), rng)
        factor_frio = beta_dist.ppf(u_frio, alpha_ff, beta_ff)

        u_tmax = _generar_uniformes_antiteticos(n, (T,), rng)
        tmax_verano = norm.ppf(u_tmax, loc=params.calor_verano_media, scale=params.calor_verano_std)
        media_fc = _media_factor_calor(tmax_verano)
        alpha_fc, beta_fc = _alpha_beta_por_media(media_fc, params.precision_factor_calor)
        u_calor = _generar_uniformes_antiteticos(n, (T,), rng)
        factor_calor = beta_dist.ppf(u_calor, alpha_fc, beta_fc)

    # Falla de plantas: mismo valor para toda la vida del cultivo (decisión de campo)
    u_supervivencia = _generar_uniformes_antiteticos(n, (1,), rng)
    supervivencia = 1.0 - beta_dist.ppf(u_supervivencia, params.plantas_alpha, params.plantas_beta)

    return _combinar_factores(
        params, curva_base, factor_veceria, factor_frio, factor_calor, supervivencia
    )


def simulate_prices_antitetico(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Versión de `simulate_prices()` con reducción de varianza por variables
    antitéticas: transforma uniformes antitéticos vía la PPF de la
    distribución triangular (scipy.stats.triang) en vez de `rng.triangular`.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    low, mode, high = ESCENARIOS_PRECIO[params.escenario]
    trayectoria = _MODELO.trayectoria_precio(params.escenario, params.n_años)
    u = _generar_uniformes_antiteticos(params.n_simulaciones, (params.n_años,), rng)
    c = (mode - low) / (high - low)  # parámetro de forma de scipy.stats.triang
    return trayectoria.reshape(1, -1) * triang_dist.ppf(u, c, loc=low, scale=high - low)


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def simulate_yields(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Simula el rendimiento en kg/ha para cada iteración y año del proyecto.

    Combina cinco fuentes de variabilidad:
    1. Curva de maduración determinista (años 1–20)
    2. Vecería (alternancia productiva) — cadena de Markov, media 1
    3. Déficit de horas de frío respecto a la media histórica real del sitio
       — media determinística + ruido Beta(media, precision_factor_frio)
    4. Déficit de calor estival (ene-feb) respecto a la media histórica real
       — media determinística + ruido Beta(media, precision_factor_calor)
    5. Tasa de falla de plantas — Beta(alpha, beta)

    Si `params.correlacionar_frio_calor` es True, las variables climáticas
    de las que salen los factores 3 y 4 (horas de frío, tmax de verano) se
    muestrean CORRELACIONADAS (r=`CORR_HORAS_FRIO_TMAX_VERANO`) vía cópula
    gaussiana en vez de independientes -- ver esa constante y
    `_muestrear_clima_correlacionado()` para la fundamentación y el
    mecanismo. El ruido Beta que se aplica sobre cada factor
    (`_muestrear_beta_por_media`) es siempre independiente, en ambas ramas
    del flag -- solo cambia si horas/tmax_verano vienen correlacionadas o
    no. Default apagado: la correlación empírica no es significativa al 5%
    (ver notas/PLAN_TESIS.md, 2026-08-26).

    LIMITACIÓN CONOCIDA de la comparación True vs. False con semilla fija:
    aun con este mecanismo, activar el flag consume el stream de `rng` en
    un orden distinto al de la rama `False` (el propio Cholesky mezcla las
    dos normales estándar antes de escalarlas), así que una comparación
    determinística no aísla EXCLUSIVAMENTE el efecto de la correlación --
    para eso hace falta comparar distribuciones agregadas (como en la
    tabla de robustez de PLAN_TESIS.md), no realizaciones individuales
    pareadas por posición.

    Parámetros
    ----------
    params : ParametrosMC
        Configuración del modelo.
    rng : np.random.Generator
        Generador de números aleatorios (para reproducibilidad).

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años) en kg/ha.
    """
    n, T = params.n_simulaciones, params.n_años

    curva_base = np.array(
        [CURVA_PRODUCCION[a] * params.rendimiento_plena for a in range(1, T + 1)]
    ).reshape(1, T)

    factor_veceria = _simular_veceria(params, rng)

    # Frío y calor: variables climáticas correlacionadas (cópula gaussiana)
    # o independientes; el ruido Beta sobre cada factor es SIEMPRE
    # independiente (ver comentario junto a `_muestrear_clima_correlacionado`
    # más arriba). La rama `False` preserva EXACTAMENTE el orden de consumo
    # de `rng` previo a agregar esta fuente, para no cambiar en silencio
    # ningún resultado ya documentado con el flag apagado (default).
    if params.correlacionar_frio_calor:
        horas, tmax_verano = _muestrear_clima_correlacionado(
            params.horas_frio_media, params.horas_frio_std,
            params.calor_verano_media, params.calor_verano_std,
            CORR_HORAS_FRIO_TMAX_VERANO, (n, T), rng,
        )
        media_ff = _media_factor_frio(horas)
        factor_frio = _muestrear_beta_por_media(media_ff, params.precision_factor_frio, rng)

        media_fc = _media_factor_calor(tmax_verano)
        factor_calor = _muestrear_beta_por_media(media_fc, params.precision_factor_calor, rng)
    else:
        horas = rng.normal(params.horas_frio_media, params.horas_frio_std, (n, T))
        media_ff = _media_factor_frio(horas)
        factor_frio = _muestrear_beta_por_media(media_ff, params.precision_factor_frio, rng)

        tmax_verano = rng.normal(params.calor_verano_media, params.calor_verano_std, (n, T))
        media_fc = _media_factor_calor(tmax_verano)
        factor_calor = _muestrear_beta_por_media(media_fc, params.precision_factor_calor, rng)

    # Falla de plantas: mismo valor para toda la vida del cultivo (decisión de campo)
    supervivencia = 1.0 - rng.beta(params.plantas_alpha, params.plantas_beta, (n, 1))

    return _combinar_factores(
        params, curva_base, factor_veceria, factor_frio, factor_calor, supervivencia
    )


def simulate_prices(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Simula el precio de venta en USD/kg: trayectoria del Excel por un
    multiplicador triangular independiente por año (ver ESCENARIOS_PRECIO).

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    low, mode, high = ESCENARIOS_PRECIO[params.escenario]
    trayectoria = _MODELO.trayectoria_precio(params.escenario, params.n_años)
    return trayectoria.reshape(1, -1) * rng.triangular(
        low, mode, high, (params.n_simulaciones, params.n_años)
    )


def _despachar_precios(
    params: ParametrosMC,
    rng: np.random.Generator,
    precio_params: ParametrosPrecioAR1 | None,
) -> np.ndarray:
    """
    Despacha al generador de precio según `params.modo_precio` -- mismo
    patrón que `capex_opex_estocastico`/`correlacionar_frio_calor`: el flag
    decide DENTRO de la función compartida, no en funciones top-level
    paralelas. Usado por `run_monte_carlo()` (muestreo directo).

    Si `modo_precio="ar1"` y `precio_params` es None, se arma con
    `ParametrosPrecioAR1(escenario=params.escenario)` -- así `escenario`
    sigue siendo la única fuente de verdad de qué escenario correr, salvo
    que se pase `precio_params` explícito (p. ej. para barrer el drift, ver
    `src/dataset_ml.py`).
    """
    if params.modo_precio == "ar1":
        if precio_params is None:
            precio_params = ParametrosPrecioAR1(escenario=params.escenario)
        return simulate_prices_ar1(params.n_simulaciones, params.n_años, precio_params, rng)
    if params.modo_precio == "triangular":
        return simulate_prices(params, rng)
    raise ValueError(
        f"modo_precio debe ser 'ar1' o 'triangular', recibido {params.modo_precio!r}"
    )


def _despachar_precios_antitetico(
    params: ParametrosMC,
    rng: np.random.Generator,
    precio_params: ParametrosPrecioAR1 | None,
) -> np.ndarray:
    """
    Versión de `_despachar_precios()` con reducción de varianza por
    variables antitéticas. Usado por `run_monte_carlo_antitetico()` y por
    `run_monte_carlo_precio_historico()` (alias fino de la anterior con
    `modo_precio="ar1"` forzado -- ver esa función).
    """
    if params.modo_precio == "ar1":
        if precio_params is None:
            precio_params = ParametrosPrecioAR1(escenario=params.escenario)
        return simulate_prices_ar1_antitetico(
            params.n_simulaciones, params.n_años, precio_params, rng
        )
    if params.modo_precio == "triangular":
        return simulate_prices_antitetico(params, rng)
    raise ValueError(
        f"modo_precio debe ser 'ar1' o 'triangular', recibido {params.modo_precio!r}"
    )


def simulate_revenue(
    yields_kg_ha: np.ndarray,
    prices_usd_kg: np.ndarray,
    hectareas: float,
) -> np.ndarray:
    """
    Calcula los ingresos brutos en USD: rendimiento × precio × superficie.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    return yields_kg_ha * prices_usd_kg * hectareas


# ---------------------------------------------------------------------------
# CAPEX/OPEX estocásticos dentro de la simulación (PLAN_TESIS.md, 2026-08-26;
# componente de CAPEX retirado en Fase B, 2026-09-24 -- ver src/costos.py)
#
# Hasta acá, `src/costos.py` era 100% determinístico dentro de una corrida de
# Monte Carlo: el único lugar donde CAPEX/OPEX variaban era el barrido LHS de
# `dataset_ml.py` (`capex_extra_pct`), como parámetro FIJO por punto del
# diseño, no como ruido dentro de las 10.000 iteraciones que arman la
# distribución del VAN. Estas cuatro funciones agregaron esa fuente de
# variabilidad -- desde Fase B, las dos de CAPEX (`simulate_capex_extra*`)
# quedaron retiradas (devuelven cero incondicionalmente: riego y pozo ya
# tienen precio fijo real, ver src/costos.py), y solo las dos de OPEX
# (`simulate_opex_multiplicador*`) siguen agregando variabilidad real, según
# `OPEX_VARIACION_PCT` en src/costos.py.
# ---------------------------------------------------------------------------

def _triangular_o_constante(
    rng: np.random.Generator, low: float, mode: float, high: float, size: int
) -> np.ndarray:
    """
    `rng.triangular(low, mode, high, size)`, o un array constante en `mode`
    si el ancho es (numéricamente) cero -- `rng.triangular` levanta
    `ValueError: left == right` en ese caso en vez de tratarlo como el caso
    límite determinístico que en realidad es.
    """
    if np.isclose(low, high):
        return np.full(size, mode)
    return rng.triangular(low, mode, high, size)


def _triangular_ppf_o_constante(
    u: np.ndarray, low: float, mode: float, high: float
) -> np.ndarray:
    """
    Igual que `_triangular_o_constante`, pero para la variante antitética
    (PPF de `scipy.stats.triang` en vez de `rng.triangular`): si el ancho es
    cero, `(mode - low) / (high - low)` sería una división por cero.
    """
    if np.isclose(low, high):
        return np.full(u.shape, mode)
    c = (mode - low) / (high - low)
    return triang_dist.ppf(u, c, loc=low, scale=high - low)


def simulate_capex_extra(
    n_simulaciones: int,
    costos: ParametrosCostos,
    rng: np.random.Generator,
    estocastico: bool = True,
) -> np.ndarray:
    """
    CAPEX estocástico adicional, en USD -- RETIRADO desde Fase B
    (notas/PLAN_TESIS.md, 2026-09-24): devuelve cero incondicionalmente.

    Hasta la resincronización con el plan de negocio 120 ha, acá se
    muestreaban dos triangulares (riego, pozo de agua) calibradas con
    estimaciones de mercado -- ver `src/costos.py`, sección "CAPEX
    estocástico: RETIRADO", para el detalle de por qué: pozo y bombas ahora
    tienen precio FIJO real confirmado en `costos.capex_inicial_ha`, así que
    seguir simulándolos duplicaría ese costo. No queda ningún ítem de CAPEX
    sin cotizar con rango de mercado hoy.

    Se mantienen la función y su firma (`costos`, `rng`, `estocastico` sin
    usar a propósito) en vez de eliminarla, para no romper
    `run_monte_carlo()`/`_orquestar_resultado()` ni la columna
    `capex_extra_estocastico_usd` -- ver `simulate_capex_extra_antitetico()`,
    idéntico criterio.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones,), en USD -- siempre ceros.
    """
    return np.zeros(n_simulaciones)


def simulate_capex_extra_antitetico(
    n_simulaciones: int,
    costos: ParametrosCostos,
    rng: np.random.Generator,
    estocastico: bool = True,
) -> np.ndarray:
    """
    Versión "antitética" de `simulate_capex_extra()` -- también RETIRADO,
    también devuelve cero incondicionalmente. Ver esa función para el
    detalle; se mantiene como función separada solo para preservar la
    interfaz de `run_monte_carlo_antitetico()`.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones,), en USD -- siempre ceros.
    """
    return np.zeros(n_simulaciones)


def simulate_opex_multiplicador(
    n_simulaciones: int,
    costos: ParametrosCostos,
    rng: np.random.Generator,
    estocastico: bool = True,
) -> np.ndarray:
    """
    Multiplicador estocástico de OPEX: triangular simétrica centrada en 1.0,
    de ancho ±`costos.opex_variacion_pct` (ver esa constante en
    src/costos.py para la justificación del valor). UN solo draw por
    simulación, aplicado a todos los años de esa simulación en
    `flujo_caja_neto()` -- representa "este proyecto en particular resultó
    más/menos caro de lo presupuestado", no ruido año a año.

    Parámetros
    ----------
    estocastico : bool
        Si es False, devuelve directamente 1.0 (sin efecto) para las
        `n_simulaciones` iteraciones, sin llamar a `rng.triangular`.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones,).
    """
    if not estocastico:
        return np.ones(n_simulaciones)

    d = costos.opex_variacion_pct
    return _triangular_o_constante(rng, 1.0 - d, 1.0, 1.0 + d, n_simulaciones)


def simulate_opex_multiplicador_antitetico(
    n_simulaciones: int,
    costos: ParametrosCostos,
    rng: np.random.Generator,
    estocastico: bool = True,
) -> np.ndarray:
    """
    Versión de `simulate_opex_multiplicador()` con reducción de varianza por
    variables antitéticas. Al ser una triangular SIMÉTRICA (mode = punto
    medio de [low, high]), el parámetro de forma de `scipy.stats.triang` es
    siempre c=0.5, sin depender de `opex_variacion_pct`. Ver
    `simulate_opex_multiplicador()` para el significado de `estocastico`.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones,).
    """
    if not estocastico:
        return np.ones(n_simulaciones)

    d = costos.opex_variacion_pct
    u = _generar_uniformes_antiteticos(n_simulaciones, (1,), rng).ravel()
    return _triangular_ppf_o_constante(u, 1.0 - d, 1.0, 1.0 + d)


def _componentes_deterministicos(
    params: ParametrosMC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (yields, prices, capex_extra, opex_multiplicador) del modo determinístico
    -- ver `ParametrosMC.deterministico`. No consume `rng`.
    """
    n, T = params.n_simulaciones, params.n_años
    curva = np.array([CURVA_EXCEL[a] for a in range(1, T + 1)]) * RENDIMIENTO_VALIDADO_KG_HA
    trayectoria = _MODELO.trayectoria_precio(params.escenario, T)
    return (
        np.tile(curva, (n, 1)),
        np.tile(trayectoria, (n, 1)),
        np.zeros(n),
        np.ones(n),
    )


def _resolver_params_costos(
    params: ParametrosMC | None, costos: ParametrosCostos | None
) -> tuple[ParametrosMC, ParametrosCostos]:
    """
    Aplica los defaults de `params`/`costos` y sincroniza `hectareas` entre
    ambos. Usado tanto por `run_monte_carlo()` como por
    `run_monte_carlo_antitetico()`.
    """
    if params is None:
        params = ParametrosMC()
    if costos is None:
        costos = ParametrosCostos(hectareas=params.hectareas)
    elif costos.hectareas != params.hectareas:
        costos.hectareas = params.hectareas
    return params, costos


def _orquestar_resultado(
    yields: np.ndarray,
    prices: np.ndarray,
    params: ParametrosMC,
    costos: ParametrosCostos,
    capex_extra_usd: np.ndarray,
    opex_multiplicador: np.ndarray,
) -> pd.DataFrame:
    """
    Arma el DataFrame de resultados (ingresos, OPEX, EBITDA, Ganancias,
    flujo neto, VAN neto) a partir de arrays de rendimiento, precio, CAPEX
    extra y multiplicador de OPEX ya simulados. El flujo replica la hoja
    Flujo del Excel v2 (ver `flujo_caja_neto()` en src/costos.py): flujo
    libre del proyecto después de Ganancias, sin financiamiento, con el
    valor residual de la tierra sumado en el último año. Usado por las tres `run_monte_carlo*()` — la única
    diferencia entre ellas es cómo se generan esos cuatro arrays (muestreo
    directo vs. variables antitéticas).

    Parámetros
    ----------
    capex_extra_usd : np.ndarray
        Forma (n_simulaciones,). CAPEX estocástico adicional -- siempre cero
        desde Fase B, ver `simulate_capex_extra()`. Se SUMA a
        `costos.capex_inicial` (el componente fijo) para obtener el CAPEX
        total de cada simulación (sin efecto numérico mientras sea cero).
    opex_multiplicador : np.ndarray
        Forma (n_simulaciones,). Ver `simulate_opex_multiplicador()`.
    """
    n, T = yields.shape

    produccion_kg = yields * params.hectareas
    revenue = simulate_revenue(yields, prices, params.hectareas)
    opex = opex_total(produccion_kg, revenue, costos, opex_multiplicador)
    flujo = flujo_caja_neto(revenue, opex, costos)
    flujo_neto = flujo["flujo_neto"]

    años = np.arange(1, params.n_años + 1)
    factores_descuento = (1 / (1 + params.tasa_descuento) ** años).reshape(1, -1)
    vp_neto = flujo_neto * factores_descuento

    capex_total_usd = costos.capex_inicial + capex_extra_usd  # (n,)
    van_neto_acum = -capex_total_usd.reshape(n, 1) + np.cumsum(vp_neto, axis=1)

    return pd.DataFrame({
        "simulacion":       np.repeat(np.arange(n), T),
        "año":              np.tile(años, n),
        "rendimiento_kg_ha": yields.ravel(),
        "precio_usd_kg":    prices.ravel(),
        "ingreso_usd":      revenue.ravel(),
        "opex_usd":         opex.ravel(),
        "ebitda_usd":       flujo["ebitda"].ravel(),
        "opex_pre_activado_usd": flujo["opex_pre_activado"].ravel(),
        "amortizacion_implantacion_usd": flujo["amortizacion_implantacion"].ravel(),
        "base_imponible_usd": flujo["base_imponible"].ravel(),
        "quebrantos_utilizados_usd": flujo["quebrantos_utilizados"].ravel(),
        "impuesto_ganancias_usd": flujo["impuesto_ganancias"].ravel(),
        "flujo_neto_usd":   flujo_neto.ravel(),
        "vp_neto_usd":      vp_neto.ravel(),
        "van_neto_usd":     van_neto_acum.ravel(),
        "capex_extra_estocastico_usd": np.repeat(capex_extra_usd, T),
    })


def run_monte_carlo(
    params: ParametrosMC | None = None,
    costos: ParametrosCostos | None = None,
    precio_params: ParametrosPrecioAR1 | None = None,
) -> pd.DataFrame:
    """
    Orquesta la simulación completa y retorna los resultados en formato tabular.

    El VAN se calcula sobre el flujo libre del proyecto (ingresos - OPEX -
    Ganancias + valor residual en el año 20, ver src/costos.py), en USD
    constantes y a tasa real. El CAPEX inicial se descuenta en el año 0 (factor 1.0):
    solo el componente fijo (`costos.capex_inicial`) tiene efecto -- el
    estocástico (`simulate_capex_extra()`) devuelve cero desde Fase B, ver
    src/costos.py. El OPEX de cada año se escala por un multiplicador
    estocástico único por simulación (`simulate_opex_multiplicador()`).

    El precio se genera según `params.modo_precio` ("ar1", default, o
    "triangular") -- ver `_despachar_precios()`.

    Parámetros
    ----------
    params : ParametrosMC, opcional
        Configuración del modelo de rendimientos. Si es None usa los valores
        por defecto.
    costos : ParametrosCostos, opcional
        Configuración de costos (CAPEX/OPEX por hectárea, ver src/costos.py).
        Si es None se crea con `hectareas=params.hectareas`. Si se pasa
        explícito y su `.hectareas` no coincide con `params.hectareas`, se
        sincroniza automáticamente (se pisa `costos.hectareas`).
    precio_params : ParametrosPrecioAR1, opcional
        Solo tiene efecto con `params.modo_precio="ar1"`. Si es None se arma
        con `ParametrosPrecioAR1(escenario=params.escenario)`.

    Retorna
    -------
    pd.DataFrame con columnas:
        simulacion, año, rendimiento_kg_ha, precio_usd_kg,
        ingreso_usd, opex_usd, ebitda_usd, opex_pre_activado_usd,
        amortizacion_implantacion_usd, base_imponible_usd,
        quebrantos_utilizados_usd,
        impuesto_ganancias_usd,
        flujo_neto_usd, vp_neto_usd, van_neto_usd,
        capex_extra_estocastico_usd
    """
    params, costos = _resolver_params_costos(params, costos)
    if params.deterministico:
        yields, prices, capex_extra, opex_mult = _componentes_deterministicos(params)
        return _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)
    _validar_fuentes_fijas(params)
    rng = np.random.default_rng(params.semilla)

    yields = simulate_yields(params, rng)
    prices = _fijar_precio(params, _despachar_precios(params, rng, precio_params), precio_params)
    capex_extra = simulate_capex_extra(
        params.n_simulaciones, costos, rng, params.capex_opex_estocastico
    )
    opex_mult = simulate_opex_multiplicador(
        params.n_simulaciones, costos, rng, params.capex_opex_estocastico
    )

    return _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)


def run_monte_carlo_antitetico(
    params: ParametrosMC | None = None,
    costos: ParametrosCostos | None = None,
    precio_params: ParametrosPrecioAR1 | None = None,
) -> pd.DataFrame:
    """
    Igual que `run_monte_carlo()` (misma interfaz, mismas columnas de
    salida, mismo despacho de precio según `params.modo_precio`), pero
    generando rendimiento, precio, CAPEX extra y multiplicador de OPEX con
    reducción de varianza por variables antitéticas
    (`simulate_yields_antitetico`, `_despachar_precios_antitetico`,
    `simulate_capex_extra_antitetico`, `simulate_opex_multiplicador_antitetico`)
    en vez de muestreo directo.

    Para el mismo `n_simulaciones`, el valor esperado de cada columna debería
    ser similar al de `run_monte_carlo()` — lo que cambia es la varianza del
    estimador (la media), no el resultado esperado.
    """
    params, costos = _resolver_params_costos(params, costos)
    if params.deterministico:
        yields, prices, capex_extra, opex_mult = _componentes_deterministicos(params)
        return _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)
    _validar_fuentes_fijas(params)
    rng = np.random.default_rng(params.semilla)

    yields = simulate_yields_antitetico(params, rng)
    prices = _fijar_precio(
        params, _despachar_precios_antitetico(params, rng, precio_params), precio_params
    )
    capex_extra = simulate_capex_extra_antitetico(
        params.n_simulaciones, costos, rng, params.capex_opex_estocastico
    )
    opex_mult = simulate_opex_multiplicador_antitetico(
        params.n_simulaciones, costos, rng, params.capex_opex_estocastico
    )

    return _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)


def run_monte_carlo_precio_historico(
    params: ParametrosMC | None = None,
    costos: ParametrosCostos | None = None,
    precio_params: ParametrosPrecioAR1 | None = None,
) -> pd.DataFrame:
    """
    Alias fino de `run_monte_carlo_antitetico()` con `modo_precio="ar1"`
    forzado. Se mantiene solo por compatibilidad con callers existentes
    (`src/dataset_ml.py`, que barre `precio_params.c` por LHS); no agrega
    lógica propia -- desde que "ar1" es el default de `ParametrosMC`, esta
    función y `run_monte_carlo_antitetico(params, costos, precio_params)` son
    equivalentes. Si `params` trae `modo_precio="triangular"` explícito, se
    lo pisa a "ar1" (esta función siempre fue AR(1); no tiene sentido que un
    caller la llame pidiendo el modo contrario).
    """
    if params is None:
        params = ParametrosMC()
    if params.modo_precio != "ar1":
        params = replace(params, modo_precio="ar1")
    return run_monte_carlo_antitetico(params, costos, precio_params)


def _tir_vectorizada(
    flujos_con_capex: np.ndarray,
    tasa_inicial: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    TIR de cada fila de `flujos_con_capex` (n_simulaciones, n_años + 1, con
    el año 0 incluido) mediante Newton-Raphson vectorizado.

    Retorna NaN en las filas que no convergen (p. ej. si el flujo nunca
    cambia de signo, la TIR no está definida). Una fila cuenta como
    convergida solo si, al final, la tasa quedó estrictamente dentro de los
    límites del clip [-0.99, 10] y el VAN a esa tasa es ~0 (relativo a la
    suma de |flujos|). Sin esa verificación, una fila que diverge queda
    clavada en el límite (el clip anula el paso) y se marcaba como
    "convergida" con TIR = 1000%: pasaba en simulaciones que nunca
    recuperan la inversión, e inflaba `tir_media` en src/dataset_ml.py.
    """
    n, T = flujos_con_capex.shape
    t = np.arange(T).reshape(1, T)
    r = np.full(n, tasa_inicial)
    limite_inf, limite_sup = -0.99, 10.0

    def _van(tasa: np.ndarray) -> np.ndarray:
        return np.sum(flujos_con_capex / (1.0 + tasa.reshape(-1, 1)) ** t, axis=1)

    paso = np.zeros(n)
    for _ in range(max_iter):
        npv = _van(r)
        dnpv = np.sum(
            -t * flujos_con_capex / (1.0 + r.reshape(-1, 1)) ** (t + 1), axis=1
        )
        dnpv = np.where(np.abs(dnpv) < 1e-12, np.nan, dnpv)
        paso = np.nan_to_num(npv / dnpv, nan=0.0)
        r = np.clip(r - paso, limite_inf, limite_sup)

    escala = np.abs(flujos_con_capex).sum(axis=1)
    convergio = (
        (np.abs(paso) < tol)
        & (r > limite_inf)
        & (r < limite_sup)
        & (np.abs(_van(r)) <= 1e-8 * escala)
    )
    r = np.where(convergio, r, np.nan)

    # Respaldo por bisección (brentq) para las filas donde Newton no
    # converge desde `tasa_inicial` pero el VAN sí cambia de signo dentro de
    # los límites: con años negativos después de la primera cosecha (precio
    # bajo + vecería) el VAN puede no ser monótono cerca de la tasa inicial.
    # Son pocas filas, el loop en Python no pesa.
    a, b = limite_inf + 1e-6, limite_sup
    for i in np.flatnonzero(~convergio):
        fila = flujos_con_capex[i]
        van_fila = lambda x: float(np.sum(fila / (1.0 + x) ** t[0]))  # noqa: E731
        if np.sign(van_fila(a)) != np.sign(van_fila(b)):
            r[i] = brentq(van_fila, a, b, xtol=tol)
    return r


def resumen_financiero(df: pd.DataFrame, costos: ParametrosCostos) -> pd.DataFrame:
    """
    Resumen por simulación: VAN neto final, TIR y año de recupero.

    Parámetros
    ----------
    df : pd.DataFrame
        Salida de `run_monte_carlo()` (requiere columnas simulacion, año,
        flujo_neto_usd, van_neto_usd, capex_extra_estocastico_usd).
    costos : ParametrosCostos
        Debe ser el mismo objeto (mismo `hectareas`, mismo CAPEX inicial fijo)
        usado para generar `df` en `run_monte_carlo()`. Es obligatorio y sin
        default a propósito: un default silencioso acá (p. ej. hectareas=50)
        daría un CAPEX incorrecto si `df` se generó con otra superficie. El
        componente ESTOCÁSTICO del CAPEX no está en `costos` -- varía por
        simulación, aunque desde Fase B sea siempre cero -- así que se toma
        de la columna `capex_extra_estocastico_usd` de `df`, no de `costos`.

    Retorna
    -------
    pd.DataFrame con columnas: simulacion, van_neto_usd, tir, año_recupero.
    `año_recupero` es NaN si el proyecto no recupera la inversión dentro
    del horizonte simulado.
    """
    tabla_flujos = df.pivot(index="simulacion", columns="año", values="flujo_neto_usd")
    n = tabla_flujos.shape[0]
    capex_extra = (
        df.groupby("simulacion")["capex_extra_estocastico_usd"]
        .first()
        .reindex(tabla_flujos.index)
        .values
    )
    flujos_con_capex = np.hstack([
        (-costos.capex_inicial - capex_extra).reshape(n, 1),
        tabla_flujos.values,
    ])

    tir = _tir_vectorizada(flujos_con_capex)

    acumulado = np.cumsum(flujos_con_capex, axis=1)
    recupero_mask = acumulado >= 0
    tiene_recupero = recupero_mask.any(axis=1)
    año_recupero = np.where(tiene_recupero, recupero_mask.argmax(axis=1), np.nan)

    van_neto_final = df.groupby("simulacion")["van_neto_usd"].last().values

    return pd.DataFrame({
        "simulacion":   tabla_flujos.index.values,
        "van_neto_usd": van_neto_final,
        "tir":          tir,
        "año_recupero": año_recupero,
    })
