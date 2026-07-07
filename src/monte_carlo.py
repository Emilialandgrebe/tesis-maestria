"""Simulación de Monte Carlo para rendimientos e ingresos del cultivo de pistacho."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.costos import ParametrosCostos, costo_operativo_anual, flujo_caja_neto

# ---------------------------------------------------------------------------
# Constantes del plan de negocios
# ---------------------------------------------------------------------------

HECTAREAS = 25.0
RENDIMIENTO_PLENA_KG_HA = 3_000  # kg/ha en plena producción (año 12+) — techo validado

# Fracción del rendimiento pleno por año del proyecto (1–20)
_CURVA_BASE: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00,
    6: 0.10, 7: 0.20, 8: 0.40, 9: 0.60, 10: 0.75,
    11: 0.85, 12: 0.95,
}
CURVA_PRODUCCION: dict[int, float] = {
    año: _CURVA_BASE.get(año, 1.00) for año in range(1, 21)
}

# Escenarios de precio (min, moda, max) en USD/kg — distribución triangular
ESCENARIOS_PRECIO: dict[str, tuple[float, float, float]] = {
    "pesimista": (6.0,  7.0,  8.5),
    "base":      (8.0,  9.5, 11.5),
    "optimista": (11.0, 13.0, 15.0),
}


# ---------------------------------------------------------------------------
# Configuración del modelo
# ---------------------------------------------------------------------------

@dataclass
class ParametrosMC:
    """Parámetros configurables del modelo de Monte Carlo."""

    n_simulaciones: int = 10_000
    n_años: int = 20
    hectareas: float = HECTAREAS
    rendimiento_plena: float = RENDIMIENTO_PLENA_KG_HA

    # Horas de frío — calibrado desde PARAMS_CLIMA_JOCOLI (Módulo 0)
    horas_frio_media: float = 984.3
    horas_frio_std: float = 212.7

    # Calor estival (tmax media diaria, enero-febrero) — calibrado con datos
    # reales ERA5-Land / Open-Meteo 1990-2024 (Módulo 0, calcular_calor_verano)
    calor_verano_media: float = 29.87
    calor_verano_std: float = 1.98

    # Precisión (kappa) del ruido Beta sobre los factores de frío/calor:
    # valores altos acercan el factor muestreado a su media determinística,
    # valores bajos agregan más variabilidad biológica no explicada por el
    # clima (microclima, estado sanitario, timing de heladas). Punto de
    # partida razonable, pendiente de calibrar con datos reales o inferencia
    # bayesiana (ver notas/PLAN_TESIS.md, Paso 4).
    precision_factor_frio: float = 15.0
    precision_factor_calor: float = 15.0

    # Vecería: cadena de Markov sobre estado alto/bajo
    p_bajo_si_alto: float = 0.65   # P(año bajo | año previo fue alto)
    p_alto_si_bajo: float = 0.80   # P(año alto | año previo fue bajo)
    veceria_factor_min: float = 0.60  # multiplicador mínimo en año bajo
    veceria_factor_max: float = 0.70  # multiplicador máximo en año bajo

    # Tasa de falla de plantas — Beta(alpha, beta); media ~9%
    plantas_alpha: float = 2.0
    plantas_beta: float = 20.0

    # Escenario de precio y tasa de descuento para VAN
    escenario: Literal["pesimista", "base", "optimista"] = "base"
    tasa_descuento: float = 0.08

    semilla: int = 42


# ---------------------------------------------------------------------------
# Funciones internas
# ---------------------------------------------------------------------------

def _media_factor_frio(horas: np.ndarray) -> np.ndarray:
    """
    Función de transferencia: horas de frío acumuladas → factor de rendimiento
    ESPERADO (media determinística; el ruido biológico se agrega después con
    `_muestrear_beta_por_media`).

    El punto de referencia (factor = 1.00) es la MEDIA HISTÓRICA REAL del
    sitio (horas_frio_media = 984.3 hs, calibrada con datos ERA5-Land de
    Jocolí 1990-2024), no el óptimo agronómico teórico de 1.000 hs. Si se
    centrara en el óptimo teórico, el clima típico del sitio (que está por
    debajo de ese óptimo) se penalizaría dos veces: una vez implícitamente,
    porque el techo de 3.000 kg/ha en plena producción (RENDIMIENTO_PLENA_KG_HA)
    ya está validado contra el clima real de Jocolí, y otra vez explícitamente,
    acá, si el factor no llegara a 1.00 en un año climáticamente típico.

    Umbrales (mismo ancho de zona que la versión centrada en el óptimo teórico,
    solo desplazados a la media real):
    - >= 984.3 hs : año típico o mejor, sin penalidad (factor = 1.00)
    - 784.3–984.3 hs: penalidad lineal moderada (0.70–1.00)
    - < 784.3 hs  : año con déficit severo, penalidad fuerte (0.40–0.70)
    """
    CENTRO = 984.3  # horas_frio_media histórica real (Módulo 0, Jocolí 1990-2024)
    ANCHO = 200.0   # mismo ancho de zona moderada que la función original
    UMBRAL_SEVERO = CENTRO - ANCHO  # 784.3
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
    sitio (calor_verano_media = 29.87 °C, calibrada con datos ERA5-Land de
    Jocolí 1990-2024), no el rango óptimo agronómico teórico (35-38 °C,
    Crane y Takeda, 1979; Ferguson, 2006). Mismo razonamiento que en
    `_media_factor_frio`: centrar en el óptimo teórico penalizaría dos veces
    el déficit de calor que ya está implícito en el techo de 3.000 kg/ha
    validado para este sitio.

    No se premia con factor > 1.00 un año más cálido que la media: eso
    requeriría que el factor pudiera superar 1.0, lo cual es incompatible
    con `_muestrear_beta_por_media` (la Beta está acotada en [0, 1] por
    definición). "Mejor que la media" se traduce en "sin penalidad adicional"
    (factor = 1.00), igual que en la función de frío.

    Umbrales (mismo ancho de zona que la versión centrada en el óptimo
    teórico, solo desplazados a la media real):
    - >= 29.87 °C  : año típico o mejor, sin penalidad (factor = 1.00)
    - 22.57–29.87 °C: penalidad lineal moderada (0.70–1.00)
    - < 22.57 °C   : año con déficit severo, penalidad fuerte (0.40–0.70)
    """
    CENTRO = 29.87  # calor_verano_media histórica real (Módulo 0, Jocolí 1990-2024)
    ANCHO = 7.3     # mismo ancho de zona moderada que la función original (35 - 27.7)
    UMBRAL_SEVERO = CENTRO - ANCHO  # 22.57
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


def _muestrear_beta_por_media(
    media: np.ndarray, precision: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Muestrea una Beta(alpha, beta) reparametrizada por media y precisión.

    precision (kappa) controla la variabilidad residual alrededor de `media`:
    kappa alto -> el factor se acerca al valor determinístico de `media`;
    kappa bajo -> más ruido biológico no explicado por el clima.
    """
    media = np.clip(media, 1e-4, 1 - 1e-4)
    alpha = media * precision
    beta = (1 - media) * precision
    return rng.beta(alpha, beta)


def _simular_veceria(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Cadena de Markov binaria (año alto / año bajo) para modelar la alternancia.

    Retorna array (n_simulaciones, n_años) con factores multiplicadores.
    """
    n, T = params.n_simulaciones, params.n_años
    es_bajo = np.zeros((n, T), dtype=bool)
    es_bajo[:, 0] = rng.random(n) < 0.5  # estado inicial aleatorio

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
    return np.where(es_bajo, factores_bajos, 1.0)


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def simulate_yields(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Simula el rendimiento en kg/ha para cada iteración y año del proyecto.

    Combina cinco fuentes de variabilidad:
    1. Curva de maduración determinista (años 1–20)
    2. Vecería (alternancia productiva) — cadena de Markov
    3. Déficit de horas de frío respecto a la media histórica real del sitio
       — media determinística + ruido Beta(media, precision_factor_frio)
    4. Déficit de calor estival (ene-feb) respecto a la media histórica real
       — media determinística + ruido Beta(media, precision_factor_calor)
    5. Tasa de falla de plantas — Beta(alpha, beta)

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

    horas = rng.normal(params.horas_frio_media, params.horas_frio_std, (n, T))
    media_ff = _media_factor_frio(horas)
    factor_frio = _muestrear_beta_por_media(media_ff, params.precision_factor_frio, rng)

    tmax_verano = rng.normal(params.calor_verano_media, params.calor_verano_std, (n, T))
    media_fc = _media_factor_calor(tmax_verano)
    factor_calor = _muestrear_beta_por_media(media_fc, params.precision_factor_calor, rng)

    # Falla de plantas: mismo valor para toda la vida del cultivo (decisión de campo)
    supervivencia = 1.0 - rng.beta(params.plantas_alpha, params.plantas_beta, (n, 1))

    return np.maximum(
        curva_base * factor_veceria * factor_frio * factor_calor * supervivencia, 0.0
    )


def simulate_prices(params: ParametrosMC, rng: np.random.Generator) -> np.ndarray:
    """
    Simula el precio de venta en USD/kg con distribución triangular.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    low, mode, high = ESCENARIOS_PRECIO[params.escenario]
    return rng.triangular(low, mode, high, (params.n_simulaciones, params.n_años))


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


def run_monte_carlo(
    params: ParametrosMC | None = None,
    costos: ParametrosCostos | None = None,
) -> pd.DataFrame:
    """
    Orquesta la simulación completa y retorna los resultados en formato tabular.

    El VAN se calcula sobre el flujo de caja neto (ingresos - OPEX), no sobre
    ingresos brutos. El CAPEX inicial se descuenta en el año 0 (factor 1.0).

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

    Retorna
    -------
    pd.DataFrame con columnas:
        simulacion, año, rendimiento_kg_ha, precio_usd_kg,
        ingreso_usd, opex_usd, flujo_neto_usd, vp_neto_usd, van_neto_usd
    """
    if params is None:
        params = ParametrosMC()
    if costos is None:
        costos = ParametrosCostos(hectareas=params.hectareas)
    elif costos.hectareas != params.hectareas:
        costos.hectareas = params.hectareas

    rng = np.random.default_rng(params.semilla)

    yields  = simulate_yields(params, rng)
    prices  = simulate_prices(params, rng)
    revenue = simulate_revenue(yields, prices, params.hectareas)

    flujo_neto = flujo_caja_neto(revenue, costos)

    años = np.arange(1, params.n_años + 1)
    factores_descuento = (1 / (1 + params.tasa_descuento) ** años).reshape(1, -1)
    vp_neto       = flujo_neto * factores_descuento
    van_neto_acum = -costos.capex_inicial + np.cumsum(vp_neto, axis=1)

    n, T = yields.shape
    opex_por_año = np.array(
        [costo_operativo_anual(a, costos) for a in range(1, T + 1)]
    )

    return pd.DataFrame({
        "simulacion":       np.repeat(np.arange(n), T),
        "año":              np.tile(años, n),
        "rendimiento_kg_ha": yields.ravel(),
        "precio_usd_kg":    prices.ravel(),
        "ingreso_usd":      revenue.ravel(),
        "opex_usd":         np.tile(opex_por_año, n),
        "flujo_neto_usd":   flujo_neto.ravel(),
        "vp_neto_usd":      vp_neto.ravel(),
        "van_neto_usd":     van_neto_acum.ravel(),
    })


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
    cambia de signo, la TIR no está definida).
    """
    n, T = flujos_con_capex.shape
    t = np.arange(T).reshape(1, T)
    r = np.full(n, tasa_inicial)
    convergio = np.zeros(n, dtype=bool)

    for _ in range(max_iter):
        denom = (1.0 + r.reshape(-1, 1)) ** t
        npv = np.sum(flujos_con_capex / denom, axis=1)
        dnpv = np.sum(
            -t * flujos_con_capex / (1.0 + r.reshape(-1, 1)) ** (t + 1), axis=1
        )
        dnpv = np.where(np.abs(dnpv) < 1e-12, np.nan, dnpv)
        paso = np.nan_to_num(npv / dnpv, nan=0.0)
        r_nuevo = np.clip(r - paso, -0.99, 10.0)
        convergio |= np.abs(r_nuevo - r) < tol
        r = r_nuevo

    return np.where(convergio, r, np.nan)


def resumen_financiero(df: pd.DataFrame, costos: ParametrosCostos) -> pd.DataFrame:
    """
    Resumen por simulación: VAN neto final, TIR y año de recupero.

    Parámetros
    ----------
    df : pd.DataFrame
        Salida de `run_monte_carlo()` (requiere columnas simulacion, año,
        flujo_neto_usd, van_neto_usd).
    costos : ParametrosCostos
        Debe ser el mismo objeto (mismo `hectareas`, mismo CAPEX inicial)
        usado para generar `df` en `run_monte_carlo()`. Es obligatorio y sin
        default a propósito: un default silencioso acá (p. ej. hectareas=25)
        daría un CAPEX incorrecto si `df` se generó con otra superficie.

    Retorna
    -------
    pd.DataFrame con columnas: simulacion, van_neto_usd, tir, año_recupero.
    `año_recupero` es NaN si el proyecto no recupera la inversión dentro
    del horizonte simulado.
    """
    tabla_flujos = df.pivot(index="simulacion", columns="año", values="flujo_neto_usd")
    n = tabla_flujos.shape[0]
    flujos_con_capex = np.hstack([
        np.full((n, 1), -costos.capex_inicial),
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
