"""Simulación de Monte Carlo para rendimientos e ingresos del cultivo de pistacho."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes del plan de negocios
# ---------------------------------------------------------------------------

HECTAREAS = 25.0
RENDIMIENTO_PLENA_KG_HA = 4_500  # kg/ha en plena producción (año 12+)

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

def _factor_frio(horas: np.ndarray) -> np.ndarray:
    """
    Función de transferencia: horas de frío acumuladas → factor de rendimiento.

    Umbrales agronómicos para pistacho Kerman:
    - >= 1.000 hs : sin penalidad (factor = 1.00)
    - 800–1.000 hs: penalidad lineal moderada (0.70–1.00)
    - < 800 hs   : año crítico, penalidad severa (0.40–0.70)
    """
    return np.clip(
        np.where(
            horas >= 1_000,
            1.0,
            np.where(
                horas >= 800,
                0.70 + 0.30 * (horas - 800) / 200,
                0.40 + 0.30 * horas / 800,
            ),
        ),
        0.0,
        1.0,
    )


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

    Combina cuatro fuentes de variabilidad:
    1. Curva de maduración determinista (años 1–20)
    2. Vecería (alternancia productiva) — cadena de Markov
    3. Penalidad por horas de frío insuficientes — función de transferencia
    4. Tasa de falla de plantas — Beta(alpha, beta)

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
    factor_clima = _factor_frio(horas)

    # Falla de plantas: mismo valor para toda la vida del cultivo (decisión de campo)
    supervivencia = 1.0 - rng.beta(params.plantas_alpha, params.plantas_beta, (n, 1))

    return np.maximum(curva_base * factor_veceria * factor_clima * supervivencia, 0.0)


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


def run_monte_carlo(params: ParametrosMC | None = None) -> pd.DataFrame:
    """
    Orquesta la simulación completa y retorna los resultados en formato tabular.

    Parámetros
    ----------
    params : ParametrosMC, opcional
        Configuración del modelo. Si es None usa los valores por defecto.

    Retorna
    -------
    pd.DataFrame con columnas:
        simulacion, año, rendimiento_kg_ha, precio_usd_kg,
        ingreso_usd, vp_usd, van_acumulado_usd
    """
    if params is None:
        params = ParametrosMC()

    rng = np.random.default_rng(params.semilla)

    yields  = simulate_yields(params, rng)
    prices  = simulate_prices(params, rng)
    revenue = simulate_revenue(yields, prices, params.hectareas)

    años = np.arange(1, params.n_años + 1)
    factores_descuento = (1 / (1 + params.tasa_descuento) ** años).reshape(1, -1)
    vp          = revenue * factores_descuento
    van_acum    = np.cumsum(vp, axis=1)

    n, T = yields.shape
    return pd.DataFrame({
        "simulacion":       np.repeat(np.arange(n), T),
        "año":              np.tile(años, n),
        "rendimiento_kg_ha": yields.ravel(),
        "precio_usd_kg":    prices.ravel(),
        "ingreso_usd":      revenue.ravel(),
        "vp_usd":           vp.ravel(),
        "van_acumulado_usd": van_acum.ravel(),
    })
