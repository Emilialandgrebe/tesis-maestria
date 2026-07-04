"""Estructura de costos del proyecto de pistacho, calibrada con datos reales
del plan de negocio (ver data/external/ y data/external/NOTAS_CALIDAD_DATOS.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Calibrado desde data/external/capex.csv, 25 ha (Fase I). El sistema de riego
# todavía no está cotizado (11 de 29 ítems con costeado=NO) — este total
# subestima el CAPEX real hasta que se consiga esa cotización.
CAPEX_INICIAL_USD = 1_129_013.12

# Calibrado desde data/external/opex_preproductivo.csv, 25 ha. Años 4 a 6 usan
# el mismo valor (columna "AÑO 4-6 (c/u)" de la fuente).
OPEX_PREPRODUCTIVO_USD: dict[int, float] = {
    1: 126_790.0,
    2: 57_267.0,
    3: 59_995.0,
    4: 62_720.0,
    5: 62_720.0,
}

# Calibrado desde data/external/opex_productivo.csv, 25 ha, estructura de
# plena producción. Se aplica constante desde el año 6 en adelante como
# primera aproximación (no escala con el % de la curva de maduración).
OPEX_PRODUCTIVO_USD = 131_400.0


@dataclass
class ParametrosCostos:
    """Parámetros de costos del proyecto, calibrados a 25 ha (Fase I).

    Estos montos NO se re-escalan automáticamente si se cambia `hectareas`
    en ParametrosMC — vienen de un plan de negocio itemizado para 25 ha.
    """

    capex_inicial: float = CAPEX_INICIAL_USD
    opex_preproductivo: dict[int, float] = field(
        default_factory=lambda: dict(OPEX_PREPRODUCTIVO_USD)
    )
    opex_productivo: float = OPEX_PRODUCTIVO_USD


def costo_operativo_anual(año: int, params: ParametrosCostos) -> float:
    """
    Costo operativo (OPEX) del año dado, en USD.

    Años 1 a 5: costos pre-productivos (sin cosecha comercial todavía).
    Año 6 en adelante: estructura operativa de plena producción.
    """
    if año in params.opex_preproductivo:
        return params.opex_preproductivo[año]
    return params.opex_productivo


def flujo_caja_neto(ingresos_usd: np.ndarray, params: ParametrosCostos) -> np.ndarray:
    """
    Flujo de caja neto anual (ingresos - OPEX), sin incluir el CAPEX inicial.

    Parámetros
    ----------
    ingresos_usd : np.ndarray
        Forma (n_simulaciones, n_años) con los ingresos brutos de cada año.

    Retorna
    -------
    np.ndarray de la misma forma que `ingresos_usd`.
    """
    n, T = ingresos_usd.shape
    opex = np.array(
        [costo_operativo_anual(año, params) for año in range(1, T + 1)]
    ).reshape(1, T)
    return ingresos_usd - opex
