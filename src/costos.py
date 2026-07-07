"""Estructura de costos del proyecto de pistacho, calibrada con datos reales
del plan de negocio (ver data/external/ y data/external/README.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Calibrado desde data/external/capex.csv (25 ha, Fase I), expresado por
# hectárea. NO incluye riego, pozo de agua, represa/cisterna ni paneles
# solares: 11 de los 29 ítems del CAPEX original están sin cotizar (ver
# data/external/README.md) — este valor subestima el CAPEX real hasta que
# se consigan esas cotizaciones.
CAPEX_INICIAL_USD_HA = 45_160.52

# Calibrado desde data/external/opex_preproductivo.csv, por hectárea. Años 4
# y 5 usan el mismo valor (columna "AÑO 4-6 (c/u)" de la fuente).
OPEX_PREPRODUCTIVO_USD_HA: dict[int, float] = {
    1: 5_071.60,
    2: 2_290.68,
    3: 2_399.80,
    4: 2_508.80,
    5: 2_508.80,
}

# Calibrado desde data/external/opex_productivo.csv, por hectárea, estructura
# de plena producción. Se aplica constante desde el año 6 en adelante como
# primera aproximación (no escala con el % de la curva de maduración).
OPEX_PRODUCTIVO_USD_HA = 5_256.00


@dataclass
class ParametrosCostos:
    """Parámetros de costos del proyecto, calibrados por hectárea.

    Los montos escalan con `hectareas`. Sincronizar este campo con
    `ParametrosMC.hectareas` es responsabilidad de quien arma la simulación
    (ver `run_monte_carlo()` en src/monte_carlo.py, que lo hace automático).
    """

    hectareas: float = 25.0
    capex_inicial_ha: float = CAPEX_INICIAL_USD_HA
    opex_preproductivo_ha: dict[int, float] = field(
        default_factory=lambda: dict(OPEX_PREPRODUCTIVO_USD_HA)
    )
    opex_productivo_ha: float = OPEX_PRODUCTIVO_USD_HA

    @property
    def capex_inicial(self) -> float:
        return self.capex_inicial_ha * self.hectareas


def costo_operativo_anual(año: int, params: ParametrosCostos) -> float:
    """
    Costo operativo (OPEX) del año dado, en USD, escalado por `params.hectareas`.

    Años 1 a 5: costos pre-productivos (sin cosecha comercial todavía).
    Año 6 en adelante: estructura operativa de plena producción.
    """
    if año in params.opex_preproductivo_ha:
        return params.opex_preproductivo_ha[año] * params.hectareas
    return params.opex_productivo_ha * params.hectareas


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
