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

# --- CAPEX estocástico: ítems sin cotizar con rango de mercado (2026-08-26) ---
#
# Hasta acá, CAPEX_INICIAL_USD_HA es 100% determinístico dentro de una
# corrida de Monte Carlo: la única variabilidad de CAPEX en todo el repo era
# el barrido `capex_extra_pct` de dataset_ml.py, y ese es un parámetro FIJO
# por punto del diseño LHS, no ruido dentro de las 10.000 iteraciones que
# arman la distribución del VAN.
#
# data/external/capex.csv tiene 14 ítems sin cotizar (columna costeado=NO).
# De esos, data/external/capex_estimaciones_web.csv solo le puso un RANGO de
# mercado investigado (rango_min_usd/rango_max_usd) a dos: riego y pozo de
# agua. Esos dos son los únicos que se modelan acá como triangulares. El
# resto de los ítems sin cotizar (desmonte/nivelación/subsolado -- solo tiene
# una estimación puntual sin rango, confianza "muy baja" --, represa/cisterna
# y paneles solares -- confianza "sin_dato"/sin estimar) NO entran en esta
# distribución: son riesgo real no cuantificado, y armarles un rango sin
# fuente sería disfrazar de variabilidad modelada algo que en realidad es
# incertidumbre epistémica sin dato. Quedan afuera del CAPEX simulado -- ver
# la advertencia en `notas/PLAN_TESIS.md` sobre el CAPEX como piso, no como
# costo total esperado.
#
# Los montos de capex.csv/capex_estimaciones_web.csv están cotizados sobre
# la superficie de Fase I de esa fuente (25 ha) -- se dividen por esa base
# para expresarlos en USD/ha, misma convención que CAPEX_INICIAL_USD_HA.
_BASE_HA_ESTIMACIONES_WEB = 25.0

# Riego por goteo enterrado (tuberías + goteros + filtros + fertirriego +
# bombeo). Rango de mercado 2025 para riego tecnificado en cultivos leñosos
# en Mendoza, confianza "media" (Los Andes, Argentina.gob.ar, Novagric --
# ver data/external/capex_estimaciones_web.csv).
CAPEX_RIEGO_LOW_HA = 70_000 / _BASE_HA_ESTIMACIONES_WEB
CAPEX_RIEGO_MODE_HA = 81_250 / _BASE_HA_ESTIMACIONES_WEB
CAPEX_RIEGO_HIGH_HA = 87_500 / _BASE_HA_ESTIMACIONES_WEB

# Pozo de agua (perforación + entubado + bomba sumergible + estudios).
# Confianza "baja": solo la mano de obra de perforación tiene fuente directa,
# el resto (entubado, bomba, estudio geoeléctrico, permisos) son estimados
# adicionales sin cotización real.
CAPEX_POZO_LOW_HA = 20_000 / _BASE_HA_ESTIMACIONES_WEB
CAPEX_POZO_MODE_HA = 33_000 / _BASE_HA_ESTIMACIONES_WEB
CAPEX_POZO_HIGH_HA = 40_000 / _BASE_HA_ESTIMACIONES_WEB

# Variación estructural del OPEX (pre-productivo y productivo) respecto al
# valor calibrado. A diferencia del CAPEX de arriba, acá no hay ningún rango
# de mercado investigado todavía en el repo -- este valor es un punto de
# partida razonable, NO una afirmación empírica, pendiente de calibrar con
# datos reales de costos operativos de otras fincas. Mismo criterio que
# `precision_factor_frio`/`precision_factor_calor` en `ParametrosMC`
# (src/monte_carlo.py): un supuesto declarado como tal, no escondido. ±15%
# es del mismo orden de magnitud que la dispersión relativa del rango de
# riego de arriba (±8% en torno al modo) y bastante más conservador que el
# barrido de `capex_extra_pct` en dataset_ml.py (0% a +30%, un sesgo puramente
# al alza pensado para el peor caso, no para variabilidad simétrica).
OPEX_VARIACION_PCT = 0.15


@dataclass
class ParametrosCostos:
    """Parámetros de costos del proyecto, calibrados por hectárea.

    Los montos escalan LINEALMENTE con `hectareas` (mismo supuesto que ya
    regía en `capex_inicial` y `costo_operativo_anual`: sin economías ni
    deseconomías de escala). Sincronizar este campo con
    `ParametrosMC.hectareas` es responsabilidad de quien arma la simulación
    (ver `run_monte_carlo()` en src/monte_carlo.py, que lo hace automático).

    Los campos `capex_riego_*_ha`, `capex_pozo_*_ha` y `opex_variacion_pct`
    parametrizan las fuentes de variabilidad estocástica de CAPEX/OPEX
    (`simulate_capex_extra()`/`simulate_opex_multiplicador()` en
    src/monte_carlo.py) -- no afectan a `capex_inicial` ni a
    `costo_operativo_anual()`, que siguen siendo el componente FIJO
    determinístico.
    """

    hectareas: float = 50.0
    capex_inicial_ha: float = CAPEX_INICIAL_USD_HA
    opex_preproductivo_ha: dict[int, float] = field(
        default_factory=lambda: dict(OPEX_PREPRODUCTIVO_USD_HA)
    )
    opex_productivo_ha: float = OPEX_PRODUCTIVO_USD_HA

    # CAPEX estocástico -- ver la sección de constantes más arriba
    capex_riego_low_ha: float = CAPEX_RIEGO_LOW_HA
    capex_riego_mode_ha: float = CAPEX_RIEGO_MODE_HA
    capex_riego_high_ha: float = CAPEX_RIEGO_HIGH_HA
    capex_pozo_low_ha: float = CAPEX_POZO_LOW_HA
    capex_pozo_mode_ha: float = CAPEX_POZO_MODE_HA
    capex_pozo_high_ha: float = CAPEX_POZO_HIGH_HA

    # OPEX estocástico -- ver la sección de constantes más arriba
    opex_variacion_pct: float = OPEX_VARIACION_PCT

    def __post_init__(self) -> None:
        """
        Valida los parámetros estocásticos de CAPEX/OPEX al construir el
        objeto: una triangular mal formada no falla en `rng.triangular`
        (que solo exige low<=mode<=high) sino más tarde y en silencio, como
        un signo de OPEX invertido o un CAPEX negativo (ver hallazgo del
        code review en notas/PLAN_TESIS.md, 2026-08-26).
        """
        if not (0.0 <= self.opex_variacion_pct < 1.0):
            raise ValueError(
                "opex_variacion_pct debe estar en [0, 1); recibido "
                f"{self.opex_variacion_pct!r}. Fuera de ese rango, la "
                "triangular(1-d, 1, 1+d) de simulate_opex_multiplicador() "
                "puede sortear un multiplicador <= 0 y flujo_caja_neto "
                "terminaría SUMANDO el OPEX en vez de restarlo."
            )
        for nombre, low, mode, high in (
            ("riego", self.capex_riego_low_ha, self.capex_riego_mode_ha, self.capex_riego_high_ha),
            ("pozo", self.capex_pozo_low_ha, self.capex_pozo_mode_ha, self.capex_pozo_high_ha),
        ):
            if not (low <= mode <= high):
                raise ValueError(
                    f"Triangular de CAPEX '{nombre}' mal formada: se requiere "
                    f"low <= mode <= high, recibido low={low!r}, "
                    f"mode={mode!r}, high={high!r}."
                )

    @property
    def capex_inicial(self) -> float:
        """CAPEX inicial FIJO (ítems con cotización real, costeado=SI).

        No incluye el CAPEX estocástico de riego/pozo -- ver
        `simulate_capex_extra()` en src/monte_carlo.py, que genera ese
        componente por separado y se suma en `_orquestar_resultado()`.
        """
        return self.capex_inicial_ha * self.hectareas  # escala lineal con hectareas


def costo_operativo_anual(año: int, params: ParametrosCostos) -> float:
    """
    Costo operativo (OPEX) BASE del año dado, en USD, escalado por
    `params.hectareas` (escala lineal, mismo supuesto que `capex_inicial`).

    Años 1 a 5: costos pre-productivos (sin cosecha comercial todavía).
    Año 6 en adelante: estructura operativa de plena producción.

    Este valor es determinístico -- el multiplicador estocástico de OPEX
    (`simulate_opex_multiplicador()` en src/monte_carlo.py) se aplica
    después, en `flujo_caja_neto()`.
    """
    if año in params.opex_preproductivo_ha:
        return params.opex_preproductivo_ha[año] * params.hectareas
    return params.opex_productivo_ha * params.hectareas


def flujo_caja_neto(
    ingresos_usd: np.ndarray,
    params: ParametrosCostos,
    opex_multiplicador: np.ndarray,
) -> np.ndarray:
    """
    Flujo de caja neto anual (ingresos - OPEX), sin incluir el CAPEX inicial.

    Parámetros
    ----------
    ingresos_usd : np.ndarray
        Forma (n_simulaciones, n_años) con los ingresos brutos de cada año.
    opex_multiplicador : np.ndarray
        Forma (n_simulaciones,) o (n_simulaciones, 1). Multiplicador
        estocástico de estructura de costos operativos: UN solo valor por
        simulación, aplicado a todos los años de esa simulación (no
        año a año -- la incertidumbre que representa es "este proyecto en
        particular resultó más/menos caro de lo presupuestado", no ruido
        anual). Ver `simulate_opex_multiplicador()` en src/monte_carlo.py.

    Retorna
    -------
    np.ndarray de la misma forma que `ingresos_usd`.
    """
    n, T = ingresos_usd.shape
    opex_base = np.array(
        [costo_operativo_anual(año, params) for año in range(1, T + 1)]
    ).reshape(1, T)
    opex = opex_base * np.asarray(opex_multiplicador).reshape(n, 1)
    return ingresos_usd - opex
