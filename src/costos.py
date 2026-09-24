"""Estructura de costos del proyecto de pistacho, calibrada con datos reales
del plan de negocio (ver data/external/ y data/external/README.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Calibrado desde data/external/capex.csv (120 ha, resincronizado con el
# plan de negocio vigente 2026-09), expresado por hectárea:
# USD 1.249.082,14 (25 de 34 ítems con costeado=SI) / 120 ha = USD 10.409,02/ha.
#
# ESTE NÚMERO VA A SUBIR cuando se confirmen los dos ítems más importantes
# de los 9 todavía sin cotizar: Red antigranizo (mención informal de
# USD 16.400 sin confirmar con proveedor) y Paneles solares (la familia lo
# está cotizando activamente) -- ver data/external/capex.csv (columna
# costeado=NO) para los otros 7 (Subsolado profundo, Filtros y
# fertilizadores, Tuberías, Represa/cisterna, y los 3 ítems de TECNOLOGIA).
# Hasta que eso pase, NO tomar este valor como el CAPEX final del proyecto
# -- es un piso, no una cotización completa. Pozo de agua y bombas SÍ están
# confirmados con precio fijo real en esta versión (antes eran estimaciones
# web sin cotizar, ver notas/PLAN_TESIS.md -- Fase B -- para el detalle de
# por qué se retiró también el componente estocástico que las representaba).
# Se guarda como cociente exacto (no redondeado a 10_409.02) para que
# CAPEX_INICIAL_USD_HA * 120 reconstruya el total del CSV sin diferencia.
CAPEX_INICIAL_USD_HA = 1_249_082.14 / 120

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

# --- CAPEX estocástico: RETIRADO (Fase B, 2026-09-24) ---
#
# Hasta la resincronización con el plan de negocio 120 ha, acá vivían
# CAPEX_RIEGO_*_HA y CAPEX_POZO_*_HA: dos triangulares calibradas con
# estimaciones de mercado (data/external/capex_estimaciones_web.csv) para
# "riego" (tuberías+goteros+filtros+fertirriego+bombeo) y "pozo de agua"
# (perforación+entubado+bomba sumergible+estudios), los únicos dos ítems sin
# cotizar que tenían un rango investigado.
#
# Ahora Pozo de agua y Bombas y tablero eléctrico tienen precio FIJO real
# confirmado en data/external/capex.csv (ver CAPEX_INICIAL_USD_HA arriba) --
# dejar esas dos triangulares activas duplicaría ese costo: una vez fijo
# (adentro de capex_inicial) y otra vez como ruido adicional. Se retiraron
# las dos, no solo pozo: la triangular de "riego" incluía bombeo en su
# rango original (no estaba desagregado en la fuente), así que una parte de
# ese rango también quedaba duplicada con el precio fijo de Bombas, aunque
# el resto (tuberías/filtros/fertirriego) siga sin cotizar.
#
# `simulate_capex_extra()`/`simulate_capex_extra_antitetico()` en
# src/monte_carlo.py devuelven cero incondicionalmente ahora -- el
# componente estocástico de CAPEX queda VACÍO (ver notas/PLAN_TESIS.md,
# Fase B): no hay ningún ítem de CAPEX sin cotizar con rango de mercado hoy.
# Pendiente: conseguir una cotización de tuberías/filtros/fertirriego que
# excluya explícitamente el bombeo ya costeado aparte, para eventualmente
# reintroducir estocasticidad ahí si corresponde.

# Variación estructural del OPEX (pre-productivo y productivo) respecto al
# valor calibrado. No hay ningún rango de mercado investigado todavía en el
# repo para esto -- este valor es un punto de partida razonable, NO una
# afirmación empírica, pendiente de calibrar con datos reales de costos
# operativos de otras fincas. Mismo criterio que
# `precision_factor_frio`/`precision_factor_calor` en `ParametrosMC`
# (src/monte_carlo.py): un supuesto declarado como tal, no escondido. ±15%
# es bastante más conservador que el barrido de `capex_extra_pct` en
# dataset_ml.py (0% a +30%, un sesgo puramente al alza pensado para el peor
# caso, no para variabilidad simétrica). Es, desde Fase B, la ÚNICA fuente
# de variabilidad estocástica que le queda a `capex_opex_estocastico` --
# ver la nota de "CAPEX estocástico: RETIRADO" más arriba.
OPEX_VARIACION_PCT = 0.15


@dataclass
class ParametrosCostos:
    """Parámetros de costos del proyecto, calibrados por hectárea.

    Los montos escalan LINEALMENTE con `hectareas` (mismo supuesto que ya
    regía en `capex_inicial` y `costo_operativo_anual`: sin economías ni
    deseconomías de escala). Sincronizar este campo con
    `ParametrosMC.hectareas` es responsabilidad de quien arma la simulación
    (ver `run_monte_carlo()` en src/monte_carlo.py, que lo hace automático).

    El campo `opex_variacion_pct` parametriza la única fuente de
    variabilidad estocástica que queda activa desde Fase B
    (`simulate_opex_multiplicador()` en src/monte_carlo.py) -- no afecta a
    `capex_inicial` ni a `costo_operativo_anual()`, que siguen siendo el
    componente FIJO determinístico. El CAPEX estocástico (riego/pozo) se
    retiró -- ver la nota "CAPEX estocástico: RETIRADO" más arriba;
    `simulate_capex_extra()`/`simulate_capex_extra_antitetico()` en
    src/monte_carlo.py devuelven cero incondicionalmente.
    """

    hectareas: float = 50.0
    capex_inicial_ha: float = CAPEX_INICIAL_USD_HA
    opex_preproductivo_ha: dict[int, float] = field(
        default_factory=lambda: dict(OPEX_PREPRODUCTIVO_USD_HA)
    )
    opex_productivo_ha: float = OPEX_PRODUCTIVO_USD_HA

    # OPEX estocástico -- ver la sección de constantes más arriba
    opex_variacion_pct: float = OPEX_VARIACION_PCT

    def __post_init__(self) -> None:
        """
        Valida `opex_variacion_pct` al construir el objeto: fuera de rango,
        no falla acá sino más tarde y en silencio, como un signo de OPEX
        invertido (ver hallazgo del code review en notas/PLAN_TESIS.md,
        2026-08-26). La validación análoga para las triangulares de CAPEX
        (riego/pozo) se retiró junto con esos campos -- ver Fase B.
        """
        if not (0.0 <= self.opex_variacion_pct < 1.0):
            raise ValueError(
                "opex_variacion_pct debe estar en [0, 1); recibido "
                f"{self.opex_variacion_pct!r}. Fuera de ese rango, la "
                "triangular(1-d, 1, 1+d) de simulate_opex_multiplicador() "
                "puede sortear un multiplicador <= 0 y flujo_caja_neto "
                "terminaría SUMANDO el OPEX en vez de restarlo."
            )

    @property
    def capex_inicial(self) -> float:
        """CAPEX inicial FIJO (ítems con cotización real, costeado=SI).

        Desde Fase B no hay componente estocástico de CAPEX que sumarle --
        `simulate_capex_extra()` en src/monte_carlo.py devuelve cero
        incondicionalmente (ver notas/PLAN_TESIS.md). Se mantiene la suma
        en `_orquestar_resultado()` por compatibilidad de la fórmula, sin
        efecto numérico.
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
