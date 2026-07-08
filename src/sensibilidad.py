"""Análisis de sensibilidad global (Sobol) sobre el simulador de Monte Carlo real.

No usa un dataset sintético ni Hipercubo Latino: cada punto de la muestra de
Sobol corre una simulación completa (`run_monte_carlo_antitetico` +
`resumen_financiero`) con esos parámetros.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.costos import ParametrosCostos
from src.monte_carlo import ParametrosMC, resumen_financiero, run_monte_carlo_antitetico

# Rango de hectareas [50, 100]: Fase II del plan de negocio real (la hoja
# "Financiamiento" de data/external/ menciona ampliación a 50 ha sobre los
# 100 ha de tierra comprada). No depende de ningún módulo de dataset
# sintético.
PROBLEMA_SOBOL: dict = {
    "num_vars": 6,
    "names": [
        "tasa_descuento",
        "precision_factor_frio",
        "precision_factor_calor",
        "p_bajo_si_alto",
        "capex_extra_pct",
        "hectareas",
    ],
    "bounds": [
        [0.05, 0.12],
        [5.0, 30.0],
        [5.0, 30.0],
        [0.50, 0.80],
        [0.0, 0.30],
        [50.0, 100.0],
    ],
}


def _evaluar_van_medio(
    fila: np.ndarray, escenario: str, n_simulaciones: int, semilla: int
) -> float:
    """
    Corre una simulación completa con los 6 parámetros de `fila` (mismo
    orden que PROBLEMA_SOBOL["names"]) y devuelve el VAN neto medio: el
    output escalar que analiza Sobol.

    `capex_extra_pct` no es un campo de ParametrosCostos (no se modifica
    src/costos.py): se aplica acá como un recargo sobre `capex_inicial_ha`,
    representando la incertidumbre de los ítems de CAPEX sin cotizar
    (riego, pozo, represa, paneles — ver data/external/README.md).

    La semilla se mantiene FIJA en todas las evaluaciones (no varía con
    `fila`): así la única fuente de variación en el output es el barrido de
    parámetros, no ruido de muestreo adicional. Sobol asume una función
    determinística de las variables analizadas; si la semilla variara, el
    ruido de Monte Carlo (más notorio con `n_simulaciones` chico) se
    mezclaría con la sensibilidad real de cada parámetro.
    """
    (
        tasa_descuento,
        precision_factor_frio,
        precision_factor_calor,
        p_bajo_si_alto,
        capex_extra_pct,
        hectareas,
    ) = fila

    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=hectareas,
        tasa_descuento=tasa_descuento,
        precision_factor_frio=precision_factor_frio,
        precision_factor_calor=precision_factor_calor,
        p_bajo_si_alto=p_bajo_si_alto,
        escenario=escenario,
        semilla=semilla,
    )
    costos = ParametrosCostos(hectareas=hectareas)
    costos.capex_inicial_ha *= 1.0 + capex_extra_pct

    df = run_monte_carlo_antitetico(params, costos)
    resumen = resumen_financiero(df, costos)
    return float(resumen["van_neto_usd"].mean())


def analizar_sensibilidad_sobol(
    escenario: str,
    n_base: int = 128,
    n_simulaciones: int = 1_000,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad global de Sobol sobre el simulador real
    (`run_monte_carlo_antitetico`), para un escenario de precio fijo.

    El parámetro `escenario` de ParametrosMC es categórico y Sobol requiere
    variables continuas, así que este análisis se corre por separado para
    cada uno de los 3 valores de escenario (ver `comparar_sensibilidad_escenarios`).

    Se usa `calc_second_order=False` porque solo hacen falta S1 y ST: el
    muestreo de Saltelli requiere entonces N * (D + 2) evaluaciones del
    modelo (D=6 variables), en vez de N * (2D + 2) si también se pidieran
    los índices de segundo orden.

    Parámetros
    ----------
    escenario : str
        "pesimista", "base" u "optimista".
    n_base : int
        Tamaño base de la muestra de Sobol (N). Total de evaluaciones del
        modelo = n_base * (6 + 2) = n_base * 8. Calibrado empíricamente:
        cada evaluación con n_simulaciones=1.000 tarda ~0.2s, así que
        n_base=128 (1.024 evaluaciones/escenario) da ~11-13 minutos en
        total para los 3 escenarios (medido en corridas reales); n_base=256
        ya sube a ~21 minutos.
    n_simulaciones : int
        n_simulaciones de cada corrida de Monte Carlo dentro de cada
        evaluación de Sobol (deliberadamente más chico que el N de
        producción, para que el total de evaluaciones sea manejable).
    semilla : int
        Semilla FIJA para todas las evaluaciones (ver `_evaluar_van_medio`) y
        para el muestreador de Sobol (`sobol_sample.sample(..., seed=semilla)`).
        Sin esto último, cada corrida usa una secuencia de Sobol distinta y
        los índices S1/ST no son reproducibles de una corrida a otra.

    Retorna
    -------
    pd.DataFrame con columnas: parametro, S1, S1_conf, ST, ST_conf.
    """
    muestra = sobol_sample.sample(
        PROBLEMA_SOBOL, n_base, calc_second_order=False, seed=semilla
    )
    total = muestra.shape[0]
    print(f"[{escenario}] {total} evaluaciones del modelo (N={n_base} x 8)...")

    outputs = np.empty(total)
    inicio = time.time()
    paso_reporte = max(1, total // 10)
    for i, fila in enumerate(muestra):
        outputs[i] = _evaluar_van_medio(fila, escenario, n_simulaciones, semilla)
        if (i + 1) % paso_reporte == 0 or (i + 1) == total:
            transcurrido = time.time() - inicio
            print(f"[{escenario}] {i + 1}/{total} ({transcurrido:.0f}s transcurridos)")

    indices = sobol_analyze.analyze(
        PROBLEMA_SOBOL,
        outputs,
        calc_second_order=False,
        print_to_console=False,
        seed=semilla,
    )

    return pd.DataFrame({
        "parametro": PROBLEMA_SOBOL["names"],
        "S1":        indices["S1"],
        "S1_conf":   indices["S1_conf"],
        "ST":        indices["ST"],
        "ST_conf":   indices["ST_conf"],
    })


def comparar_sensibilidad_escenarios(
    n_base: int = 128, n_simulaciones: int = 1_000, semilla: int = 42
) -> pd.DataFrame:
    """
    Corre `analizar_sensibilidad_sobol()` para los 3 escenarios de precio y
    devuelve un único DataFrame combinado (columna `escenario` adicional).
    Guarda el resultado en data/processed/sobol_indices.parquet.
    """
    resultados = []
    for escenario in ["pesimista", "base", "optimista"]:
        df_escenario = analizar_sensibilidad_sobol(escenario, n_base, n_simulaciones, semilla)
        df_escenario["escenario"] = escenario
        resultados.append(df_escenario)

    combinado = pd.concat(resultados, ignore_index=True)

    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "sobol_indices.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(ruta)
    print(f"Guardado en {ruta}")

    return combinado
