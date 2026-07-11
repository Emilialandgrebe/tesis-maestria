"""Generación del dataset sintético para ML (capa 3 de la arquitectura de datos).

No hay recolección de datos externos acá: el dataset se genera evaluando el
simulador real (`run_monte_carlo_antitetico` + `resumen_financiero`) en
puntos elegidos con Latin Hypercube Sampling sobre el espacio de parámetros,
uno por escenario de precio. Cada fila del dataset resultante es:

    [parámetros de entrada] -> [resumen estadístico del VAN de esa corrida]

Es decir, es un dataset para *emular* el simulador (surrogate modeling),
no para predecir un rendimiento agroclimático real observado.

Dos targets principales:
    - van_neto_medio_usd (+ p10/p90): regresión, aproxima el VAN esperado.
    - prob_van_negativo: fracción de simulaciones con VAN < 0 en esa corrida,
      target de riesgo (conecta con el "Paso 6" de PLAN_TESIS.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.costos import ParametrosCostos
from src.monte_carlo import ESCENARIOS_PRECIO, ParametrosMC, resumen_financiero, run_monte_carlo_precio_historico
from src.precio_estocastico import ParametrosPrecioAR1

# ---------------------------------------------------------------------------
# Espacio de parámetros
# ---------------------------------------------------------------------------

# Mismo criterio que PROBLEMA_SOBOL en src/sensibilidad.py, con dos cambios:
# - hectareas: piso bajado de 50 a 25 para cubrir también la escala actual
#   del proyecto (Fase I), no solo la ampliación a Fase II.
# - p_alto_si_bajo: agregado (Sobol lo dejaba fijo en 0.80); acá no hay
#   costo adicional en barrerlo también.
ESPACIO_PARAMETROS: dict[str, tuple[float, float]] = {
    "tasa_descuento":         (0.05, 0.12),
    "hectareas":              (25.0, 100.0),
    "capex_extra_pct":        (0.0, 0.30),
    "precision_factor_frio":  (5.0, 30.0),
    "precision_factor_calor": (5.0, 30.0),
    "p_bajo_si_alto":         (0.50, 0.80),
    "p_alto_si_bajo":         (0.60, 0.90),
    "precio_drift_anual":     (0.0, 0.025),  # incertidumbre del drift AR(1), ver plan
}

NOMBRES_PARAMETROS = list(ESPACIO_PARAMETROS.keys())
_BOUNDS_INF = [ESPACIO_PARAMETROS[n][0] for n in NOMBRES_PARAMETROS]
_BOUNDS_SUP = [ESPACIO_PARAMETROS[n][1] for n in NOMBRES_PARAMETROS]

ESCENARIOS = list(ESCENARIOS_PRECIO.keys())  # ["pesimista", "base", "optimista"]


# ---------------------------------------------------------------------------
# Muestreo LHS
# ---------------------------------------------------------------------------

def generar_muestra_lhs(n_por_escenario: int, semilla: int) -> pd.DataFrame:
    """
    Genera una muestra por Latin Hypercube Sampling del espacio de parámetros
    continuos, repetida para cada escenario de precio (el escenario no entra
    al LHS porque es categórico: se estratifica aparte, con el mismo diseño
    continuo replicado en los 3 niveles).

    Retorna
    -------
    pd.DataFrame con columnas: NOMBRES_PARAMETROS + ["escenario"],
    n_por_escenario * len(ESCENARIOS) filas.
    """
    sampler = qmc.LatinHypercube(d=len(NOMBRES_PARAMETROS), seed=semilla)
    muestra_unitaria = sampler.random(n=n_por_escenario)
    muestra_escalada = qmc.scale(muestra_unitaria, _BOUNDS_INF, _BOUNDS_SUP)

    df_base = pd.DataFrame(muestra_escalada, columns=NOMBRES_PARAMETROS)

    bloques = []
    for escenario in ESCENARIOS:
        bloque = df_base.copy()
        bloque["escenario"] = escenario
        bloques.append(bloque)

    return pd.concat(bloques, ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluación de un punto del espacio de parámetros
# ---------------------------------------------------------------------------

@dataclass
class ResultadoPunto:
    van_neto_medio_usd: float
    van_p10_usd: float
    van_p90_usd: float
    van_std_usd: float
    prob_van_negativo: float
    tir_media: float
    año_recupero_medio: float
    prob_recupero_antes_12: float
    prob_recupero_antes_15: float
    prob_recupero_antes_20: float


def evaluar_punto(
    fila: pd.Series,
    n_simulaciones: int,
    semilla: int,
) -> ResultadoPunto:
    """
    Corre una simulación completa (run_monte_carlo_precio_historico +
    resumen_financiero) con los parámetros de `fila` y resume el resultado en
    los targets del dataset.

    El precio usa el modelo AR(1) sobre retornos calibrado con FRED
    (src/precio_estocastico.py), no el triangular independiente viejo. El
    drift (`c`) se barre como parámetro de entrada (`precio_drift_anual`) en
    vez de quedar fijo: no está identificado con precisión (ver
    notas/plan_precio_historico.md, sección 0) y mueve sustancialmente tanto
    el VAN medio como prob_van_negativo. phi/sigma_eps quedan en su default
    calibrado (no se barren, tienen mejor soporte estadístico).

    La semilla se mantiene fija entre puntos (igual que en sensibilidad.py):
    así la variación entre filas del dataset viene solo del barrido de
    parámetros, no de ruido de muestreo adicional entre evaluaciones.
    """
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=fila["hectareas"],
        tasa_descuento=fila["tasa_descuento"],
        precision_factor_frio=fila["precision_factor_frio"],
        precision_factor_calor=fila["precision_factor_calor"],
        p_bajo_si_alto=fila["p_bajo_si_alto"],
        p_alto_si_bajo=fila["p_alto_si_bajo"],
        escenario=fila["escenario"],
        semilla=semilla,
    )
    costos = ParametrosCostos(hectareas=fila["hectareas"])
    costos.capex_inicial_ha *= 1.0 + fila["capex_extra_pct"]

    precio_params = ParametrosPrecioAR1(
        escenario=fila["escenario"],
        c=fila["precio_drift_anual"],
    )

    df = run_monte_carlo_precio_historico(params, costos, precio_params)
    resumen = resumen_financiero(df, costos)

    van = resumen["van_neto_usd"]
    tir = resumen["tir"].dropna()
    recupero = resumen["año_recupero"].dropna()

    return ResultadoPunto(
        van_neto_medio_usd=float(van.mean()),
        van_p10_usd=float(van.quantile(0.10)),
        van_p90_usd=float(van.quantile(0.90)),
        van_std_usd=float(van.std()),
        prob_van_negativo=float((van < 0).mean()),
        tir_media=float(tir.mean()) if len(tir) else float("nan"),
        año_recupero_medio=float(recupero.mean()) if len(recupero) else float("nan"),
        prob_recupero_antes_12=float((recupero <= 12).mean()) if len(recupero) else 0.0,
        prob_recupero_antes_15=float((recupero <= 15).mean()) if len(recupero) else 0.0,
        prob_recupero_antes_20=float((recupero <= 20).mean()) if len(recupero) else 0.0,
    )


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def generar_dataset(
    n_por_escenario: int = 800,
    n_simulaciones: int = 1_000,
    semilla: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Genera el dataset sintético completo: barre el espacio de parámetros con
    LHS (`generar_muestra_lhs`) y evalúa el simulador real en cada punto
    (`evaluar_punto`).

    Parámetros
    ----------
    n_por_escenario : int
        Puntos LHS por escenario de precio. Total de evaluaciones del
        modelo = n_por_escenario * 3.
    n_simulaciones : int
        n_simulaciones de cada corrida de Monte Carlo dentro de cada punto
        (mismo trade-off que en sensibilidad.py: no hace falta el N de
        producción para el resumen estadístico de cada punto).
    semilla : int
        Semilla para el sampler LHS y para todas las evaluaciones del
        simulador (reproducibilidad).

    Retorna
    -------
    pd.DataFrame con columnas: NOMBRES_PARAMETROS + ["escenario"] + targets.
    """
    muestra = generar_muestra_lhs(n_por_escenario, semilla)
    total = len(muestra)
    if verbose:
        print(f"Generando dataset: {total} evaluaciones del modelo "
              f"({n_por_escenario} x {len(ESCENARIOS)} escenarios)...")

    inicio = time.time()
    resultados = []
    paso_reporte = max(1, total // 10)
    for i, fila in muestra.iterrows():
        resultados.append(evaluar_punto(fila, n_simulaciones, semilla))
        if verbose and ((i + 1) % paso_reporte == 0 or (i + 1) == total):
            transcurrido = time.time() - inicio
            print(f"  {i + 1}/{total} ({transcurrido:.0f}s transcurridos)")

    df_resultados = pd.DataFrame([r.__dict__ for r in resultados])
    return pd.concat([muestra.reset_index(drop=True), df_resultados], axis=1)


def generar_dataset_train_test(
    n_train_por_escenario: int = 800,
    n_test_por_escenario: int = 200,
    n_simulaciones: int = 1_000,
    semilla_train: int = 42,
    semilla_test: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera train y test con dos diseños LHS independientes (semillas
    distintas, no un split de la misma muestra). Esto mide generalización
    a puntos del espacio de parámetros que el diseño de entrenamiento nunca
    vio, que es la pregunta relevante para un modelo surrogate.
    """
    print("=== Dataset de entrenamiento ===")
    df_train = generar_dataset(n_train_por_escenario, n_simulaciones, semilla_train)

    print("\n=== Dataset de test ===")
    df_test = generar_dataset(n_test_por_escenario, n_simulaciones, semilla_test)

    return df_train, df_test


if __name__ == "__main__":
    from pathlib import Path

    df_train, df_test = generar_dataset_train_test()

    out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(out_dir / "dataset_ml_train.parquet")
    df_test.to_parquet(out_dir / "dataset_ml_test.parquet")
    print(f"\nGuardado en {out_dir}/dataset_ml_train.parquet y dataset_ml_test.parquet")
