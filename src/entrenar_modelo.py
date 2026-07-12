"""Entrena y evalúa los primeros modelos surrogate sobre el dataset sintético
(data/processed/dataset_ml_train.parquet / _test.parquet, generado en
src/dataset_ml.py).

Dos modelos, mismo criterio bagging vs. boosting que se puede citar en la
tesis (ver notas/PLAN_TESIS.md, sección de técnicas de la materia):
    - Random Forest (bagging, reduce varianza) — baseline interpretable.
    - LightGBM (boosting, reduce sesgo) — para exprimir performance.

Dos targets, evaluados por separado:
    - van_neto_medio_usd (regresión)
    - prob_van_negativo (regresión — es una probabilidad estimada por Monte
      Carlo en cada punto del espacio de parámetros, no una etiqueta binaria,
      así que se trata como regresión, no como clasificación)

Evaluación en el TEST SET (data/processed/dataset_ml_test.parquet, un diseño
LHS independiente del de train, semilla distinta) — mide generalización real
a puntos del espacio de parámetros nunca vistos, no solo ajuste al train.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

FEATURES = [
    "tasa_descuento",
    "hectareas",
    "capex_extra_pct",
    "precision_factor_frio",
    "precision_factor_calor",
    "p_bajo_si_alto",
    "p_alto_si_bajo",
    "precio_drift_anual",
    # "escenario" se agrega aparte, one-hot, ver preparar_features()
]

TARGETS = ["van_neto_medio_usd", "prob_van_negativo"]


def cargar_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga train y test desde data/processed/."""
    df_train = pd.read_parquet(PROCESSED_DIR / "dataset_ml_train.parquet")
    df_test = pd.read_parquet(PROCESSED_DIR / "dataset_ml_test.parquet")
    return df_train, df_test


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la matriz de features: los 8 parámetros numéricos + `escenario`
    one-hot (3 categorías -> 3 columnas binarias, sin drop_first porque los
    modelos de árbol no tienen problema de colinealidad como una regresión
    lineal).
    """
    X_num = df[FEATURES].copy()
    X_escenario = pd.get_dummies(df["escenario"], prefix="escenario")
    return pd.concat([X_num, X_escenario], axis=1)


@dataclass
class ResultadoModelo:
    nombre: str
    target: str
    r2_test: float
    rmse_test: float
    mae_test: float
    importancias: pd.Series


def entrenar_y_evaluar(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    semilla: int = 42,
) -> list[ResultadoModelo]:
    """Entrena Random Forest y LightGBM para `target`, evalúa en test."""
    X_train = preparar_features(df_train)
    X_test = preparar_features(df_test)
    y_train = df_train[target]
    y_test = df_test[target]

    modelos = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=None, random_state=semilla, n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, random_state=semilla, verbosity=-1,
            importance_type="gain",  # comparable con la importancia de RF
                                       # (reducción de error), no conteo de splits
        ),
    }

    resultados = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        importancias = pd.Series(
            modelo.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        resultados.append(
            ResultadoModelo(
                nombre=nombre,
                target=target,
                r2_test=r2_score(y_test, pred),
                rmse_test=root_mean_squared_error(y_test, pred),
                mae_test=mean_absolute_error(y_test, pred),
                importancias=importancias,
            )
        )
    return resultados


def comparar_con_sobol(importancias_rf: pd.Series) -> None:
    """
    Compara el ranking de importancia de variables del modelo de ML contra
    los índices de Sobol ya calculados (data/processed/sobol_indices.parquet,
    sobol_indices_riesgo.parquet) -- valida si dos metodologías distintas
    (feature importance de un ensemble vs. sensibilidad global de Sobol)
    coinciden en qué parámetros importan más.
    """
    try:
        sobol_van = pd.read_parquet(PROCESSED_DIR / "sobol_indices.parquet")
        print("\n--- Ranking Sobol (VAN medio, ST, promedio entre escenarios) ---")
        print(sobol_van.groupby("parametro")["ST"].mean().sort_values(ascending=False))
    except FileNotFoundError:
        print("(sobol_indices.parquet no encontrado, se salta la comparación)")


if __name__ == "__main__":
    df_train, df_test = cargar_datasets()
    print(f"Train: {df_train.shape}, Test: {df_test.shape}\n")

    for target in TARGETS:
        print(f"{'='*60}\nTarget: {target}\n{'='*60}")
        resultados = entrenar_y_evaluar(df_train, df_test, target)
        for r in resultados:
            print(f"\n{r.nombre}:")
            print(f"  R2 (test)   = {r.r2_test:.4f}")
            print(f"  RMSE (test) = {r.rmse_test:,.2f}")
            print(f"  MAE (test)  = {r.mae_test:,.2f}")
            print(f"  Top 5 features:")
            print(r.importancias.head(5).to_string())

        if target == "van_neto_medio_usd":
            comparar_con_sobol(resultados[0].importancias)
