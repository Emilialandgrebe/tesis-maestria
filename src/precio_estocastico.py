"""Simulación de precio del pistacho con memoria: AR(1) sobre retornos log reales.

El precio simulado es la trayectoria base del Excel v2 (USD constantes 2026,
por escenario -- `data/modelo_v2/anual.csv`) multiplicada por un DESVÍO
logarítmico estocástico:

    precio_t = trayectoria_t * exp(d_t),    d_t = r_1 + ... + r_t

donde los r_t siguen un AR(1) sobre retornos log calibrado con datos reales
del precio (FRED, serie WPU01190106, DEFLACTADA por CPI-U -- ver
`src/data/price_features.py::deflactar_por_cpi`). El AR(1) es sobre los
RETORNOS, no sobre el nivel: el ADF sobre el nivel real tampoco rechaza raíz
unitaria (p=0.399), así que no hay sustento para reversión a un nivel de
largo plazo. Es un ARIMA(1,1,0) sobre log(precio real).

La tendencia la pone la trayectoria del Excel, no el AR(1): por eso el drift
`c` vale 0 por defecto y el desvío tiene media cero en logaritmos (la
MEDIANA del precio simulado es la trayectoria base; la media queda por
encima en un factor exp(Var[d_t]/2), que crece con t). El desvío se acumula
desde el año 1 del proyecto, no desde la primera cosecha: la trayectoria es
un pronóstico hecho hoy, y la incertidumbre sobre el precio del año 7
incluye los 7 años que faltan para llegar.

Ver `notebooks/04_calibracion_precio.ipynb` para el diagnóstico de la serie
nominal y `notas/plan_precio_historico.md` para la justificación del AR(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.modelo_v2 import cargar_modelo_v2


@dataclass
class ParametrosPrecioAR1:
    """Parámetros del AR(1) sobre retornos log reales y escenario de trayectoria."""

    escenario: Literal["pesimista", "base", "optimista"] = "base"
    # Drift ADICIONAL sobre la trayectoria del Excel, en retorno log real
    # anual. 0 por defecto: la estimación sobre la serie real 1991-2026 da
    # c=-0.0015 (indistinguible de cero), y la tendencia ya está en la
    # trayectoria base. `src/dataset_ml.py` lo barre como input.
    c: float = 0.0
    # Reestimados sobre la serie deflactada por CPI-U (2026-11-04, 1991-2026,
    # n=34): phi=-0.217495 (p=0.218), sigma_eps=0.140922. Los valores
    # anteriores, sobre la serie NOMINAL, eran phi=-0.265423 y
    # sigma_eps=0.137358 (con c=0.031127 en la serie completa).
    phi: float = -0.217495
    sigma_eps: float = 0.140922


def _simular_desvio_log(z: np.ndarray, params: ParametrosPrecioAR1) -> np.ndarray:
    """
    Desvío logarítmico acumulado d_t a partir de choques normales `z`
    (forma (n_simulaciones, n_años)):

        r_1 = c/(1-phi) + sigma_eps * Z_1          (arranca en el retorno de
                                                      largo plazo del AR(1))
        r_t = c + phi * r_(t-1) + sigma_eps * Z_t   (t = 2..n_años)
        d_t = cumsum(r_1..r_t)
    """
    n_simulaciones, n_años = z.shape
    retornos = np.empty((n_simulaciones, n_años))
    retornos[:, 0] = params.c / (1 - params.phi) + params.sigma_eps * z[:, 0]
    for t in range(1, n_años):
        retornos[:, t] = (
            params.c + params.phi * retornos[:, t - 1] + params.sigma_eps * z[:, t]
        )
    return np.cumsum(retornos, axis=1)


def _simular_precios_desde_normales(
    z: np.ndarray, params: ParametrosPrecioAR1
) -> np.ndarray:
    """Trayectoria base del escenario (1, n_años) * exp(desvío) (n, n_años)."""
    n_años = z.shape[1]
    trayectoria = cargar_modelo_v2().trayectoria_precio(params.escenario, n_años)
    return trayectoria.reshape(1, -1) * np.exp(_simular_desvio_log(z, params))


def _generar_normales_antiteticos(
    n: int, n_años: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Genera `n` filas de `n_años` normales estándar usando variables
    antitéticas: la primera mitad son N(0,1) frescas, la segunda mitad es su
    negativo fila a fila (fila i <-> fila i + n//2) — mismo criterio de
    `_generar_uniformes_antiteticos()` en `src/monte_carlo.py`, aplicado
    directo sobre la normal (Z y -Z tienen la misma distribución, no hace
    falta invertir por PPF).

    Si `n` es impar, la última fila queda como una normal fresca sin pareja.
    """
    mitad = n // 2
    z = np.empty((n, n_años))
    z[:mitad] = rng.standard_normal((mitad, n_años))
    z[mitad : 2 * mitad] = -z[:mitad]
    if n % 2 == 1:
        z[-1] = rng.standard_normal(n_años)
    return z


def simulate_prices_ar1(
    n_simulaciones: int,
    n_años: int,
    params: ParametrosPrecioAR1,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simula el precio de venta en USD/kg (USD constantes 2026): trayectoria
    base del escenario por un desvío log AR(1) sobre retornos reales.

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    z = rng.standard_normal((n_simulaciones, n_años))
    return _simular_precios_desde_normales(z, params)


def simulate_prices_ar1_antitetico(
    n_simulaciones: int,
    n_años: int,
    params: ParametrosPrecioAR1,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Igual que `simulate_prices_ar1()`, pero con reducción de varianza por
    variables antitéticas: la mitad de las simulaciones usa los choques Z, la
    otra mitad usa -Z (mismo criterio que `simulate_prices_antitetico()` en
    `src/monte_carlo.py`).

    Retorna
    -------
    np.ndarray de forma (n_simulaciones, n_años).
    """
    z = _generar_normales_antiteticos(n_simulaciones, n_años, rng)
    return _simular_precios_desde_normales(z, params)
