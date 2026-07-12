"""Simulación de precio del pistacho con memoria: AR(1) sobre retornos log.

Reemplaza el supuesto de `simulate_prices()` (triangular independiente por año,
sin memoria) por un proceso calibrado con datos reales del precio (FRED, serie
WPU01190106). El AR(1) es sobre los RETORNOS logarítmicos, no sobre el nivel de
precio: el ADF sobre el nivel no rechaza raíz unitaria (p=0.726), así que no hay
sustento para modelar reversión a un nivel de precio de largo plazo (un
Ornstein-Uhlenbeck sobre log(precio) no estaría respaldado por los datos). En
la nomenclatura estándar de series de tiempo, esto es un ARIMA(1,1,0) con
drift: el nivel de precio es I(1) (no estacionario), la diferenciación de
orden 1 (retornos logarítmicos) lo estacionariza, y el AR(1) se ajusta sobre
esa serie ya diferenciada. Ver `notebooks/04_calibracion_precio.ipynb` para
el diagnóstico completo y `notas/plan_precio_historico.md` para la
justificación epistemológica.

Integrado a `src/monte_carlo.py` vía `run_monte_carlo_precio_historico()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Precio de anclaje del año 1 (USD/kg) por escenario — misma moda que
# ESCENARIOS_PRECIO en src/monte_carlo.py.
PRECIO_ANCLA_POR_ESCENARIO: dict[str, float] = {
    "pesimista": 7.0,
    "base": 9.5,
    "optimista": 13.0,
}


@dataclass
class ParametrosPrecioAR1:
    """Parámetros del AR(1) sobre retornos log y del precio de anclaje."""

    escenario: Literal["pesimista", "base", "optimista"] = "base"
    precio_ancla: float | None = None  # None -> tabla PRECIO_ANCLA_POR_ESCENARIO
    # c=0.008 (no 0.031127, la estimación puntual sobre la serie completa
    # 1991-2026): elección conservadora del drift, calibrada sobre la ventana
    # reciente 2010-2026 en vez de la serie completa. El IC95% del drift con
    # la serie completa es aprox. [-2%, +7%] — el parámetro no está
    # identificado con precisión (N chico de datos anuales reales), así que
    # usar la estimación puntual de la serie completa reclamaría más certeza
    # de la que hay. phi y sigma_eps sí quedan calibrados con la serie
    # completa: son estimaciones distintas, con distinta fuente de evidencia.
    c: float = 0.008
    phi: float = -0.265423
    sigma_eps: float = 0.137358

    def resolver_precio_ancla(self) -> float:
        """Devuelve `precio_ancla` si fue fijado, o la tabla por escenario."""
        if self.precio_ancla is not None:
            return self.precio_ancla
        return PRECIO_ANCLA_POR_ESCENARIO[self.escenario]


def _simular_precios_desde_normales(
    z: np.ndarray, params: ParametrosPrecioAR1
) -> np.ndarray:
    """
    Construye la trayectoria de precio a partir de choques normales `z`
    (forma (n_simulaciones, n_años)):

        r_1 = c/(1-phi) + sigma_eps * Z_1          (arranca en el retorno de
                                                      largo plazo del AR(1))
        r_t = c + phi * r_(t-1) + sigma_eps * Z_t   (t = 2..n_años)
        log(precio_t) = log(precio_ancla) + cumsum(r_1..r_t)
    """
    n_simulaciones, n_años = z.shape
    precio_ancla = params.resolver_precio_ancla()

    retornos = np.empty((n_simulaciones, n_años))
    retornos[:, 0] = params.c / (1 - params.phi) + params.sigma_eps * z[:, 0]
    for t in range(1, n_años):
        retornos[:, t] = (
            params.c + params.phi * retornos[:, t - 1] + params.sigma_eps * z[:, t]
        )

    log_precio = np.log(precio_ancla) + np.cumsum(retornos, axis=1)
    return np.exp(log_precio)


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
    Simula el precio de venta en USD/kg con un AR(1) sobre retornos log.

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
