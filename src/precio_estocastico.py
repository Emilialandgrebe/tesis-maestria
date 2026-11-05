"""Simulación de precio del pistacho con memoria: AR(1) sobre retornos log reales.

El precio simulado es la trayectoria base del Excel v2 (USD constantes 2026,
por escenario -- `data/modelo_v2/anual.csv`) multiplicada por un DESVÍO
logarítmico estocástico:

    precio_t = trayectoria_t * exp(d_t - Var[d_t] / 2),    d_t = r_1 + ... + r_t

donde los r_t siguen un AR(1) sobre retornos log calibrado con datos reales
del precio (FRED, serie WPU01190106, DEFLACTADA por CPI-U -- ver
`src/data/price_features.py::deflactar_por_cpi`). El AR(1) es sobre los
RETORNOS, no sobre el nivel: el ADF sobre el nivel real tampoco rechaza raíz
unitaria (p=0.399), así que no hay sustento para reversión a un nivel de
largo plazo. Es un ARIMA(1,1,0) sobre log(precio real).

La tendencia la pone la trayectoria del Excel, no el AR(1): por eso el drift
`c` vale 0 por defecto. Criterio de VALOR ESPERADO: la trayectoria del Excel
es la MEDIA del precio simulado, no la mediana. Como d_t es normal,
E[exp(d_t)] = exp(E[d_t] + Var[d_t]/2); restar Var[d_t]/2 (la varianza
acumulada exacta bajo la dinámica del AR(1), ver `varianza_desvio_log()`)
deja E[exp(d_t - Var[d_t]/2)] = 1 con c=0 en cada año. La mediana queda
por debajo de la trayectoria en ese mismo factor. Con c != 0 el precio
medio es trayectoria * exp(drift acumulado de c): el drift sigue siendo
un desplazamiento deliberado de la media. El desvío se acumula
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
    # "retornos" (default): AR(1) sobre retornos log, ARIMA(1,1,0) -- el
    # desvío respecto de la trayectoria no revierte. "niveles": AR(1) sobre
    # el DESVÍO log respecto de la trayectoria, x_t = phi * x_(t-1) +
    # sigma_eps * Z_t con x_0 = 0, que revierte a la trayectoria del Excel.
    # Solo para análisis de sensibilidad de la especificación (ver
    # PHI_NIVELES y notas/PLAN_TESIS.md); con "niveles", `phi` tiene que ser
    # el de niveles y `c` tiene que valer 0.
    especificacion: Literal["retornos", "niveles"] = "retornos"
    # Trayectoria base propia (USD/kg por año 1..n), en lugar de la del Excel
    # del `escenario`. None (default) usa la del Excel. Solo para análisis
    # de sensibilidad (p. ej. precio real constante).
    trayectoria: tuple[float, ...] | None = None

    def trayectoria_base(self, n_años: int) -> np.ndarray:
        if self.trayectoria is None:
            return cargar_modelo_v2().trayectoria_precio(self.escenario, n_años)
        if len(self.trayectoria) != n_años:
            raise ValueError(
                f"trayectoria tiene {len(self.trayectoria)} años, se esperaban {n_años}"
            )
        return np.asarray(self.trayectoria, dtype=float)


# AR(1) por OLS sobre log(precio real) anual 1991-2026 con constante y
# tendencia lineal (la trayectoria del Excel tiene tendencia, así que la
# reversión es a una senda con pendiente): phi=0,8391 (EE 0,106), semivida
# ~4 años. Sin tendencia da 0,8761. El ADF sobre el log del nivel no rechaza
# raíz unitaria (p=0,55 con constante, 0,82 con tendencia): la serie no
# permite distinguir esta especificación de la de retornos.
PHI_NIVELES = 0.8391


def varianza_desvio_log(params: ParametrosPrecioAR1, n_años: int) -> np.ndarray:
    """
    Var[d_t] para t = 1..n_años, forma (n_años,), en forma cerrada.

    La parte aleatoria de los retornos es e_t = phi * e_(t-1) + sigma * Z_t
    con e_1 = sigma * Z_1, así que e_s = sigma * sum_(k<=s) phi^(s-k) Z_k y

        d_t = sum_(s<=t) e_s = sigma * sum_(k<=t) Z_k (1 - phi^(t-k+1)) / (1 - phi)
        Var[d_t] = sigma^2 / (1 - phi)^2 * sum_(j=1..t) (1 - phi^j)^2

    El drift c no aporta varianza. Verificado contra la varianza muestral
    de 400.000 trayectorias (coinciden al tercer decimal en los 20 años).

    Con `especificacion="niveles"`: Var[x_t] = sigma^2 (1 - phi^(2t)) / (1 - phi^2).
    """
    j = np.arange(1, n_años + 1)
    if params.especificacion == "niveles":
        return params.sigma_eps**2 * (1 - params.phi ** (2 * j)) / (1 - params.phi**2)
    return params.sigma_eps**2 / (1 - params.phi) ** 2 * np.cumsum((1 - params.phi**j) ** 2)


def _simular_desvio_log(z: np.ndarray, params: ParametrosPrecioAR1) -> np.ndarray:
    """
    Desvío logarítmico acumulado, con corrección de convexidad, a partir de
    choques normales `z` (forma (n_simulaciones, n_años)):

        r_1 = c/(1-phi) + sigma_eps * Z_1          (arranca en el retorno de
                                                      largo plazo del AR(1))
        r_t = c + phi * r_(t-1) + sigma_eps * Z_t   (t = 2..n_años)
        d_t = cumsum(r_1..r_t) - Var[d_t] / 2
    """
    n_simulaciones, n_años = z.shape
    if params.especificacion == "niveles":
        if params.c != 0:
            raise ValueError("especificacion='niveles' no admite drift c != 0")
        x = np.empty((n_simulaciones, n_años))
        x[:, 0] = params.sigma_eps * z[:, 0]
        for t in range(1, n_años):
            x[:, t] = params.phi * x[:, t - 1] + params.sigma_eps * z[:, t]
        return x - varianza_desvio_log(params, n_años).reshape(1, -1) / 2
    if params.especificacion != "retornos":
        raise ValueError(
            f"especificacion debe ser 'retornos' o 'niveles', recibido {params.especificacion!r}"
        )
    retornos = np.empty((n_simulaciones, n_años))
    retornos[:, 0] = params.c / (1 - params.phi) + params.sigma_eps * z[:, 0]
    for t in range(1, n_años):
        retornos[:, t] = (
            params.c + params.phi * retornos[:, t - 1] + params.sigma_eps * z[:, t]
        )
    return np.cumsum(retornos, axis=1) - varianza_desvio_log(params, n_años).reshape(1, -1) / 2


def _simular_precios_desde_normales(
    z: np.ndarray, params: ParametrosPrecioAR1
) -> np.ndarray:
    """Trayectoria base del escenario (1, n_años) * exp(desvío) (n, n_años)."""
    n_años = z.shape[1]
    trayectoria = params.trayectoria_base(n_años)
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
