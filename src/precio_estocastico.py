"""Simulación de precio del pistacho con memoria: AR(1) en log, calibrado con FRED.

El precio simulado es la trayectoria base del Excel v2 (USD constantes 2026,
por escenario -- `data/modelo_v2/anual.csv`) multiplicada por un DESVÍO
logarítmico estocástico:

    precio_t = nivel_mult * trayectoria_t * exp(d_t - Var[d_t] / 2)

Dos especificaciones de d_t, calibradas sobre la serie FRED WPU01190106
DEFLACTADA por CPI-U (`src/data/price_features.py::deflactar_por_cpi`),
con el mismo sigma_eps:

- "niveles" (PRINCIPAL desde 2026-11-04): AR(1) del desvío log respecto de
  la trayectoria, d_t = phi * d_(t-1) + sigma_eps * Z_t + c (acumulado como
  c * t), d_0 = 0, phi = PHI_NIVELES. Revierte a la trayectoria (semivida
  ~4 años); la varianza converge a sigma^2 / (1 - phi^2).
- "retornos" (escenario de ESTRÉS): AR(1) sobre los retornos log,
  ARIMA(1,1,0), phi = PHI_RETORNOS. El desvío no revierte y la varianza
  crece con el horizonte.

El ADF sobre el log del precio real no rechaza raíz unitaria (p=0,55 con
constante, 0,82 con tendencia), así que los datos no distinguen entre las
dos. Se eligió "niveles" por plausibilidad: su dispersión estacionaria
(desvío log 0,26, P95/P5 ~2,4) coincide con el rango histórico de la serie
(0,23-0,28 y 2,44), mientras que "retornos" llega al año 20 con el doble
(0,52, P95/P5 5,7). Ver notas/PLAN_TESIS.md.

La tendencia la pone la trayectoria del Excel, no el AR(1): por eso el drift
`c` vale 0 por defecto. Criterio de VALOR ESPERADO: la trayectoria es la
MEDIA del precio simulado, no la mediana. Como d_t es normal,
E[exp(d_t)] = exp(E[d_t] + Var[d_t]/2); restar Var[d_t]/2 (la varianza
exacta bajo cada especificación, ver `varianza_desvio_log()`) deja
E[exp(d_t - Var[d_t]/2)] = 1 con c=0 en cada año. La mediana queda por
debajo de la trayectoria en ese mismo factor. Con c != 0 el precio medio es
trayectoria * exp(drift acumulado de c): el drift es un desplazamiento
deliberado de la media. El desvío arranca en el año 1 del proyecto, no en
la primera cosecha: la trayectoria es un pronóstico hecho hoy, y la
incertidumbre sobre el precio del año 7 incluye los 7 años que faltan.

Ver `notebooks/04_calibracion_precio.ipynb` para el diagnóstico de la serie
nominal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.modelo_v2 import cargar_modelo_v2

# AR(1) por OLS sobre log(precio real) anual 1991-2026 con constante y
# tendencia lineal (la trayectoria del Excel tiene tendencia, así que la
# reversión es a una senda con pendiente): phi=0,8391 (EE 0,106), semivida
# ~4 años. Sin tendencia da 0,8761.
PHI_NIVELES = 0.8391

# AR(1) sobre los retornos log reales (2026-11-04, 1991-2026, n=34):
# phi=-0.217495 (p=0.218). Sobre la serie NOMINAL era -0.265423.
PHI_RETORNOS = -0.217495


@dataclass
class ParametrosPrecioAR1:
    """Parámetros del AR(1) del precio y de su trayectoria base."""

    escenario: Literal["pesimista", "base", "optimista"] = "base"
    # Drift real ADICIONAL sobre la trayectoria, en log por año. 0 por
    # defecto: la estimación sobre la serie real da c=-0.0015
    # (indistinguible de cero), y la tendencia ya está en la trayectoria.
    # Sobol y `src/dataset_ml.py` lo barren como supuesto incierto. En
    # "niveles" desplaza la senda a la que revierte el precio (c * t); en
    # "retornos" es la constante del AR(1) sobre retornos.
    c: float = 0.0
    phi: float = PHI_NIVELES
    # Desvío de los residuos del AR(1) sobre retornos reales; se usa el
    # mismo en las dos especificaciones (el de niveles da 0,1412).
    sigma_eps: float = 0.140922
    # "niveles" (principal) o "retornos" (estrés). `phi` tiene que ser el
    # de la especificación: ver `precio_retornos_estres()`.
    especificacion: Literal["retornos", "niveles"] = "niveles"
    # Multiplicador de TODA la trayectoria base: incertidumbre sobre el
    # nivel del precio (Sobol y dataset lo barren en 0,8-1,2).
    nivel_mult: float = 1.0
    # Trayectoria base propia (USD/kg por año 1..n), en lugar de la del Excel
    # del `escenario`. None (default) usa la del Excel. Solo para análisis
    # de sensibilidad (p. ej. precio real constante).
    trayectoria: tuple[float, ...] | None = None

    def trayectoria_base(self, n_años: int) -> np.ndarray:
        if self.trayectoria is None:
            base = cargar_modelo_v2().trayectoria_precio(self.escenario, n_años)
        elif len(self.trayectoria) != n_años:
            raise ValueError(
                f"trayectoria tiene {len(self.trayectoria)} años, se esperaban {n_años}"
            )
        else:
            base = np.asarray(self.trayectoria, dtype=float)
        return self.nivel_mult * base


def precio_retornos_estres(escenario: str = "base", **kwargs) -> ParametrosPrecioAR1:
    """Escenario de estrés: AR(1) sobre retornos, sin reversión a la trayectoria."""
    return ParametrosPrecioAR1(
        escenario=escenario, phi=PHI_RETORNOS, especificacion="retornos", **kwargs
    )


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

    o, con `especificacion="niveles"`,

        x_1 = sigma_eps * Z_1,  x_t = phi * x_(t-1) + sigma_eps * Z_t
        d_t = x_t + c * t - Var[x_t] / 2
    """
    n_simulaciones, n_años = z.shape
    if params.especificacion == "niveles":
        x = np.empty((n_simulaciones, n_años))
        x[:, 0] = params.sigma_eps * z[:, 0]
        for t in range(1, n_años):
            x[:, t] = params.phi * x[:, t - 1] + params.sigma_eps * z[:, t]
        drift = params.c * np.arange(1, n_años + 1)
        return x + drift - varianza_desvio_log(params, n_años).reshape(1, -1) / 2
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
    base del escenario por un desvío log AR(1) (ver `ParametrosPrecioAR1`).

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
