"""Precio bayesiano del pistacho seco con cáscara al productor (Paso 7).

Mismo esqueleto que el AR(1) en niveles de `src/precio_estocastico.py`, pero
con los tres parámetros inciertos y sorteados por simulación:

    precio_t = mu * exp(x_t - Var[x_t | phi, sigma] / 2)     (años con venta)
    x_t = phi * x_(t-1) + sigma * Z_t,   x_0 = 0

- (phi, sigma): sorteos de la posterior del AR(1) sobre el log del precio
  real de FRED (WPU01190106 deflactado, 1991-2026, sin tendencia), guardada
  en `data/processed/posterior_precio.parquet` por
  `scripts/calibrar_precio_bayesiano.py`. Prior phi ~ U(0, 1] (admite raíz
  unitaria: no se rechaza phi = 1) y sigma ~ HalfNormal(0,3). FRED es un
  índice: informa la DINÁMICA, no el nivel en USD/kg.
- mu, nivel de largo plazo en USD/kg seco al productor (USD constantes
  2026): mezcla de dos lognormales con desvío log 0,10,
      w * LogN(media 5)  +  (1 - w) * LogN(media 9)
  paridad de exportación (~5) y paridad de importación (~9). Las
  lognormales están centradas en la MEDIA (mu_log = ln(media) - s^2/2), así
  que E[mu] = 5 w + 9 (1 - w). Los escenarios del Excel son pesos fijos:
  pesimista w = 1 (E = 5), base w = 0,5 (E = 7), optimista w = 0 (E = 9).
  mu es una prior pura (no se actualiza con FRED).
- Arranque x_0 = 0: el precio arranca en su nivel mu (no arrastra el pico
  de 2026). La corrección de convexidad usa el (phi, sigma) de cada
  simulación, así que E[precio_t | mu] = mu.

Con variables antitéticas, cada par comparte (phi, sigma, mu) y usa Z y -Z.
Solo para venta en seco (bloque 6): con venta en verde no hay paridades
definidas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.modelo_v2 import cargar_modelo_v2
from src.precio_estocastico import _generar_normales_antiteticos

RUTA_POSTERIOR = Path(__file__).resolve().parents[1] / "data" / "processed" / "posterior_precio.parquet"

MEDIA_EXPORTACION = 5.0   # USD/kg seco al productor, paridad de exportación
MEDIA_IMPORTACION = 9.0   # USD/kg seco al productor, paridad de importación
DESVIO_LOG_PARIDAD = 0.10  # ±~20 % al 95 % en cada componente
PESO_POR_ESCENARIO = {"pesimista": 1.0, "base": 0.5, "optimista": 0.0}


@dataclass
class ParametrosPrecioBayes:
    """Parámetros del precio bayesiano. `peso_exportacion=None` toma el peso
    del `escenario` (PESO_POR_ESCENARIO)."""

    escenario: Literal["pesimista", "base", "optimista"] = "base"
    peso_exportacion: float | None = None
    media_exportacion: float = MEDIA_EXPORTACION
    media_importacion: float = MEDIA_IMPORTACION
    desvio_log_paridad: float = DESVIO_LOG_PARIDAD
    ruta_posterior: str = str(RUTA_POSTERIOR)
    # Igual que en ParametrosPrecioAR1: lo completa el motor desde
    # `ParametrosCostos.venta_verde`. Solo se admite False.
    venta_verde: bool | None = None

    def __post_init__(self) -> None:
        w = self.peso()
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"peso_exportacion debe estar en [0, 1]; recibido {w!r}")
        if self.venta_verde:
            raise ValueError("el precio bayesiano solo está definido para venta en seco")

    def peso(self) -> float:
        if self.peso_exportacion is not None:
            return float(self.peso_exportacion)
        return PESO_POR_ESCENARIO[self.escenario]

    def nivel_esperado(self) -> float:
        """E[mu] = w * media_exportacion + (1 - w) * media_importacion."""
        w = self.peso()
        return w * self.media_exportacion + (1 - w) * self.media_importacion


@lru_cache(maxsize=4)
def cargar_posterior(ruta: str = str(RUTA_POSTERIOR)) -> pd.DataFrame:
    """Sorteos de la posterior de (phi, sigma). Falla con un mensaje claro si
    todavía no se corrió la calibración."""
    ruta_p = Path(ruta)
    if not ruta_p.exists():
        raise FileNotFoundError(
            f"No existe {ruta_p}. Correr `python scripts/calibrar_precio_bayesiano.py`."
        )
    df = pd.read_parquet(ruta_p)
    if not {"phi", "sigma"} <= set(df.columns):
        raise ValueError(f"{ruta_p} tiene que tener columnas phi y sigma")
    return df


def mascara_venta(n_años: int) -> np.ndarray:
    """1 en los años con venta (precio del Excel > 0, años 7-20), 0 antes."""
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", n_años, venta_verde=False)
    return (trayectoria > 0).astype(float)


def varianza_x(phi: np.ndarray, sigma: np.ndarray, n_años: int) -> np.ndarray:
    """Var[x_t | phi, sigma] = sigma^2 (1 - phi^(2t)) / (1 - phi^2), t = 1..n,
    forma (n, n_años); con phi -> 1 el límite es t sigma^2."""
    phi = np.asarray(phi, dtype=float).reshape(-1, 1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1, 1)
    t = np.arange(1, n_años + 1).reshape(1, -1)
    denom = 1.0 - phi**2
    casi_uno = np.abs(denom) < 1e-10
    seguro = np.where(casi_uno, 1.0, denom)
    return sigma**2 * np.where(casi_uno, t, (1.0 - phi ** (2 * t)) / seguro)


def _sortear_parametros(
    k: int, params: ParametrosPrecioBayes, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """k juegos (phi, sigma, mu, es_exportacion)."""
    post = cargar_posterior(params.ruta_posterior)
    idx = rng.integers(0, len(post), size=k)
    phi = post["phi"].to_numpy()[idx]
    sigma = post["sigma"].to_numpy()[idx]
    es_exportacion = rng.random(k) < params.peso()
    s = params.desvio_log_paridad
    media = np.where(es_exportacion, params.media_exportacion, params.media_importacion)
    mu = media * np.exp(s * rng.standard_normal(k) - s**2 / 2)
    return phi, sigma, mu, es_exportacion


def _precios(z: np.ndarray, phi, sigma, mu) -> np.ndarray:
    n, n_años = z.shape
    x = np.empty((n, n_años))
    x[:, 0] = sigma * z[:, 0]
    for t in range(1, n_años):
        x[:, t] = phi * x[:, t - 1] + sigma * z[:, t]
    desvio = x - varianza_x(phi, sigma, n_años) / 2
    return mu.reshape(-1, 1) * mascara_venta(n_años).reshape(1, -1) * np.exp(desvio)


def simulate_prices_bayes(
    n_simulaciones: int,
    n_años: int,
    params: ParametrosPrecioBayes,
    rng: np.random.Generator,
    devolver_parametros: bool = False,
):
    """Precios (n_simulaciones, n_años) con muestreo directo."""
    phi, sigma, mu, exp_ = _sortear_parametros(n_simulaciones, params, rng)
    z = rng.standard_normal((n_simulaciones, n_años))
    precios = _precios(z, phi, sigma, mu)
    if devolver_parametros:
        return precios, pd.DataFrame({"phi": phi, "sigma": sigma, "mu": mu, "exportacion": exp_})
    return precios


def simulate_prices_bayes_antitetico(
    n_simulaciones: int,
    n_años: int,
    params: ParametrosPrecioBayes,
    rng: np.random.Generator,
    devolver_parametros: bool = False,
):
    """Precios con variables antitéticas: la fila i y la i + n//2 comparten
    (phi, sigma, mu) y usan Z y -Z (si n es impar, la última fila es libre)."""
    mitad = n_simulaciones // 2
    k = mitad + n_simulaciones % 2
    phi_k, sigma_k, mu_k, exp_k = _sortear_parametros(k, params, rng)
    mapa = np.r_[np.arange(mitad), np.arange(mitad), np.arange(mitad, k)]
    phi, sigma, mu, exp_ = phi_k[mapa], sigma_k[mapa], mu_k[mapa], exp_k[mapa]
    z = _generar_normales_antiteticos(n_simulaciones, n_años, rng)
    precios = _precios(z, phi, sigma, mu)
    if devolver_parametros:
        return precios, pd.DataFrame({"phi": phi, "sigma": sigma, "mu": mu, "exportacion": exp_})
    return precios
