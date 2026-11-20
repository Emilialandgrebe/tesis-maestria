"""Precio bayesiano del pistacho al productor (Paso 7): en verde (caso base)
o seco con cáscara (alternativa).

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
- mu, nivel de largo plazo en USD/kg vendido (USD constantes 2026): mezcla
  de dos lognormales con desvío log 0,10,
      w * LogN(media_exportacion)  +  (1 - w) * LogN(media_importacion)
  paridad de exportación y paridad de importación. Las medias dependen de
  la base de venta (`MEDIAS_PARIDAD`):
    - venta en verde a la procesadora (caso base desde 2026-11-20):
      2,5 y 5,5 USD/kg verde;
    - venta en seco con cáscara (alternativa): 5 y 9 USD/kg seco.
  Las lognormales están centradas en la MEDIA (mu_log = ln(media) - s^2/2),
  así que E[mu] = w m_exp + (1 - w) m_imp. Los escenarios del Excel son
  pesos fijos: pesimista w = 1, base w = 0,5, optimista w = 0 (en verde,
  E[mu] = 2,5 / 4 / 5,5 = los precios del bloque 9; en seco, 5 / 7 / 9 = los
  del bloque 6). mu es una prior pura (no se actualiza con FRED).
- Arranque x_0 = 0: el precio arranca en su nivel mu (no arrastra el pico
  de 2026). La corrección de convexidad usa el (phi, sigma) de cada
  simulación, así que E[precio_t | mu] = mu.

Con variables antitéticas, cada par comparte (phi, sigma, mu) y usa Z y -Z.
La dinámica (phi, sigma) es la misma en verde y en seco: el precio en verde
sigue al del seco (la procesadora paga una fracción del valor del seco).
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

# (paridad de exportación, paridad de importación), USD/kg vendido, por base
# de venta: True = en verde (caso base, bloque 9), False = seco con cáscara
# al productor (alternativa, bloque 6).
MEDIAS_PARIDAD = {True: (2.5, 5.5), False: (5.0, 9.0)}
DESVIO_LOG_PARIDAD = 0.10  # ±~20 % al 95 % en cada componente
PESO_POR_ESCENARIO = {"pesimista": 1.0, "base": 0.5, "optimista": 0.0}


@dataclass
class ParametrosPrecioBayes:
    """Parámetros del precio bayesiano. `peso_exportacion=None` toma el peso
    del `escenario` (PESO_POR_ESCENARIO); `media_exportacion` y
    `media_importacion` en None toman las de `MEDIAS_PARIDAD` según la base
    de venta."""

    escenario: Literal["pesimista", "base", "optimista"] = "base"
    peso_exportacion: float | None = None
    media_exportacion: float | None = None
    media_importacion: float | None = None
    desvio_log_paridad: float = DESVIO_LOG_PARIDAD
    ruta_posterior: str = str(RUTA_POSTERIOR)
    # Igual que en ParametrosPrecioAR1: lo completa el motor desde
    # `ParametrosCostos.venta_verde`; None fuera del motor = caso base del
    # Excel (Supuestos!B78).
    venta_verde: bool | None = None

    def __post_init__(self) -> None:
        w = self.peso()
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"peso_exportacion debe estar en [0, 1]; recibido {w!r}")

    def es_verde(self) -> bool:
        if self.venta_verde is None:
            return cargar_modelo_v2().venta_verde
        return bool(self.venta_verde)

    def medias(self) -> tuple[float, float]:
        """(media de exportación, media de importación) en USD/kg vendido."""
        m_exp, m_imp = MEDIAS_PARIDAD[self.es_verde()]
        if self.media_exportacion is not None:
            m_exp = float(self.media_exportacion)
        if self.media_importacion is not None:
            m_imp = float(self.media_importacion)
        return m_exp, m_imp

    def peso(self) -> float:
        if self.peso_exportacion is not None:
            return float(self.peso_exportacion)
        return PESO_POR_ESCENARIO[self.escenario]

    def nivel_esperado(self) -> float:
        """E[mu] = w * media_exportacion + (1 - w) * media_importacion."""
        w = self.peso()
        m_exp, m_imp = self.medias()
        return w * m_exp + (1 - w) * m_imp


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


def mascara_venta(n_años: int, venta_verde: bool | None = None) -> np.ndarray:
    """1 en los años con venta (precio del Excel > 0, años 7-20), 0 antes."""
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", n_años, venta_verde=venta_verde)
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
    m_exp, m_imp = params.medias()
    media = np.where(es_exportacion, m_exp, m_imp)
    mu = media * np.exp(s * rng.standard_normal(k) - s**2 / 2)
    return phi, sigma, mu, es_exportacion


def _precios(z: np.ndarray, phi, sigma, mu, venta_verde: bool) -> np.ndarray:
    n, n_años = z.shape
    x = np.empty((n, n_años))
    x[:, 0] = sigma * z[:, 0]
    for t in range(1, n_años):
        x[:, t] = phi * x[:, t - 1] + sigma * z[:, t]
    desvio = x - varianza_x(phi, sigma, n_años) / 2
    mascara = mascara_venta(n_años, venta_verde).reshape(1, -1)
    return mu.reshape(-1, 1) * mascara * np.exp(desvio)


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
    precios = _precios(z, phi, sigma, mu, params.es_verde())
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
    precios = _precios(z, phi, sigma, mu, params.es_verde())
    if devolver_parametros:
        return precios, pd.DataFrame({"phi": phi, "sigma": sigma, "mu": mu, "exportacion": exp_})
    return precios
