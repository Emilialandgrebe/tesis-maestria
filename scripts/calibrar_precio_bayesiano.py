"""Calibra la posterior de (phi, sigma) del AR(1) del precio real (Paso 7).

Modelo sobre y_t = log del precio real anual de FRED (WPU01190106 deflactado
por CPI-U, 1991-2026), sin tendencia (drift real ~ 0):

    y_t = c + phi * y_(t-1) + sigma * e_t,   e_t ~ N(0, 1)

Priors: c plana, phi ~ U(0, 1] (admite raíz unitaria) y sigma ~
HalfNormal(0,3).

Muestreo EXACTO, sin MCMC: con la prior de referencia p(c, phi, sigma) ∝
1/sigma la posterior es normal-gamma inversa conjugada,

    sigma^2 | y ~ InvGamma((n - k)/2, SSR/2)
    (c, phi) | sigma^2, y ~ N(beta_OLS, sigma^2 (X'X)^-1)

Se sacan propuestas de ahí, se descartan las de phi fuera de (0, 1] (prior
uniforme truncada) y se re-muestrea por importancia (SIR) con peso
HalfNormal(sigma; 0,3) / (1/sigma) para pasar de la prior de referencia a la
HalfNormal. Es la misma posterior que daría PyMC/NUTS con estas priors, sin
agregar la dependencia (ver notas/PLAN_TESIS.md, Paso 7).

Uso:
    python scripts/calibrar_precio_bayesiano.py [n_sorteos]

Salida: data/processed/posterior_precio.parquet (columnas c, phi, sigma).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data.price_features import calcular_serie_anual, deflactar_por_cpi  # noqa: E402
from src.data.price_fetcher import fetch_cpi_data, fetch_price_data  # noqa: E402
from src.precio_bayesiano import RUTA_POSTERIOR  # noqa: E402

ESCALA_HALFNORMAL_SIGMA = 0.3
SEMILLA = 2026


def serie_log_real() -> pd.Series:
    real = deflactar_por_cpi(fetch_price_data(), fetch_cpi_data())
    return np.log(calcular_serie_anual(real).astype(float))


def muestrear_posterior(y: np.ndarray, n_sorteos: int = 4_000, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    Y = y[1:]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    ssr = float(((Y - X @ beta) ** 2).sum())
    L = np.linalg.cholesky(XtX_inv)

    aceptados = []
    total = 0
    while sum(len(a) for a in aceptados) < 50 * n_sorteos:
        m = 200_000
        sigma2 = (ssr / 2) / rng.gamma((n - k) / 2, 1.0, size=m)  # InvGamma
        b = beta + np.sqrt(sigma2)[:, None] * (rng.standard_normal((m, k)) @ L.T)
        ok = (b[:, 1] > 0) & (b[:, 1] <= 1)
        total += m
        aceptados.append(np.column_stack([b[ok], np.sqrt(sigma2[ok])]))
    prop = np.vstack(aceptados)
    sigma = prop[:, 2]
    # Peso de importancia: HalfNormal(sigma; s) / (1 / sigma).
    logw = np.log(sigma) - sigma**2 / (2 * ESCALA_HALFNORMAL_SIGMA**2)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    idx = rng.choice(len(prop), size=n_sorteos, replace=True, p=w)
    ess = 1.0 / (w**2).sum()
    print(f"propuestas {total:,}, aceptadas phi en (0,1]: {len(prop):,} "
          f"({len(prop) / total:.1%}), ESS de la reponderación: {ess:,.0f}")
    return pd.DataFrame({"c": prop[idx, 0], "phi": prop[idx, 1], "sigma": prop[idx, 2]})


if __name__ == "__main__":
    n_sorteos = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000
    s = serie_log_real()
    print(f"Serie: {s.index.min()}-{s.index.max()}, n = {len(s)}")
    post = muestrear_posterior(s.to_numpy(), n_sorteos)
    RUTA_POSTERIOR.parent.mkdir(parents=True, exist_ok=True)
    post.to_parquet(RUTA_POSTERIOR)
    q = post.quantile([0.05, 0.5, 0.95])
    print(q.round(4).to_string())
    print(f"P(phi > 0,95) = {(post['phi'] > 0.95).mean():.3f}   "
          f"P(phi > 0,99) = {(post['phi'] > 0.99).mean():.3f}")
    print(f"Guardado en {RUTA_POSTERIOR}")
