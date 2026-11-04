"""Tests de `_tir_vectorizada()` (src/monte_carlo.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.monte_carlo import _tir_vectorizada  # noqa: E402


def _tir_brentq(flujo: np.ndarray) -> float:
    return brentq(lambda r: np.sum(flujo / (1 + r) ** np.arange(len(flujo))), -0.9, 5)


def test_flujo_sin_recupero_no_devuelve_el_limite_del_clip():
    """Regresión: una fila que diverge quedaba en r=10 marcada como convergida."""
    flujo = np.array([[-2.7e6, -2.4e5, -2.6e5, -2.9e5, -3.4e5, -3.7e5, -3.9e5,
                       -1.5e5, 4.5e4, -1.3e4, -2.0e3, 2.9e5, 4.2e3, -5.0e4,
                       4.7e5, 4.8e5, 3.1e5, -1.3e5, 3.5e5, -1.6e5, 8.9e5]])
    tir = _tir_vectorizada(flujo)[0]
    assert np.isnan(tir) or tir < 1.0
    if not np.isnan(tir):
        assert abs(tir - _tir_brentq(flujo[0])) < 1e-6


def test_coincide_con_brentq_en_flujos_convencionales():
    rng = np.random.default_rng(0)
    flujos = np.hstack([
        -rng.uniform(1e6, 3e6, (50, 1)),
        rng.uniform(-5e5, 1e5, (50, 6)),
        rng.uniform(1e5, 2e6, (50, 14)),
    ])
    tir = _tir_vectorizada(flujos)
    esperado = np.array([_tir_brentq(f) for f in flujos])
    np.testing.assert_allclose(tir, esperado, atol=1e-6)


def test_flujo_sin_cambio_de_signo_es_nan():
    assert np.isnan(_tir_vectorizada(np.array([[-1e6] + [-1e4] * 20]))[0])
