"""Criterio de valor esperado del precio (src/precio_estocastico.py).

La trayectoria del Excel v2 tiene que ser la MEDIA del precio simulado en
cada año, no la mediana: E[exp(desvío_t)] = 1 con el drift c = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modelo_v2 import cargar_modelo_v2  # noqa: E402
from src.precio_estocastico import (  # noqa: E402
    PHI_NIVELES,
    ParametrosPrecioAR1,
    _simular_desvio_log,
    simulate_prices_ar1,
    simulate_prices_ar1_antitetico,
    varianza_desvio_log,
)

N_AÑOS = 20
ESCENARIOS = ["pesimista", "base", "optimista"]


def _desvio_relativo_de_la_media(precios: np.ndarray, escenario: str) -> np.ndarray:
    """(media simulada / trayectoria) - 1, solo en los años con precio > 0."""
    trayectoria = cargar_modelo_v2().trayectoria_precio(escenario, N_AÑOS)
    con_precio = trayectoria > 0
    return precios.mean(axis=0)[con_precio] / trayectoria[con_precio] - 1


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_precio_medio_igual_a_trayectoria_excel_10000_simulaciones(escenario):
    """±1% en cada año con 10.000 simulaciones antitéticas (el generador que
    usan Sobol y el dataset ML). Con muestreo directo y N=10.000 el error
    estándar del año 20 es ~0,55%, así que ±1% no se garantiza para toda
    semilla: ese caso lo cubre el test con N grande de abajo."""
    precios = simulate_prices_ar1_antitetico(
        10_000, N_AÑOS, ParametrosPrecioAR1(escenario=escenario), np.random.default_rng(42)
    )
    desvio = _desvio_relativo_de_la_media(precios, escenario)
    assert np.abs(desvio).max() < 0.01, np.round(desvio, 4)


def test_precio_medio_muestreo_directo_n_grande():
    precios = simulate_prices_ar1(
        200_000, N_AÑOS, ParametrosPrecioAR1(escenario="base"), np.random.default_rng(42)
    )
    desvio = _desvio_relativo_de_la_media(precios, "base")
    assert np.abs(desvio).max() < 0.005, np.round(desvio, 4)


def test_varianza_cerrada_coincide_con_la_simulada():
    params = ParametrosPrecioAR1()
    z = np.random.default_rng(0).standard_normal((200_000, N_AÑOS))
    np.testing.assert_allclose(
        _simular_desvio_log(z, params).var(axis=0),
        varianza_desvio_log(params, N_AÑOS),
        rtol=0.02,
    )


def test_mediana_queda_debajo_de_la_trayectoria():
    """Consecuencia del criterio de valor esperado: con la corrección de
    convexidad, la mediana del precio es trayectoria * exp(-Var/2)."""
    params = ParametrosPrecioAR1(escenario="base")
    precios = simulate_prices_ar1_antitetico(100_000, N_AÑOS, params, np.random.default_rng(1))
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", N_AÑOS)
    esperada = trayectoria * np.exp(-varianza_desvio_log(params, N_AÑOS) / 2)
    np.testing.assert_allclose(np.median(precios, axis=0)[6:], esperada[6:], rtol=0.01)


PRECIO_NIVELES = ParametrosPrecioAR1(phi=PHI_NIVELES, especificacion="niveles")


def test_niveles_precio_medio_igual_a_trayectoria():
    precios = simulate_prices_ar1_antitetico(10_000, N_AÑOS, PRECIO_NIVELES, np.random.default_rng(42))
    desvio = _desvio_relativo_de_la_media(precios, "base")
    assert np.abs(desvio).max() < 0.01, np.round(desvio, 4)


def test_niveles_varianza_cerrada_coincide_con_la_simulada():
    z = np.random.default_rng(0).standard_normal((200_000, N_AÑOS))
    np.testing.assert_allclose(
        _simular_desvio_log(z, PRECIO_NIVELES).var(axis=0),
        varianza_desvio_log(PRECIO_NIVELES, N_AÑOS),
        rtol=0.02,
    )
