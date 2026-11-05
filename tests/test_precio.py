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
    precio_retornos_estres,
    _simular_desvio_log,
    simulate_prices_ar1,
    simulate_prices_ar1_antitetico,
    varianza_desvio_log,
)

N_AÑOS = 20
ESCENARIOS = ["pesimista", "base", "optimista"]
# Principal (niveles, el default) y estrés (retornos).
ESPECIFICACIONES = {
    "niveles": lambda escenario="base": ParametrosPrecioAR1(escenario=escenario),
    "retornos": precio_retornos_estres,
}


def _desvio_relativo_de_la_media(precios: np.ndarray, escenario: str) -> np.ndarray:
    """(media simulada / trayectoria) - 1, solo en los años con precio > 0."""
    trayectoria = cargar_modelo_v2().trayectoria_precio(escenario, N_AÑOS)
    con_precio = trayectoria > 0
    return precios.mean(axis=0)[con_precio] / trayectoria[con_precio] - 1


def test_default_es_niveles():
    params = ParametrosPrecioAR1()
    assert params.especificacion == "niveles" and params.phi == PHI_NIVELES


@pytest.mark.parametrize("especificacion", ESPECIFICACIONES)
@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_precio_medio_igual_a_trayectoria_excel_10000_simulaciones(escenario, especificacion):
    """±1% en cada año con 10.000 simulaciones antitéticas (el generador que
    usan Sobol y el dataset ML). Con muestreo directo y N=10.000 el error
    estándar del año 20 es ~0,55%, así que ±1% no se garantiza para toda
    semilla: ese caso lo cubre el test con N grande de abajo."""
    precios = simulate_prices_ar1_antitetico(
        10_000, N_AÑOS, ESPECIFICACIONES[especificacion](escenario), np.random.default_rng(42)
    )
    desvio = _desvio_relativo_de_la_media(precios, escenario)
    assert np.abs(desvio).max() < 0.01, np.round(desvio, 4)


@pytest.mark.parametrize("especificacion", ESPECIFICACIONES)
def test_precio_medio_muestreo_directo_n_grande(especificacion):
    precios = simulate_prices_ar1(
        200_000, N_AÑOS, ESPECIFICACIONES[especificacion](), np.random.default_rng(42)
    )
    desvio = _desvio_relativo_de_la_media(precios, "base")
    assert np.abs(desvio).max() < 0.005, np.round(desvio, 4)


@pytest.mark.parametrize("especificacion", ESPECIFICACIONES)
def test_varianza_cerrada_coincide_con_la_simulada(especificacion):
    params = ESPECIFICACIONES[especificacion]()
    z = np.random.default_rng(0).standard_normal((200_000, N_AÑOS))
    np.testing.assert_allclose(
        _simular_desvio_log(z, params).var(axis=0),
        varianza_desvio_log(params, N_AÑOS),
        rtol=0.02,
    )


@pytest.mark.parametrize("especificacion", ESPECIFICACIONES)
def test_mediana_queda_debajo_de_la_trayectoria(especificacion):
    """Consecuencia del criterio de valor esperado: con la corrección de
    convexidad, la mediana del precio es trayectoria * exp(-Var/2)."""
    params = ESPECIFICACIONES[especificacion]()
    precios = simulate_prices_ar1_antitetico(100_000, N_AÑOS, params, np.random.default_rng(1))
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", N_AÑOS)
    esperada = trayectoria * np.exp(-varianza_desvio_log(params, N_AÑOS) / 2)
    np.testing.assert_allclose(np.median(precios, axis=0)[6:], esperada[6:], rtol=0.01)


def test_nivel_mult_escala_la_media():
    params = ParametrosPrecioAR1(nivel_mult=1.2)
    precios = simulate_prices_ar1_antitetico(10_000, N_AÑOS, params, np.random.default_rng(42))
    desvio = _desvio_relativo_de_la_media(precios / 1.2, "base")
    assert np.abs(desvio).max() < 0.01, np.round(desvio, 4)


def test_niveles_drift_desplaza_la_media():
    """Con c != 0 en niveles, la media es trayectoria * exp(c * t)."""
    c = 0.01
    precios = simulate_prices_ar1(200_000, N_AÑOS, ParametrosPrecioAR1(c=c), np.random.default_rng(0))
    desvio = _desvio_relativo_de_la_media(precios / np.exp(c * np.arange(1, N_AÑOS + 1)), "base")
    assert np.abs(desvio).max() < 0.005, np.round(desvio, 4)
