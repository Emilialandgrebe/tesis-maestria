"""Tests del precio bayesiano (src/precio_bayesiano.py, Paso 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.monte_carlo import ParametrosMC, run_monte_carlo, run_monte_carlo_antitetico  # noqa: E402
from src.precio_bayesiano import (  # noqa: E402
    PESO_POR_ESCENARIO,
    RUTA_POSTERIOR,
    ParametrosPrecioBayes,
    cargar_posterior,
    mascara_venta,
    simulate_prices_bayes,
    simulate_prices_bayes_antitetico,
    varianza_x,
)

N_AÑOS = 20
pytestmark = pytest.mark.skipif(
    not RUTA_POSTERIOR.exists(), reason="falta data/processed/posterior_precio.parquet"
)


def test_posterior_en_soporte_de_la_prior():
    post = cargar_posterior()
    assert len(post) >= 1_000
    assert (post["phi"] > 0).all() and (post["phi"] <= 1).all()
    assert (post["sigma"] > 0).all()
    # Calibración 1991-2026: mediana de phi ~0,87 y de sigma ~0,14.
    assert 0.8 < post["phi"].median() < 0.93
    assert 0.12 < post["sigma"].median() < 0.17


def test_pesos_y_niveles_de_los_escenarios():
    """Pesimista w = 1, base 0,5, optimista 0; E[mu] = 5 / 7 / 9 (Excel)."""
    assert PESO_POR_ESCENARIO == {"pesimista": 1.0, "base": 0.5, "optimista": 0.0}
    for escenario, nivel in (("pesimista", 5.0), ("base", 7.0), ("optimista", 9.0)):
        assert ParametrosPrecioBayes(escenario=escenario).nivel_esperado() == pytest.approx(nivel)
    assert ParametrosPrecioBayes(peso_exportacion=0.25).nivel_esperado() == pytest.approx(8.0)


@pytest.mark.parametrize("escenario, nivel", [("pesimista", 5.0), ("base", 7.0), ("optimista", 9.0)])
def test_media_del_precio_igual_al_nivel(escenario, nivel):
    """Lognormales centradas en la media y corrección de convexidad por
    simulación: E[precio_t] = E[mu] en todos los años con venta."""
    precios = simulate_prices_bayes(
        200_000, N_AÑOS, ParametrosPrecioBayes(escenario=escenario), np.random.default_rng(7)
    )
    np.testing.assert_array_equal(precios[:, :6], 0.0)
    np.testing.assert_allclose(precios[:, 6:].mean(axis=0), nivel, rtol=0.01)


def test_mezcla_bimodal_con_peso():
    _, par = simulate_prices_bayes(
        100_000, N_AÑOS, ParametrosPrecioBayes(peso_exportacion=0.3), np.random.default_rng(1),
        devolver_parametros=True,
    )
    assert par["exportacion"].mean() == pytest.approx(0.3, abs=0.01)
    assert par.loc[par["exportacion"], "mu"].mean() == pytest.approx(5.0, rel=0.01)
    assert par.loc[~par["exportacion"], "mu"].mean() == pytest.approx(9.0, rel=0.01)
    assert np.log(par.loc[par["exportacion"], "mu"]).std() == pytest.approx(0.10, rel=0.03)


def test_antiteticos_comparten_parametros():
    n = 1_001
    precios, par = simulate_prices_bayes_antitetico(
        n, N_AÑOS, ParametrosPrecioBayes(), np.random.default_rng(3), devolver_parametros=True
    )
    mitad = n // 2
    for col in ("phi", "sigma", "mu"):
        np.testing.assert_array_equal(par[col][:mitad].to_numpy(), par[col][mitad: 2 * mitad].to_numpy())
    assert precios.shape == (n, N_AÑOS)


def test_varianza_con_phi_uno_es_t_sigma2():
    v = varianza_x(np.array([1.0, 0.999999999999]), np.array([0.2, 0.2]), 5)
    np.testing.assert_allclose(v[0], 0.04 * np.arange(1, 6))
    assert np.isfinite(v).all()


def test_arranque_en_el_nivel():
    """x_0 = 0: la mediana del año 7 queda cerca de mu (sin pico de 2026)."""
    precios, par = simulate_prices_bayes(
        50_000, N_AÑOS, ParametrosPrecioBayes(peso_exportacion=1.0), np.random.default_rng(5),
        devolver_parametros=True,
    )
    ratio = precios[:, 6] / par["mu"].to_numpy()
    assert np.mean(ratio) == pytest.approx(1.0, rel=0.01)


def test_mascara_venta_años_7_a_20():
    m = mascara_venta(N_AÑOS)
    np.testing.assert_array_equal(m[:6], 0.0)
    np.testing.assert_array_equal(m[6:], 1.0)


@pytest.mark.parametrize("runner", [run_monte_carlo, run_monte_carlo_antitetico], ids=lambda f: f.__name__)
def test_motor_modo_bayesiano(runner):
    df = runner(ParametrosMC(n_simulaciones=2_000, modo_precio="bayesiano"), ParametrosCostos())
    precio12 = df.loc[df["año"] == 12, "precio_usd_kg"]
    assert precio12.mean() == pytest.approx(7.0, rel=0.03)


def test_modo_bayesiano_rechaza_venta_verde():
    with pytest.raises(ValueError):
        run_monte_carlo(
            ParametrosMC(n_simulaciones=10, modo_precio="bayesiano"),
            ParametrosCostos(venta_verde=True),
        )
    with pytest.raises(ValueError):
        ParametrosPrecioBayes(peso_exportacion=1.5)


def test_precio_fijo_usa_el_nivel_esperado():
    df = run_monte_carlo_antitetico(
        ParametrosMC(n_simulaciones=10, modo_precio="bayesiano", escenario="optimista",
                     fuentes_fijas=frozenset({"precio"})),
        ParametrosCostos(),
    )
    np.testing.assert_allclose(df.loc[df["año"] >= 7, "precio_usd_kg"], 9.0)
