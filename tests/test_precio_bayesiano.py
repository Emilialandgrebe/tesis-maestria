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
    MEDIAS_PARIDAD,
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
    """Pesimista w = 1, base 0,5, optimista 0. En verde (caso base) E[mu] =
    2,5 / 4 / 5,5 USD/kg verde (bloque 9); en seco 5 / 7 / 9 (bloque 6)."""
    assert PESO_POR_ESCENARIO == {"pesimista": 1.0, "base": 0.5, "optimista": 0.0}
    assert MEDIAS_PARIDAD == {True: (2.5, 5.5), False: (5.0, 9.0)}
    for escenario, verde, seco in (("pesimista", 2.5, 5.0), ("base", 4.0, 7.0), ("optimista", 5.5, 9.0)):
        assert ParametrosPrecioBayes(escenario=escenario).nivel_esperado() == pytest.approx(verde)
        assert ParametrosPrecioBayes(escenario=escenario, venta_verde=False).nivel_esperado() == pytest.approx(seco)
    assert ParametrosPrecioBayes(peso_exportacion=0.25).nivel_esperado() == pytest.approx(4.75)
    assert ParametrosPrecioBayes(peso_exportacion=0.25, venta_verde=False).nivel_esperado() == pytest.approx(8.0)


@pytest.mark.parametrize("escenario, nivel", [("pesimista", 2.5), ("base", 4.0), ("optimista", 5.5)])
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
    assert par.loc[par["exportacion"], "mu"].mean() == pytest.approx(2.5, rel=0.01)
    assert par.loc[~par["exportacion"], "mu"].mean() == pytest.approx(5.5, rel=0.01)
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
    """Caso base en verde (E = 4 USD/kg verde) y alternativa en seco (7)."""
    params = ParametrosMC(n_simulaciones=2_000, modo_precio="bayesiano")
    for costos, nivel in ((ParametrosCostos(), 4.0), (ParametrosCostos(venta_verde=False), 7.0)):
        df = runner(params, costos)
        precio12 = df.loc[df["año"] == 12, "precio_usd_kg"]
        assert precio12.mean() == pytest.approx(nivel, rel=0.03)


def test_modo_bayesiano_base_incoherente_falla():
    """Paridades de una base de venta e ingresos de la otra: error."""
    with pytest.raises(ValueError):
        run_monte_carlo(
            ParametrosMC(n_simulaciones=10, modo_precio="bayesiano"),
            ParametrosCostos(venta_verde=True),
            ParametrosPrecioBayes(venta_verde=False),
        )
    with pytest.raises(ValueError):
        ParametrosPrecioBayes(peso_exportacion=1.5)


def test_precio_fijo_usa_el_nivel_esperado():
    df = run_monte_carlo_antitetico(
        ParametrosMC(n_simulaciones=10, modo_precio="bayesiano", escenario="optimista",
                     fuentes_fijas=frozenset({"precio"})),
        ParametrosCostos(),
    )
    np.testing.assert_allclose(df.loc[df["año"] >= 7, "precio_usd_kg"], 5.5)


# ---------------------------------------------------------------------------
# Tipo de cambio real: ParametrosCostos.factor_ars
# ---------------------------------------------------------------------------

def test_factor_ars_solo_toca_costos_en_pesos():
    from src.monte_carlo import resumen_financiero

    # Alternativa en seco: tiene maquila (en pesos), que en verde vale 0.
    base, shock = ParametrosCostos(venta_verde=False), ParametrosCostos(venta_verde=False, factor_ars=1.2)
    items = base.capex_items
    ars = items["moneda_pago"] == "ARS"
    esperado = (items.loc[ars, "total_escalado_usd"].sum() * 1.2
                + items.loc[~ars, "total_escalado_usd"].sum())
    assert shock.capex_inicial == pytest.approx(esperado)
    np.testing.assert_allclose(shock.costo_personal(20), 1.2 * base.costo_personal(20))
    assert shock.costo_por_kg == pytest.approx(1.2 * base.costo_por_kg)  # maquila en ARS
    assert shock.alicuota_iibb == base.alicuota_iibb
    assert shock.valor_residual == base.valor_residual
    # Con factor 1 no cambia (VAN de control del Excel, alternativa en seco).
    p = ParametrosMC(n_simulaciones=1, deterministico=True)
    van = resumen_financiero(run_monte_carlo(p, base), base)["van_neto_usd"].iloc[0]
    assert van == pytest.approx(-1_487_137.36, rel=1e-6)


def test_factor_ars_reproduce_sensibilidad_en_verde():
    """outputs/tipo_cambio_real.csv (caso en verde): VAN -2.337.910,61 con
    0,8 y -3.341.978,13 con 1,2."""
    from src.monte_carlo import resumen_financiero

    p = ParametrosMC(n_simulaciones=1, deterministico=True)
    for f, esperado in ((0.8, -2_337_910.61), (1.2, -3_341_978.13)):
        c = ParametrosCostos(venta_verde=True, factor_ars=f)
        van = resumen_financiero(run_monte_carlo(p, c), c)["van_neto_usd"].iloc[0]
        assert van == pytest.approx(esperado, rel=1e-6)
    with pytest.raises(ValueError):
        ParametrosCostos(factor_ars=0)


def test_factor_ars_estocastico_lineal_exacto():
    """Un sorteo u por simulación da el mismo VAN que ParametrosCostos(factor_ars=u)."""
    from src.monte_carlo import _componentes_deterministicos, _orquestar_resultado, resumen_financiero

    p = ParametrosMC(n_simulaciones=3, deterministico=True)
    c = ParametrosCostos()
    y, pr, ce, om = _componentes_deterministicos(p, c)
    u = np.array([0.8, 1.0, 1.2])
    van = resumen_financiero(_orquestar_resultado(y, pr, p, c, ce, om, u), c)["van_neto_usd"]
    for i, f in enumerate(u):
        cf = ParametrosCostos(factor_ars=f)
        p1 = ParametrosMC(n_simulaciones=1, deterministico=True)
        esperado = resumen_financiero(run_monte_carlo(p1, cf), cf)["van_neto_usd"].iloc[0]
        assert van.iloc[i] == pytest.approx(esperado, rel=1e-10)


def test_factor_ars_flag_por_defecto():
    from src.monte_carlo import factor_ars_activo

    assert factor_ars_activo(ParametrosMC(modo_precio="bayesiano")) is True
    assert factor_ars_activo(ParametrosMC()) is False
    assert factor_ars_activo(ParametrosMC(modo_precio="bayesiano", factor_ars_estocastico=False)) is False
    assert factor_ars_activo(ParametrosMC(factor_ars_estocastico=True)) is True


def test_factor_ars_apagado_no_cambia_corridas():
    """Sin el flag no se consume rng: el AR(1) da exactamente lo mismo."""
    a = run_monte_carlo_antitetico(ParametrosMC(n_simulaciones=200), ParametrosCostos())
    b = run_monte_carlo_antitetico(
        ParametrosMC(n_simulaciones=200, factor_ars_estocastico=False), ParametrosCostos()
    )
    np.testing.assert_array_equal(a["van_neto_usd"], b["van_neto_usd"])


def test_factor_ars_sorteo_uniforme_y_antitetico():
    from src.monte_carlo import simulate_factor_ars

    p = ParametrosMC(n_simulaciones=100_000, factor_ars_estocastico=True)
    f = simulate_factor_ars(p, np.random.default_rng(0))
    assert f.min() >= 0.8 and f.max() <= 1.2
    assert f.mean() == pytest.approx(1.0, abs=0.002)
    fa = simulate_factor_ars(p, np.random.default_rng(0), antitetico=True)
    np.testing.assert_allclose(fa[:50_000] + fa[50_000:], 2.0)


def test_factor_ars_estocastico_agrega_dispersion():
    from src.monte_carlo import resumen_financiero

    kw = dict(n_simulaciones=4_000, modo_precio="bayesiano")
    c = ParametrosCostos()
    sin = resumen_financiero(run_monte_carlo_antitetico(ParametrosMC(**kw, factor_ars_estocastico=False), c), c)
    con = resumen_financiero(run_monte_carlo_antitetico(ParametrosMC(**kw), c), c)
    assert con["van_neto_usd"].std() > sin["van_neto_usd"].std()
    # Media ~igual: E[u] = 1 y el VAN es casi lineal en u.
    assert con["van_neto_usd"].mean() == pytest.approx(sin["van_neto_usd"].mean(), rel=0.03)
