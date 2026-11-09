"""Traslado de quebrantos impositivos (FIFO con vencimiento) y activación
del OPEX pre-productivo.

Casos chicos armados a mano para la lógica de `quebrantos_utilizados()`
(hoja Quebrantos del Excel v2) y de `opex_pre_activado()` (Flujo!33-34), y
la propiedad que importa en Monte Carlo: cada simulación lleva su propio
saldo de quebrantos y su propio total activado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos, opex_pre_activado, quebrantos_utilizados  # noqa: E402
from src.monte_carlo import ParametrosMC, run_monte_carlo  # noqa: E402


def _q(ebit, anios=5):
    return quebrantos_utilizados(np.array([ebit], dtype=float), anios)[0]


def test_sin_perdidas_no_hay_quebrantos():
    np.testing.assert_array_equal(_q([10, 20, 30]), [0, 0, 0])


def test_perdida_se_usa_contra_ganancias_siguientes():
    # pérdida de 100 en el año 1, ganancias de 60 y 60
    np.testing.assert_allclose(_q([-100, 60, 60]), [0, 60, 40])


def test_nunca_se_usa_mas_que_el_ebit_positivo():
    np.testing.assert_allclose(_q([-1000, 50, 30]), [0, 50, 30])


def test_vencimiento_a_los_n_anios():
    # pérdida en el año 1 (índice 0), plazo 2: usable en los años 2 y 3, no en el 4
    np.testing.assert_allclose(_q([-100, 0, 0, 80], anios=2), [0, 0, 0, 0])
    np.testing.assert_allclose(_q([-100, 0, 80, 80], anios=2), [0, 0, 80, 0])


def test_fifo_usa_primero_la_mas_antigua():
    # plazo 2. Pérdidas de 50 (año 1) y 50 (año 2). Año 3: ganancia 50 -> usa
    # la del año 1 (que vence ese año). Año 4: ganancia 50 -> usa la del año 2.
    # Con LIFO, la del año 1 vencería sin usarse y el año 4 no tendría saldo.
    np.testing.assert_allclose(_q([-50, -50, 50, 50], anios=2), [0, 0, 50, 50])


def test_anio_con_perdida_no_usa_quebrantos():
    np.testing.assert_allclose(_q([-100, -20, 200]), [0, 0, 120])


def test_filas_independientes():
    """Cada fila (simulación) tiene su propio saldo: calcular la matriz
    entera da lo mismo que calcular fila por fila."""
    rng = np.random.default_rng(0)
    ebit = rng.normal(0, 100, size=(50, 20))
    junto = quebrantos_utilizados(ebit, 5)
    separado = np.vstack([quebrantos_utilizados(ebit[[i]], 5) for i in range(len(ebit))])
    np.testing.assert_array_equal(junto, separado)


def test_quebrantos_anios_invalido():
    with pytest.raises(ValueError):
        ParametrosCostos(quebrantos_anios=0)
    with pytest.raises(ValueError):
        ParametrosCostos(quebrantos_anios=2.5)


def test_monte_carlo_quebrantos_por_simulacion():
    """En Monte Carlo los quebrantos dependen de la trayectoria: no son una
    serie fija igual para todas las simulaciones, y con el interruptor
    apagado son cero en todas."""
    params = ParametrosMC(n_simulaciones=200, semilla=7)
    con = run_monte_carlo(params, ParametrosCostos(quebrantos=True))
    sin = run_monte_carlo(params, ParametrosCostos(quebrantos=False))

    assert (sin["quebrantos_utilizados_usd"] == 0).all()
    tabla = con.pivot(index="simulacion", columns="año", values="quebrantos_utilizados_usd")
    assert tabla.nunique(axis=0).max() > 1  # varía entre simulaciones

    # Mismas trayectorias de ingresos y OPEX con y sin quebrantos
    np.testing.assert_array_equal(con["ingreso_usd"], sin["ingreso_usd"])
    np.testing.assert_array_equal(con["opex_usd"], sin["opex_usd"])
    assert (con["impuesto_ganancias_usd"] <= sin["impuesto_ganancias_usd"] + 1e-6).all()


def test_opex_pre_activado_caso_chico():
    """Último año pre-productivo 6 (Supuestos!B8): se activan los años 1-6
    y la cuota (total / vida útil) se deduce desde el año 7 durante la vida
    útil, cortada por el horizonte."""
    opex = np.arange(1.0, 11.0).reshape(1, 10)  # años 1..10
    costos = ParametrosCostos(activar_opex_pre=True, vida_util_implantacion=3)
    activado, amort = opex_pre_activado(opex, costos)
    np.testing.assert_array_equal(activado[0], [1, 2, 3, 4, 5, 6, 0, 0, 0, 0])
    np.testing.assert_allclose(amort[0], [0, 0, 0, 0, 0, 0, 7, 7, 7, 0])

    activado, amort = opex_pre_activado(opex, ParametrosCostos(activar_opex_pre=False))
    assert not activado.any() and not amort.any()


def test_vida_util_implantacion_invalida():
    with pytest.raises(ValueError):
        ParametrosCostos(vida_util_implantacion=0)
    with pytest.raises(ValueError):
        ParametrosCostos(vida_util_implantacion=12.5)


def test_monte_carlo_activacion_por_simulacion():
    """El total activado es el OPEX pre-productivo de cada simulación (el
    OPEX es estocástico), y la activación no cambia ingresos ni OPEX."""
    params = ParametrosMC(n_simulaciones=200, semilla=7)
    con = run_monte_carlo(params, ParametrosCostos(activar_opex_pre=True))
    sin = run_monte_carlo(params, ParametrosCostos(activar_opex_pre=False))

    assert (sin["opex_pre_activado_usd"] == 0).all()
    assert (sin["amortizacion_implantacion_usd"] == 0).all()
    total = con.groupby("simulacion")["opex_pre_activado_usd"].sum()
    assert total.nunique() > 1
    cuota = con[con["año"] == 7].set_index("simulacion")["amortizacion_implantacion_usd"]
    np.testing.assert_allclose(cuota, total / 20)

    np.testing.assert_array_equal(con["ingreso_usd"], sin["ingreso_usd"])
    np.testing.assert_array_equal(con["opex_usd"], sin["opex_usd"])
