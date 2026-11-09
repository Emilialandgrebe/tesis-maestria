"""Escenario de estrés climático (`ParametrosMC.estres_climatico`).

Las medias de horas de frío y tmax de verano se desplazan año a año según
la pendiente de Sen de ERA5-Land extrapolada desde 2024, sin renormalizar el
rendimiento ideal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.monte_carlo import (  # noqa: E402
    ANIO_CERO_PROYECTO,
    ANIO_FIN_SERIE_CLIMATICA,
    PENDIENTES_ESTRES_CLIMATICO,
    RENDIMIENTO_IDEAL_KG_HA,
    ParametrosMC,
    medias_climaticas,
    run_monte_carlo,
    run_monte_carlo_antitetico,
)


def test_sin_estres_medias_constantes():
    p = ParametrosMC()
    frio, calor = medias_climaticas(p)
    assert frio.shape == calor.shape == (1, p.n_años)
    assert np.all(frio == p.horas_frio_media)
    assert np.all(calor == p.calor_verano_media)


@pytest.mark.parametrize("variante", ["puntual", "extremo"])
def test_desplazamiento_lineal_desde_2024(variante):
    p = ParametrosMC(estres_climatico=True, estres_variante=variante)
    frio, calor = medias_climaticas(p)
    pend = PENDIENTES_ESTRES_CLIMATICO[variante]
    años = ANIO_CERO_PROYECTO + np.arange(1, p.n_años + 1) - ANIO_FIN_SERIE_CLIMATICA
    np.testing.assert_allclose(frio[0], p.horas_frio_media + pend["horas_frio"] * años)
    np.testing.assert_allclose(calor[0], p.calor_verano_media + pend["calor_verano"] * años)
    # Año 1 del proyecto = 2027: tres años después del fin de la serie
    assert años[0] == 3


def test_pendientes_son_las_de_era5_land():
    assert PENDIENTES_ESTRES_CLIMATICO == {
        "puntual": {"horas_frio": -2.65, "calor_verano": 0.088},
        "extremo": {"horas_frio": -11.92, "calor_verano": 0.131},
    }


def test_variante_invalida():
    with pytest.raises(ValueError):
        medias_climaticas(ParametrosMC(estres_climatico=True, estres_variante="otra"))


def test_no_se_renormaliza_el_rendimiento_ideal():
    p = ParametrosMC(estres_climatico=True, estres_variante="extremo")
    assert p.rendimiento_plena == RENDIMIENTO_IDEAL_KG_HA


@pytest.mark.parametrize("runner", [run_monte_carlo, run_monte_carlo_antitetico], ids=lambda f: f.__name__)
def test_estres_extremo_baja_el_rendimiento_y_mismo_consumo_de_rng(runner):
    """Con el estrés extremo (frío -11,92 h/año) la meseta cae. En la
    versión antitética el precio sale idéntico (todo se sortea por PPF, el
    consumo de `rng` no depende de las medias). En `run_monte_carlo` no:
    `rng.beta` muestrea por rechazo y consume una cantidad de números que
    depende de los parámetros, así que el precio cambia de realización."""
    base = runner(ParametrosMC(n_simulaciones=2_000))
    estres = runner(ParametrosMC(n_simulaciones=2_000, estres_climatico=True, estres_variante="extremo"))
    if runner is run_monte_carlo_antitetico:
        np.testing.assert_array_equal(base["precio_usd_kg"], estres["precio_usd_kg"])
    meseta = lambda df: df[df["año"] >= 12]["rendimiento_kg_ha"].mean()  # noqa: E731
    assert meseta(estres) < meseta(base) * 0.92


def test_clima_fijo_usa_la_media_de_cada_año():
    """Con "clima" en fuentes_fijas y estrés activo, el factor fijo sigue la
    media desplazada de cada año: el rendimiento de la meseta decrece."""
    p = ParametrosMC(
        n_simulaciones=500, estres_climatico=True, estres_variante="extremo",
        fuentes_fijas=frozenset({"clima", "veceria", "plantas"}),
    )
    y = run_monte_carlo(p).pivot(index="simulacion", columns="año", values="rendimiento_kg_ha")
    assert y.nunique(axis=0).max() == 1  # sin ruido: todas las filas iguales
    meseta = y.iloc[0].loc[12:20].to_numpy()
    assert np.all(np.diff(meseta) < 0)
