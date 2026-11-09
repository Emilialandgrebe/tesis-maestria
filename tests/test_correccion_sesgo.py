"""Corrección de sesgo de niveles contra normales SMN 1991-2020
(`src/data/correccion_sesgo.py`, usada solo por el notebook 00)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

pymannkendall = pytest.importorskip("pymannkendall")

from src.data import correccion_sesgo as cs  # noqa: E402
from src.data.climate_fetcher import fetch_climate_data  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return fetch_climate_data()


def test_factor_precipitacion_mendoza_aero(df):
    assert cs.factor_precipitacion(df) == pytest.approx(0.349, abs=0.0005)


def test_precipitacion_corregida_iguala_la_normal_y_conserva_tendencia(df):
    corr = cs.corregir_precipitacion(df)
    anual = lambda d: d.precipitation.groupby(d.index.year).sum()  # noqa: E731
    a, c = anual(df), anual(corr)
    assert c.loc[1991:2020].mean() == pytest.approx(238.1, rel=1e-9)
    assert (c / a).std() < 1e-12  # factor constante
    mk_a = pymannkendall.hamed_rao_modification_test(a.values)
    mk_c = pymannkendall.hamed_rao_modification_test(c.values)
    assert mk_c.Tau == pytest.approx(mk_a.Tau)
    # Hamed-Rao corrige la varianza con la autocorrelación de los residuos
    # rankeados (sin tendencia de Sen): el redondeo de punto flotante puede
    # mover algún empate, así que p coincide solo aproximadamente.
    assert mk_c.p == pytest.approx(mk_a.p, rel=0.02)
    # Temperatura y ET0 sin cambios
    assert corr.temperature_2m.equals(df.temperature_2m)
    assert corr.et0_fao_evapotranspiration.equals(df.et0_fao_evapotranspiration)


def test_tmax_corregida_iguala_la_normal_mensual_y_conserva_desvio(df):
    bruta = cs._tmax_diaria(df)
    corr = cs.tmax_diaria_corregida(df)
    p = corr.loc["1991":"2020"]
    np.testing.assert_allclose(
        p.groupby(p.index.month).mean().to_numpy(),
        cs.NORMALES_SMN_1991_2020[cs.ESTACION_REFERENCIA]["tmax_mensual"],
    )
    for mes in (1, 7):
        sel = lambda s: s[s.index.month == mes]  # noqa: E731
        assert sel(corr).std() == pytest.approx(sel(bruta).std())


def test_tmin_sin_sesgo_contra_la_normal(df):
    """Justifica no corregir horas de frío: la tmin media mensual de
    ERA5-Land queda a menos de 1 °C de la normal de Mendoza Aero."""
    t = df.loc["1991":"2020", "temperature_2m"]
    tmin = t.groupby(t.index.date).min()
    tmin.index = __import__("pandas").to_datetime(tmin.index)
    media = tmin.groupby(tmin.index.month).mean().to_numpy()
    normal = np.array(cs.NORMALES_SMN_1991_2020[cs.ESTACION_REFERENCIA]["tmin_mensual"])
    assert np.abs(media - normal).max() < 1.0
