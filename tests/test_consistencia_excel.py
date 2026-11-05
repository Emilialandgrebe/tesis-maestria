"""Consistencia del motor con el modelo financiero v2 (Excel).

La corrida determinística del motor (`ParametrosMC(deterministico=True)`:
curva y precios del Excel, sin clima, vecería simulada ni ruido de OPEX) en
el escenario base a 120 ha tiene que reproducir el VAN y la TIR real de la
hoja Flujo del Excel con una diferencia relativa menor al 1%.

Los valores de referencia salen de `data/modelo_v2/referencia_indicadores.csv`
(exportado de Flujo!B5 y Flujo!B6 por `scripts/importar_modelo_v2.py`), y el
último test verifica que esos CSV sigan sincronizados con el .xlsx: si
alguien edita el Excel sin volver a correr el importador, falla.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.modelo_v2 import DIR_MODELO_V2  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    ParametrosMC,
    resumen_financiero,
    run_monte_carlo,
    run_monte_carlo_antitetico,
    run_monte_carlo_precio_historico,
)

TOLERANCIA_RELATIVA = 0.01

_RUNNERS = [run_monte_carlo, run_monte_carlo_antitetico, run_monte_carlo_precio_historico]


@pytest.fixture(scope="module")
def referencia() -> dict[str, float]:
    df = pd.read_csv(DIR_MODELO_V2 / "referencia_indicadores.csv")
    return dict(zip(df["indicador"], df["valor"]))


@pytest.fixture(scope="module")
def flujo_referencia() -> pd.DataFrame:
    return pd.read_csv(DIR_MODELO_V2 / "referencia_flujo.csv").set_index("anio")


def _corrida_deterministica(runner, n_simulaciones: int = 2):
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=120.0,
        escenario="base",
        deterministico=True,
    )
    costos = ParametrosCostos(hectareas=120.0)
    df = runner(params, costos)
    return df, resumen_financiero(df, costos), costos


def test_referencia_es_escenario_base_a_tasa_del_motor(referencia):
    """El Excel tiene que estar guardado en el escenario base (Supuestos!B22=2)
    y la tasa del Flujo tiene que ser la tasa por defecto del motor."""
    assert referencia["escenario_precio"] == 2
    assert ParametrosMC().tasa_descuento == pytest.approx(referencia["tasa_descuento_real"], rel=1e-12)


@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_coincide_con_excel(runner, referencia):
    _, resumen, _ = _corrida_deterministica(runner)
    van_excel = referencia["van_usd"]
    for van in resumen["van_neto_usd"]:
        assert abs(van - van_excel) / abs(van_excel) < TOLERANCIA_RELATIVA, (
            f"VAN motor {van:,.2f} vs Excel {van_excel:,.2f}"
        )


@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_tir_coincide_con_excel(runner, referencia):
    _, resumen, _ = _corrida_deterministica(runner)
    tir_excel = referencia["tir_real"]
    for tir in resumen["tir"]:
        assert abs(tir - tir_excel) / abs(tir_excel) < TOLERANCIA_RELATIVA, (
            f"TIR motor {tir:.6f} vs Excel {tir_excel:.6f}"
        )


def test_capex_coincide_con_excel(referencia):
    assert ParametrosCostos(hectareas=120.0).capex_inicial == pytest.approx(
        referencia["capex_total_usd"], rel=TOLERANCIA_RELATIVA
    )


@pytest.mark.parametrize(
    "columna_motor, columna_excel",
    [
        ("ingreso_usd", "ingresos_usd"),
        ("opex_usd", "opex_usd"),
        ("ebitda_usd", "ebitda_usd"),
        ("impuesto_ganancias_usd", "impuesto_ganancias_usd"),
        ("flujo_neto_usd", "flujo_evaluacion_usd"),
    ],
)
def test_flujo_anual_coincide_con_excel(columna_motor, columna_excel, flujo_referencia):
    """Año por año (1-20), para ubicar en qué rubro está una diferencia si el
    VAN deja de coincidir. Tolerancia relativa al mayor valor de la serie."""
    df, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    motor = df.set_index("año")[columna_motor]
    excel = flujo_referencia.loc[1:, columna_excel]
    escala = max(excel.abs().max(), 1.0)
    diferencia = (motor - excel).abs().max()
    assert diferencia / escala < TOLERANCIA_RELATIVA, (
        f"{columna_motor}: máxima diferencia {diferencia:,.2f} USD"
    )


def test_csv_sincronizados_con_el_excel(tmp_path):
    """Los CSV de data/modelo_v2/ tienen que ser exactamente lo que exporta
    el importador desde el .xlsx de data/referencia/."""
    pytest.importorskip("openpyxl")
    sys.path.insert(0, str(RAIZ / "scripts"))
    try:
        import importar_modelo_v2
    finally:
        sys.path.remove(str(RAIZ / "scripts"))

    importar_modelo_v2.importar(salida=tmp_path)
    for csv in sorted(DIR_MODELO_V2.glob("*.csv")):
        versionado = pd.read_csv(csv)
        regenerado = pd.read_csv(tmp_path / csv.name)
        pd.testing.assert_frame_equal(versionado, regenerado, obj=csv.name)
