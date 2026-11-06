"""Consistencia del motor con el modelo financiero v2 (Excel).

La corrida determinística del motor (`ParametrosMC(deterministico=True)`:
curva y precios del Excel, sin clima, vecería simulada ni ruido de OPEX) en
el escenario base a 120 ha tiene que reproducir el VAN y la TIR real de la
hoja Flujo del Excel con una diferencia relativa menor al 1%, en las dos
posiciones del interruptor de quebrantos (Supuestos!B64):

- Apagado (como está guardado el Excel): los valores de referencia salen de
  `data/modelo_v2/referencia_indicadores.csv` y `referencia_flujo.csv`
  (exportados por `scripts/importar_modelo_v2.py`). El último test verifica
  que esos CSV sigan sincronizados con el .xlsx: si alguien edita el Excel
  sin volver a correr el importador, falla.
- Encendido: el .xlsx guarda solo los valores calculados con el interruptor
  en 0, así que la referencia con B64=1 está fijada abajo
  (`REFERENCIA_CON_QUEBRANTOS`), tomada del Excel recalculado con B64=1 y
  B65=5. Si cambia el Excel, hay que volver a tomarla.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.modelo_v2 import DIR_MODELO_V2, cargar_modelo_v2  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    ParametrosMC,
    resumen_financiero,
    run_monte_carlo,
    run_monte_carlo_antitetico,
    run_monte_carlo_precio_historico,
)

TOLERANCIA_RELATIVA = 0.01

_RUNNERS = [run_monte_carlo, run_monte_carlo_antitetico, run_monte_carlo_precio_historico]

# Excel con Supuestos!B64 = 1 (traslado de quebrantos activo, plazo 5 años),
# escenario base, 120 ha. Flujo!B5, Flujo!B6 y Flujo!32 (= Quebrantos!55).
REFERENCIA_CON_QUEBRANTOS = {
    "van_usd": -67_346.0,
    "tir_real": 0.140464,
    # año -> quebranto utilizado (USD); el resto de los años, cero
    "quebrantos_utilizados_usd": {8: 471_852.0, 9: 1_524_451.0, 10: 267_989.0},
}


@pytest.fixture(scope="module")
def referencia() -> dict[str, float]:
    df = pd.read_csv(DIR_MODELO_V2 / "referencia_indicadores.csv")
    return dict(zip(df["indicador"], df["valor"]))


@pytest.fixture(scope="module")
def flujo_referencia() -> pd.DataFrame:
    return pd.read_csv(DIR_MODELO_V2 / "referencia_flujo.csv").set_index("anio")


def _referencia_indicadores(quebrantos: bool, referencia: dict[str, float]) -> dict[str, float]:
    return REFERENCIA_CON_QUEBRANTOS if quebrantos else referencia


def _corrida_deterministica(runner, n_simulaciones: int = 2, quebrantos: bool = False):
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=120.0,
        escenario="base",
        deterministico=True,
    )
    costos = ParametrosCostos(hectareas=120.0, quebrantos=quebrantos)
    df = runner(params, costos)
    return df, resumen_financiero(df, costos), costos


_INTERRUPTOR = pytest.mark.parametrize(
    "quebrantos", [False, True], ids=["sin_quebrantos", "con_quebrantos"]
)


def test_referencia_es_escenario_base_a_tasa_del_motor(referencia):
    """El Excel tiene que estar guardado en el escenario base (Supuestos!B22=2)
    y la tasa del Flujo tiene que ser la tasa por defecto del motor."""
    assert referencia["escenario_precio"] == 2
    assert ParametrosMC().tasa_descuento == pytest.approx(referencia["tasa_descuento_real"], rel=1e-12)


def test_interruptor_de_quebrantos_coincide_con_excel():
    """Los CSV de referencia se exportan con el interruptor apagado
    (Supuestos!B64=0), que es también el default del motor, y el plazo por
    defecto del motor es el de Supuestos!B65."""
    supuestos = cargar_modelo_v2().supuestos
    assert supuestos["quebrantos_aplicar"] == 0
    assert ParametrosCostos().quebrantos is False
    assert ParametrosCostos().quebrantos_anios == supuestos["quebrantos_anios"] == 5


@_INTERRUPTOR
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_coincide_con_excel(runner, quebrantos, referencia):
    _, resumen, _ = _corrida_deterministica(runner, quebrantos=quebrantos)
    van_excel = _referencia_indicadores(quebrantos, referencia)["van_usd"]
    for van in resumen["van_neto_usd"]:
        assert abs(van - van_excel) / abs(van_excel) < TOLERANCIA_RELATIVA, (
            f"VAN motor {van:,.2f} vs Excel {van_excel:,.2f}"
        )


@_INTERRUPTOR
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_tir_coincide_con_excel(runner, quebrantos, referencia):
    _, resumen, _ = _corrida_deterministica(runner, quebrantos=quebrantos)
    tir_excel = _referencia_indicadores(quebrantos, referencia)["tir_real"]
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
        ("quebrantos_utilizados_usd", "quebrantos_utilizados_usd"),
        ("impuesto_ganancias_usd", "impuesto_ganancias_usd"),
        ("flujo_neto_usd", "flujo_evaluacion_usd"),
    ],
)
def test_flujo_anual_coincide_con_excel(columna_motor, columna_excel, flujo_referencia):
    """Año por año (1-20), sin quebrantos, para ubicar en qué rubro está una
    diferencia si el VAN deja de coincidir. Tolerancia relativa al mayor
    valor de la serie (o absoluta de 1 USD si la serie es toda cero, como
    los quebrantos con el interruptor apagado)."""
    df, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    motor = df.set_index("año")[columna_motor]
    excel = flujo_referencia.loc[1:, columna_excel]
    escala = max(excel.abs().max(), 1.0)
    diferencia = (motor - excel).abs().max()
    assert diferencia / escala < TOLERANCIA_RELATIVA, (
        f"{columna_motor}: máxima diferencia {diferencia:,.2f} USD"
    )


def test_quebrantos_utilizados_por_año_con_interruptor_encendido():
    """Fila Flujo!32 con Supuestos!B64=1: quebrantos en los años 8, 9 y 10
    y cero en el resto. Las pérdidas de los años 1-2 vencen sin usarse, y
    también ~21k de la del año 3 (vence en el año 8, que solo tiene 472k de
    ganancia); las de los años 4-7 se consumen en los años 9 y 10."""
    df, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1, quebrantos=True)
    motor = df.set_index("año")["quebrantos_utilizados_usd"]
    esperado = pd.Series(0.0, index=motor.index)
    for año, valor in REFERENCIA_CON_QUEBRANTOS["quebrantos_utilizados_usd"].items():
        esperado[año] = valor
    for año in motor.index:
        if esperado[año] == 0:
            assert motor[año] == pytest.approx(0.0, abs=1.0), f"año {año}: {motor[año]:,.2f}"
        else:
            assert motor[año] == pytest.approx(esperado[año], rel=TOLERANCIA_RELATIVA), (
                f"año {año}: motor {motor[año]:,.2f} vs Excel {esperado[año]:,.2f}"
            )


def test_con_quebrantos_ganancias_es_sobre_ebit_menos_quebrantos():
    """Flujo!21 = MAX(0, (EBIT - quebrantos) * alícuota): con el interruptor
    encendido, el impuesto de cada año nunca supera al del motor sin
    quebrantos y la diferencia es exactamente quebrantos * alícuota donde
    la base queda positiva."""
    sin, _, costos = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    con, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1, quebrantos=True)
    imp_sin = sin["impuesto_ganancias_usd"].to_numpy()
    imp_con = con["impuesto_ganancias_usd"].to_numpy()
    usados = con["quebrantos_utilizados_usd"].to_numpy()
    assert np.all(imp_con <= imp_sin + 1e-6)
    base_positiva = imp_con > 0
    np.testing.assert_allclose(
        (imp_sin - imp_con)[base_positiva],
        usados[base_positiva] * costos.alicuota_ganancias,
        rtol=1e-9, atol=1e-6,
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
