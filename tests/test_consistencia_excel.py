"""Consistencia del motor con el modelo financiero v2 (Excel).

La corrida determinística del motor (`ParametrosMC(deterministico=True)`:
curva y precios del Excel, sin clima, vecería simulada ni ruido de OPEX) en
el escenario base a 120 ha tiene que reproducir el VAN y la TIR real de la
hoja Flujo del Excel con una diferencia relativa menor al 1%, en las cuatro
combinaciones de los dos interruptores fiscales de Supuestos, bloque 7:

- B64, traslado de quebrantos (`ParametrosCostos.quebrantos`), y
- B66, activación del OPEX pre-productivo como costo de implantación
  (`ParametrosCostos.activar_opex_pre`, vida útil B67 = 20 años).

El Excel está guardado en el caso base (B64 = 1, B66 = 0), que es también
el default del motor. Para ese caso la referencia sale de
`data/modelo_v2/referencia_indicadores.csv` y `referencia_flujo.csv`
(exportados por `scripts/importar_modelo_v2.py`). El último test verifica
que esos CSV sigan sincronizados con el .xlsx: si alguien edita el Excel sin
volver a correr el importador, falla.

El .xlsx guarda solo los valores calculados en el caso base, así que las
otras tres combinaciones están fijadas abajo (`REFERENCIAS`), tomadas del
Excel recalculado con cada posición de B64 y B66. Si cambia el Excel, hay
que volver a tomarlas.
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

# (quebrantos, activar_opex_pre) -> Flujo!B5, Flujo!B6 y Flujo!32 (=
# Quebrantos!55; año -> quebranto utilizado, el resto de los años cero).
# Escenario base, 120 ha, plazo de quebrantos 5 años, vida útil 20 años.
REFERENCIAS = {
    (True, False): {  # caso base del Excel
        "van_usd": -67_345.99, "tir_real": 0.140464,
        "quebrantos_utilizados_usd": {8: 471_852.0, 9: 1_524_451.0, 10: 267_989.0},
    },
    (False, False): {
        "van_usd": -240_830.30, "tir_real": 0.135699,
        "quebrantos_utilizados_usd": {},
    },
    (False, True): {
        "van_usd": -176_116.21, "tir_real": 0.137505,
        "quebrantos_utilizados_usd": {},
    },
    (True, True): {
        "van_usd": -117_716.31, "tir_real": 0.139099,
        "quebrantos_utilizados_usd": {8: 358_209.0, 9: 364_408.0},
    },
}

# Con la activación encendida (Flujo!33 y 34): OPEX de los años 1-6
# activado (total 2.272.848,83) y amortizado en 20 años desde el año 7 ->
# cuota de 113.642,44 en los años 7-20 (el saldo de los años 21-26 queda
# fuera del horizonte).
OPEX_ACTIVADO_TOTAL_USD = 2_272_848.83
CUOTA_AMORTIZACION_IMPLANTACION_USD = 113_642.44

_IDS = [
    f"{'con' if q else 'sin'}_quebrantos-{'con' if a else 'sin'}_activacion"
    for q, a in REFERENCIAS
]
_COMBINACIONES = pytest.mark.parametrize(
    "quebrantos, activar_opex_pre", list(REFERENCIAS), ids=_IDS
)


@pytest.fixture(scope="module")
def referencia() -> dict[str, float]:
    df = pd.read_csv(DIR_MODELO_V2 / "referencia_indicadores.csv")
    return dict(zip(df["indicador"], df["valor"]))


@pytest.fixture(scope="module")
def flujo_referencia() -> pd.DataFrame:
    return pd.read_csv(DIR_MODELO_V2 / "referencia_flujo.csv").set_index("anio")


def _corrida_deterministica(
    runner,
    n_simulaciones: int = 2,
    quebrantos: bool = True,
    activar_opex_pre: bool = False,
):
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=120.0,
        escenario="base",
        deterministico=True,
    )
    costos = ParametrosCostos(
        hectareas=120.0, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre
    )
    df = runner(params, costos)
    return df, resumen_financiero(df, costos), costos


def test_referencia_es_escenario_base_a_tasa_del_motor(referencia):
    """El Excel tiene que estar guardado en el escenario base (Supuestos!B22=2)
    y la tasa del Flujo tiene que ser la tasa por defecto del motor."""
    assert referencia["escenario_precio"] == 2
    assert ParametrosMC().tasa_descuento == pytest.approx(referencia["tasa_descuento_real"], rel=1e-12)


def test_interruptores_fiscales_coinciden_con_excel():
    """El Excel está guardado en el caso base (quebrantos sí, activación no),
    que es también el default del motor; los plazos por defecto del motor
    son los de Supuestos!B65 y B67."""
    supuestos = cargar_modelo_v2().supuestos
    assert supuestos["quebrantos_aplicar"] == 1
    assert supuestos["activar_opex_pre"] == 0
    defaults = ParametrosCostos()
    assert defaults.quebrantos is True
    assert defaults.activar_opex_pre is False
    assert defaults.quebrantos_anios == supuestos["quebrantos_anios"] == 5
    assert defaults.vida_util_implantacion == supuestos["vida_util_implantacion"] == 20


def test_caso_base_fijado_coincide_con_csv(referencia):
    """La referencia fijada del caso base es la que guarda el .xlsx."""
    base = REFERENCIAS[(True, False)]
    assert referencia["van_usd"] == pytest.approx(base["van_usd"], abs=0.01)
    assert referencia["tir_real"] == pytest.approx(base["tir_real"], abs=1e-6)


@_COMBINACIONES
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_coincide_con_excel(runner, quebrantos, activar_opex_pre):
    _, resumen, _ = _corrida_deterministica(
        runner, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre
    )
    van_excel = REFERENCIAS[(quebrantos, activar_opex_pre)]["van_usd"]
    for van in resumen["van_neto_usd"]:
        assert abs(van - van_excel) / abs(van_excel) < TOLERANCIA_RELATIVA, (
            f"VAN motor {van:,.2f} vs Excel {van_excel:,.2f}"
        )


@_COMBINACIONES
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_tir_coincide_con_excel(runner, quebrantos, activar_opex_pre):
    _, resumen, _ = _corrida_deterministica(
        runner, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre
    )
    tir_excel = REFERENCIAS[(quebrantos, activar_opex_pre)]["tir_real"]
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
        ("opex_pre_activado_usd", "opex_pre_activado_usd"),
        ("amortizacion_implantacion_usd", "amortizacion_implantacion_usd"),
        ("base_imponible_usd", "base_imponible_usd"),
        ("quebrantos_utilizados_usd", "quebrantos_utilizados_usd"),
        ("impuesto_ganancias_usd", "impuesto_ganancias_usd"),
        ("flujo_neto_usd", "flujo_evaluacion_usd"),
    ],
)
def test_flujo_anual_coincide_con_excel(columna_motor, columna_excel, flujo_referencia):
    """Año por año (1-20), en el caso base, para ubicar en qué rubro está
    una diferencia si el VAN deja de coincidir. Tolerancia relativa al mayor
    valor de la serie (o absoluta de 1 USD si la serie es toda cero, como
    las filas de activación con el interruptor apagado)."""
    df, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    motor = df.set_index("año")[columna_motor]
    excel = flujo_referencia.loc[1:, columna_excel]
    escala = max(excel.abs().max(), 1.0)
    diferencia = (motor - excel).abs().max()
    assert diferencia / escala < TOLERANCIA_RELATIVA, (
        f"{columna_motor}: máxima diferencia {diferencia:,.2f} USD"
    )


@_COMBINACIONES
def test_quebrantos_utilizados_por_año(quebrantos, activar_opex_pre):
    """Fila Flujo!32 en cada combinación. En el caso base, quebrantos en los
    años 8, 9 y 10: las pérdidas de los años 1-2 vencen sin usarse, y
    también ~21k de la del año 3 (vence en el año 8, que solo tiene 472k de
    base positiva); las de los años 4-7 se consumen en los años 9 y 10. Con
    la activación, las pérdidas fiscales de los años 1-6 son mucho menores
    (el OPEX no se deduce) y solo se usan en los años 8 y 9."""
    df, _, _ = _corrida_deterministica(
        run_monte_carlo, n_simulaciones=1,
        quebrantos=quebrantos, activar_opex_pre=activar_opex_pre,
    )
    motor = df.set_index("año")["quebrantos_utilizados_usd"]
    esperado = pd.Series(0.0, index=motor.index)
    for año, valor in REFERENCIAS[(quebrantos, activar_opex_pre)]["quebrantos_utilizados_usd"].items():
        esperado[año] = valor
    for año in motor.index:
        if esperado[año] == 0:
            assert motor[año] == pytest.approx(0.0, abs=1.0), f"año {año}: {motor[año]:,.2f}"
        else:
            assert motor[año] == pytest.approx(esperado[año], rel=TOLERANCIA_RELATIVA), (
                f"año {año}: motor {motor[año]:,.2f} vs Excel {esperado[año]:,.2f}"
            )


@pytest.mark.parametrize("quebrantos", [False, True], ids=["sin_quebrantos", "con_quebrantos"])
def test_activacion_opex_pre_filas_33_y_34(quebrantos):
    """Flujo!33 = OPEX de los años 1-6; Flujo!34 = total activado / 20 en
    los años 7-20; Flujo!35 = EBIT + 33 - 34. La caja solo cambia vía
    Ganancias: ingresos, OPEX y EBITDA son idénticos con y sin activación."""
    sin, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1, quebrantos=quebrantos)
    con, _, costos = _corrida_deterministica(
        run_monte_carlo, n_simulaciones=1, quebrantos=quebrantos, activar_opex_pre=True
    )
    sin, con = sin.set_index("año"), con.set_index("año")
    pre = con.index <= 6

    np.testing.assert_allclose(con.loc[pre, "opex_pre_activado_usd"], con.loc[pre, "opex_usd"])
    assert (con.loc[~pre, "opex_pre_activado_usd"] == 0).all()
    assert con["opex_pre_activado_usd"].sum() == pytest.approx(OPEX_ACTIVADO_TOTAL_USD, abs=0.01)

    assert (con.loc[pre, "amortizacion_implantacion_usd"] == 0).all()
    np.testing.assert_allclose(
        con.loc[~pre, "amortizacion_implantacion_usd"], CUOTA_AMORTIZACION_IMPLANTACION_USD, atol=0.01
    )

    ebit = con["ebitda_usd"] - costos.amortizaciones(20)
    np.testing.assert_allclose(
        con["base_imponible_usd"],
        ebit + con["opex_pre_activado_usd"] - con["amortizacion_implantacion_usd"],
    )
    for col in ("ingreso_usd", "opex_usd", "ebitda_usd"):
        pd.testing.assert_series_equal(sin[col], con[col])
    np.testing.assert_allclose(
        con["flujo_neto_usd"] - sin["flujo_neto_usd"],
        sin["impuesto_ganancias_usd"] - con["impuesto_ganancias_usd"],
        atol=1e-6,
    )


def test_sin_activacion_base_imponible_es_ebit():
    df, _, costos = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    np.testing.assert_allclose(
        df["base_imponible_usd"], df["ebitda_usd"] - costos.amortizaciones(20)
    )


def test_con_quebrantos_ganancias_es_sobre_base_menos_quebrantos():
    """Flujo!21 = MAX(0, (base imponible - quebrantos) * alícuota): con el
    traslado encendido, el impuesto de cada año nunca supera al del motor
    sin quebrantos y la diferencia es exactamente quebrantos * alícuota
    donde la base queda positiva."""
    for activar in (False, True):
        sin, _, costos = _corrida_deterministica(
            run_monte_carlo, n_simulaciones=1, quebrantos=False, activar_opex_pre=activar
        )
        con, _, _ = _corrida_deterministica(
            run_monte_carlo, n_simulaciones=1, quebrantos=True, activar_opex_pre=activar
        )
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
