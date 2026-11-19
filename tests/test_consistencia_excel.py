"""Consistencia del motor con el modelo financiero v2 (Excel).

La corrida determinística del motor (`ParametrosMC(deterministico=True)`:
curva y precios del Excel, sin clima, vecería simulada ni ruido de OPEX) a
120 ha tiene que reproducir el VAN y la TIR real de la hoja Flujo del Excel.

Caso base del Excel (como está guardado, desde 2026-11-19): la finca vende
SECO con cáscara (Supuestos B78 = 0). El rendimiento (3.000 kg/ha) ya está
en kg secos (conversión B71 = 1,00), los precios del bloque 6 ya son al
productor (B72 = 0, proporción apagada; planos 5 / 7 / 9 USD/kg seco en los
años 7-20) y la finca paga la maquila de pelado y secado (0,40 USD/kg
seco). Quebrantos sí, activación no. VAN base -1.487.137,36, pesimista
-2.512.883,25 y optimista -468.952,76 (tolerancia relativa 1e-6). La
referencia del base sale de `data/modelo_v2/referencia_indicadores.csv` y
`referencia_flujo.csv` (exportados por `scripts/importar_modelo_v2.py`).

Alternativa: venta en verde a la procesadora (B78 = 1): ingreso = kg
verdes x precio en verde (planos 2,5 / 4 / 6 USD/kg verde), sin conversión
ni maquila. VAN base -2.836.458,08, pesimista -3.713.633,54 y optimista
-1.788.373,93.

Plan original: venta_verde = 0, los dos interruptores del bloque 8
apagados y la
trayectoria de precio creciente del plan original (Supuestos fila 58,
escenario "plan_original"). Ahí se verifican las cuatro combinaciones de
los dos interruptores fiscales de Supuestos, bloque 7:

- B65, traslado de quebrantos (`ParametrosCostos.quebrantos`), y
- B67, activación del OPEX pre-productivo como costo de implantación
  (`ParametrosCostos.activar_opex_pre`, vida útil B68 = 20 años).

con los valores fijados en `REFERENCIAS` (tomados del Excel recalculado con
cada posición de B65 y B67, antes del bloque 8). El caso base fiscal (VAN
-67.345,99, TIR 14,0464%) se verifica además con tolerancia estricta.

El último test verifica que los CSV sigan sincronizados con el .xlsx: si
alguien edita el Excel sin volver a correr el importador, falla.
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
    precio_equilibrio,
    resumen_financiero,
    run_monte_carlo,
    run_monte_carlo_antitetico,
    run_monte_carlo_precio_historico,
)

TOLERANCIA_RELATIVA = 0.01

_RUNNERS = [run_monte_carlo, run_monte_carlo_antitetico, run_monte_carlo_precio_historico]

# Caso base del Excel: venta en seco con cáscara (Flujo!B5 con
# Supuestos!B22 = 1/2/3). Si cambian los precios del bloque 6, la
# conversión (B71) o la maquila, hay que volver a tomarlos.
VAN_VENTA_SECO_USD = {
    "pesimista": -2_512_883.25,
    "base": -1_487_137.36,
    "optimista": -468_952.76,
}
# Alternativa: venta en verde (B78 = 1). Si cambian los precios en verde
# (bloque 9), hay que volver a tomarlos.
VAN_VENTA_VERDE_USD = {
    "pesimista": -3_713_633.54,
    "base": -2_836_458.08,
    "optimista": -1_788_373.93,
}
TOLERANCIA_ESTRICTA_RELATIVA = 1e-6

# PLAN ORIGINAL (bloque 8 apagado, trayectoria de precio creciente):
# (quebrantos, activar_opex_pre) -> Flujo!B5, Flujo!B6 y Flujo!32 (=
# Quebrantos!55; año -> quebranto utilizado, el resto de los años cero).
# 120 ha, plazo de quebrantos 5 años, vida útil 20 años.
REFERENCIAS = {
    (True, False): {  # caso base fiscal
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
    plan_original: bool = False,
    venta_verde: bool = False,
    escenario: str = "base",
):
    """Default: caso base del Excel (venta en seco con cáscara, bloque 8
    tal como está guardado). `venta_verde=True`: alternativa en verde
    (bloque 9). `plan_original=True`: venta_verde = 0, bloque 8 apagado y
    trayectoria de precio vieja."""
    if plan_original:
        venta_verde, escenario = False, "plan_original"
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=120.0,
        escenario=escenario,
        deterministico=True,
    )
    costos = ParametrosCostos(
        hectareas=120.0, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre,
        venta_verde=venta_verde,
        **({"convertir_a_seco": False, "precio_al_productor": False} if plan_original else {}),
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


def test_base_de_ingresos_coincide_con_excel():
    """Caso base seco: el Excel está guardado con venta en seco (B78 = 0),
    conversión 1,00 (B71) y precio al productor apagado (B72 = 0), y los
    defaults del motor son los del Excel."""
    supuestos = cargar_modelo_v2().supuestos
    assert supuestos["venta_verde"] == 0
    assert supuestos["factor_conversion_seco"] == 1.0
    assert supuestos["precio_al_productor"] == 0
    defaults = ParametrosCostos()
    assert defaults.venta_verde is False
    assert defaults.precio_al_productor is False
    assert defaults.factor_conversion_seco == supuestos["factor_conversion_seco"]
    assert defaults.proporcion_precio_productor == supuestos["proporcion_precio_productor"]
    assert defaults.factor_kg_vendidos == 1.0
    assert defaults.factor_precio_recibido == 1.0
    assert defaults.costo_por_kg == pytest.approx(0.4)


def test_interruptores_apagados_reproducen_plan_original():
    apagados = ParametrosCostos(
        venta_verde=False, convertir_a_seco=False, precio_al_productor=False
    )
    assert apagados.factor_kg_vendidos == 1.0
    assert apagados.factor_precio_recibido == 1.0


def test_venta_verde_sin_conversion_ni_maquila():
    """Con venta en verde no se aplican conversión, proporción ni maquila,
    aunque los interruptores del bloque 8 estén encendidos."""
    verde = ParametrosCostos(venta_verde=True)
    assert verde.factor_kg_vendidos == 1.0
    assert verde.factor_precio_recibido == 1.0
    assert verde.costo_por_kg == 0.0
    # Bloque 8 con los interruptores encendidos y valores pisados: se aplican.
    seco = ParametrosCostos(
        venta_verde=False, factor_conversion_seco=0.5, precio_al_productor=True,
        proporcion_precio_productor=0.5,
    )
    assert seco.factor_kg_vendidos == 0.5
    assert seco.factor_precio_recibido == 0.5
    assert seco.costo_por_kg == pytest.approx(0.4)


def test_caso_base_fijado_coincide_con_csv(referencia):
    """La referencia fijada del caso base es la que guarda el .xlsx."""
    assert referencia["van_usd"] == pytest.approx(
        VAN_VENTA_SECO_USD["base"], rel=TOLERANCIA_ESTRICTA_RELATIVA
    )


@pytest.mark.parametrize("escenario", list(VAN_VENTA_SECO_USD))
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_venta_seco_coincide_con_excel(runner, escenario):
    """Caso base (venta en seco con cáscara) en los tres escenarios: VAN
    del Excel con tolerancia relativa 1e-6."""
    _, resumen, _ = _corrida_deterministica(runner, escenario=escenario)
    for van in resumen["van_neto_usd"]:
        assert van == pytest.approx(
            VAN_VENTA_SECO_USD[escenario], rel=TOLERANCIA_ESTRICTA_RELATIVA
        )


@pytest.mark.parametrize("escenario", list(VAN_VENTA_VERDE_USD))
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_venta_verde_coincide_con_excel(runner, escenario):
    """Alternativa (venta en verde) en los tres escenarios: VAN del Excel
    con tolerancia relativa 1e-6."""
    _, resumen, _ = _corrida_deterministica(runner, escenario=escenario, venta_verde=True)
    for van in resumen["van_neto_usd"]:
        assert van == pytest.approx(
            VAN_VENTA_VERDE_USD[escenario], rel=TOLERANCIA_ESTRICTA_RELATIVA
        )


@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_tir_caso_base_coincide_con_excel(runner, referencia):
    _, resumen, _ = _corrida_deterministica(runner)
    for tir in resumen["tir"]:
        assert tir == pytest.approx(referencia["tir_real"], abs=1e-6)


@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_plan_original_caso_base_fiscal_estricto(runner):
    """venta_verde = 0, interruptores del bloque 8 apagados y trayectoria
    vieja: VAN -67.345,99 y TIR 14,0464% (caso base fiscal antes del
    bloque 8)."""
    _, resumen, _ = _corrida_deterministica(runner, plan_original=True)
    for van, tir in zip(resumen["van_neto_usd"], resumen["tir"]):
        assert van == pytest.approx(-67_345.99, abs=0.01)
        assert tir == pytest.approx(0.140464, abs=1e-6)


def test_precio_equilibrio_coincide_con_excel(referencia):
    """Flujo!B11: OPEX del año 12 / kg vendidos del año 12 (kg secos con
    cáscara en el caso base), en USD por kg vendido."""
    df, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    assert precio_equilibrio(df, 12).iloc[0] == pytest.approx(
        referencia["precio_equilibrio_anio12_usd_kg_vendido"], rel=TOLERANCIA_ESTRICTA_RELATIVA
    )


def test_kilos_y_precio_coinciden_con_produccion_excel():
    """Año 12 del Excel (hoja Producción, 324.000 kg). Caso base seco: se
    venden los 324.000 kg secos con cáscara a 7 USD/kg al productor,
    ingresos 2.268.000. Alternativa en verde: los mismos 324.000 kg leídos
    como verdes, a 4 USD/kg verde, ingresos 1.296.000."""
    seco, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    año12 = seco.set_index("año").loc[12]
    assert año12["rendimiento_kg_ha"] * 120 == pytest.approx(324_000)
    assert año12["produccion_kg_vendidos"] == pytest.approx(324_000)
    assert año12["precio_usd_kg"] == pytest.approx(7.0)
    assert año12["ingreso_usd"] == pytest.approx(2_268_000)

    verde, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1, venta_verde=True)
    año12 = verde.set_index("año").loc[12]
    assert año12["produccion_kg_vendidos"] == pytest.approx(324_000)
    assert año12["precio_usd_kg"] == pytest.approx(4.0)
    assert año12["ingreso_usd"] == pytest.approx(1_296_000)


def test_maquila_cero_en_venta_verde():
    """OPEX!49 x IF(venta_verde=1, 0, 1): el OPEX con venta en verde es el
    de venta en seco menos la maquila (0,40 x kg secos) y menos el IIBB
    sobre la diferencia de ventas."""
    verde, _, costos = _corrida_deterministica(run_monte_carlo, n_simulaciones=1, venta_verde=True)
    seco, _, _ = _corrida_deterministica(run_monte_carlo, n_simulaciones=1)
    verde, seco = verde.set_index("año"), seco.set_index("año")
    esperado = (
        seco["opex_usd"]
        - 0.4 * seco["produccion_kg_vendidos"]
        + costos.alicuota_iibb * (verde["ingreso_usd"] - seco["ingreso_usd"])
    )
    np.testing.assert_allclose(verde["opex_usd"], esperado, rtol=1e-12, atol=1e-6)


def test_precio_params_incoherente_con_costos_falla():
    """Pasar un AR(1) con la otra base de precio que `costos` es un error,
    no una mezcla silenciosa."""
    from src.precio_estocastico import ParametrosPrecioAR1

    with pytest.raises(ValueError):
        run_monte_carlo(
            ParametrosMC(n_simulaciones=4),
            ParametrosCostos(venta_verde=True),
            ParametrosPrecioAR1(venta_verde=False),
        )


@_COMBINACIONES
@pytest.mark.parametrize("runner", _RUNNERS, ids=lambda f: f.__name__)
def test_van_coincide_con_excel(runner, quebrantos, activar_opex_pre):
    _, resumen, _ = _corrida_deterministica(
        runner, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre, plan_original=True
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
        runner, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre, plan_original=True
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
    """Año por año (1-20), en el caso base del Excel actual (base seca y
    precio al productor), para ubicar en qué rubro está
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
    """Fila Flujo!32 en cada combinación, con el plan original. En el caso
    base fiscal, quebrantos en los
    años 8, 9 y 10: las pérdidas de los años 1-2 vencen sin usarse, y
    también ~21k de la del año 3 (vence en el año 8, que solo tiene 472k de
    base positiva); las de los años 4-7 se consumen en los años 9 y 10. Con
    la activación, las pérdidas fiscales de los años 1-6 son mucho menores
    (el OPEX no se deduce) y solo se usan en los años 8 y 9."""
    df, _, _ = _corrida_deterministica(
        run_monte_carlo, n_simulaciones=1,
        quebrantos=quebrantos, activar_opex_pre=activar_opex_pre, plan_original=True,
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
