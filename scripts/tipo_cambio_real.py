"""Sensibilidad del caso base al tipo de cambio real (corrida determinística).

El motor trabaja todo en USD y NO modela el tipo de cambio real: la columna
`moneda_pago` de data/modelo_v2/{capex,opex}.csv es informativa y ningún
cálculo la usa. El ruido de OPEX (`simulate_opex_multiplicador`, triangular
±15 %, un sorteo por simulación) escala todo el OPEX operativo por igual,
sin distinguir moneda: no es un shock cambiario.

Este script:
1. Mide qué proporción del CAPEX y del OPEX se paga en pesos (ARS).
   - CAPEX: sobre el CAPEX del año 0.
   - OPEX: sobre el OPEX operativo de los 20 años (sin IIBB, que es una
     alícuota sobre ventas en USD), nominal y a valor presente.
   - Personal (sueldos, hoja Personal) cuenta como ARS.
2. Recalcula el VAN determinístico del caso base (venta en verde, 4 USD/kg,
   120 ha) con los costos en pesos multiplicados por 0,8 (devaluación real:
   los costos en pesos bajan en USD) y 1,2 (atraso cambiario real: suben).
   El shock es permanente (todos los años) y afecta CAPEX y OPEX en ARS.
   No toca: ítems en USD, IIBB, valor residual de la tierra (en USD).

Uso:
    python scripts/tipo_cambio_real.py

Salida: outputs/tipo_cambio_real.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    ParametrosMC,
    _componentes_deterministicos,
    _orquestar_resultado,
    _tir_vectorizada,
)

SALIDA = RAIZ / "outputs" / "tipo_cambio_real.csv"
FACTORES_ARS = {"devaluacion_real_-20%": 0.8, "base": 1.0, "atraso_cambiario_+20%": 1.2}


def _modelo_con_ars(modelo, factor: float):
    """ModeloV2 con los precios en pesos (expresados en USD) × `factor`."""
    capex = modelo.capex.copy()
    opex = modelo.opex.copy()
    for df in (capex, opex):
        ars = df["moneda_pago"] == "ARS"
        df.loc[ars, "precio_unit_usd"] = df.loc[ars, "precio_unit_usd"] * factor
    supuestos = dict(modelo.supuestos)
    for k in ("salario_operario_usd_mes", "salario_encargado_usd_mes"):
        supuestos[k] = supuestos[k] * factor
    return replace(modelo, capex=capex, opex=opex, supuestos=supuestos)


def corrida(factor_ars: float = 1.0) -> dict:
    params = ParametrosMC(n_simulaciones=1, deterministico=True)
    base = ParametrosCostos(hectareas=params.hectareas)
    costos = replace(base, modelo=_modelo_con_ars(base.modelo, factor_ars))
    yields, prices, capex_extra, opex_mult = _componentes_deterministicos(params, costos)
    df = _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)
    flujos = np.concatenate([[-costos.capex_inicial], df["flujo_neto_usd"].to_numpy()])
    return {
        "van_usd": float(df["van_neto_usd"].iloc[-1]),
        "tir_real": float(_tir_vectorizada(flujos.reshape(1, -1))[0]),
        "capex_inicial_usd": costos.capex_inicial,
        "opex_total_20_anios_usd": float(df["opex_usd"].sum()),
    }


def proporciones_ars() -> dict:
    params = ParametrosMC(n_simulaciones=1, deterministico=True)
    costos = ParametrosCostos(hectareas=params.hectareas)
    T = params.n_años

    items = costos.capex_items
    capex_ars = items.loc[items["moneda_pago"] == "ARS", "total_escalado_usd"].sum()

    # OPEX operativo por moneda: solo con ARS (factor 0 en USD) vs todo.
    def opex_fijo(con_usd: bool, con_ars: bool) -> np.ndarray:
        m = costos.modelo
        opex = m.opex.copy()
        opex.loc[opex["moneda_pago"] == "USD", "precio_unit_usd"] *= float(con_usd)
        opex.loc[opex["moneda_pago"] == "ARS", "precio_unit_usd"] *= float(con_ars)
        sup = dict(m.supuestos)
        for k in ("salario_operario_usd_mes", "salario_encargado_usd_mes"):
            sup[k] *= float(con_ars)
        c = replace(costos, modelo=replace(m, opex=opex, supuestos=sup))
        return c.opex_fijo(T)  # en verde no hay maquila; IIBB va aparte

    total, ars = opex_fijo(True, True), opex_fijo(False, True)
    personal = costos.costo_personal(T)
    desc = 1 / (1 + params.tasa_descuento) ** np.arange(1, T + 1)
    return {
        "capex_ars_pct": capex_ars / costos.capex_inicial,
        "opex_ars_pct_nominal": ars.sum() / total.sum(),
        "opex_ars_pct_vp": (ars * desc).sum() / (total * desc).sum(),
        "opex_personal_pct_nominal": personal.sum() / total.sum(),
        "opex_ars_sin_personal_pct_nominal": (ars - personal).sum() / total.sum(),
    }


def main() -> pd.DataFrame:
    filas = [{"caso": k, "factor_costos_ars": f, **corrida(f)} for k, f in FACTORES_ARS.items()]
    df = pd.DataFrame(filas)
    van_base = df.loc[df["caso"] == "base", "van_usd"].iloc[0]
    df["delta_van_vs_base_usd"] = df["van_usd"] - van_base
    prop = proporciones_ars()
    for k, v in prop.items():
        df[k] = v
    return df


if __name__ == "__main__":
    df = main()
    SALIDA.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA, index=False)
    pd.set_option("display.width", 200)
    print(df.T.to_string())
    print(f"\nGuardado en {SALIDA}")
