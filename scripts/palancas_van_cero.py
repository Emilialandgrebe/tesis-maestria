"""Palancas para llevar el VAN a cero en el caso base (corrida determinística).

Caso base: venta en verde a la procesadora (Supuestos bloque 9), 4 USD/kg
verde, 3.000 kg verdes/ha a plena producción, 120 ha, tasa real del Excel
(Supuestos!B49). Es la misma corrida que `ParametrosMC(deterministico=True)`
(curva del Excel, trayectoria de precio plana, sin ruido), con tres palancas:

- Reducción de CAPEX: `ParametrosCostos.capex_factor = 1 - r`. Escala todo
  el CAPEX del año 0 y sus amortizaciones (escudo fiscal), no el valor
  residual de la tierra (se recupera a su valor de compra).
- Reducción de OPEX: multiplicador `1 - r` sobre el OPEX operativo (fijo +
  maquila, que es 0 en verde). IIBB queda como % de las ventas.
- Tasa para VAN = 0: la TIR real del flujo determinístico.

Además, una grilla reducción de CAPEX (0, 20, 40, 50 %) × precio en verde
(4, 5, 6, 7 USD/kg verde) con el VAN determinístico. El precio escala la
trayectoria en verde completa (`precio / 4`).

Uso:
    python scripts/palancas_van_cero.py

Salida: outputs/palancas_van_cero.csv (formato largo: tipo, parámetros, VAN).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

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

SALIDA = RAIZ / "outputs" / "palancas_van_cero.csv"
REDUCCIONES_CAPEX = [0.0, 0.20, 0.40, 0.50]
PRECIOS_VERDE = [4.0, 5.0, 6.0, 7.0]


def _corrida(
    reduccion_capex: float = 0.0,
    reduccion_opex: float = 0.0,
    precio_verde: float | None = None,
    tasa: float | None = None,
) -> tuple[pd.DataFrame, ParametrosCostos, ParametrosMC]:
    extra = {} if tasa is None else {"tasa_descuento": tasa}
    params = ParametrosMC(n_simulaciones=1, deterministico=True, **extra)
    costos = ParametrosCostos(hectareas=params.hectareas, capex_factor=1.0 - reduccion_capex)
    assert costos.venta_verde, "el caso base es venta en verde"
    yields, prices, capex_extra, _ = _componentes_deterministicos(params, costos)
    if precio_verde is not None:
        precio_base = prices[0][prices[0] > 0][0]
        prices = prices * (precio_verde / precio_base)
    opex_mult = np.full(1, 1.0 - reduccion_opex)
    df = _orquestar_resultado(yields, prices, params, costos, capex_extra, opex_mult)
    return df, costos, params


def van(**kw) -> float:
    df, _, _ = _corrida(**kw)
    return float(df["van_neto_usd"].iloc[-1])


def tir(**kw) -> float:
    df, costos, _ = _corrida(**kw)
    flujos = np.concatenate([[-costos.capex_inicial], df["flujo_neto_usd"].to_numpy()])
    return float(_tir_vectorizada(flujos.reshape(1, -1))[0])


def _raiz(f, a: float, b: float) -> float:
    """Raíz de f en [a, b]; NaN si no cambia de signo (la palanca no alcanza)."""
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):
        return float("nan")
    return float(brentq(f, a, b, xtol=1e-6))


def contexto() -> dict:
    df, costos, params = _corrida()
    ha = params.hectareas
    ultimo_pre = int(costos.modelo.supuestos["ultimo_anio_preproductivo"])
    opex_pre = float(df.loc[df["año"] <= ultimo_pre, "opex_usd"].sum())
    flujos = np.concatenate([[-costos.capex_inicial], df["flujo_neto_usd"].to_numpy()])
    exposicion_max = float(-np.cumsum(flujos).min())
    return {
        "hectareas": ha,
        "tasa_real_excel": params.tasa_descuento,
        "capex_inicial_usd": costos.capex_inicial,
        "capex_usd_ha": costos.capex_inicial / ha,
        "opex_preproductivo_usd_ha": opex_pre / ha,
        "capex_mas_opex_pre_usd_ha": (costos.capex_inicial + opex_pre) / ha,
        "exposicion_maxima_caja_usd_ha": exposicion_max / ha,
    }


def palancas() -> pd.DataFrame:
    filas = []
    ctx = contexto()
    van_base = van()
    filas.append({"tipo": "van_base", "valor": van_base})

    r_capex = _raiz(lambda r: van(reduccion_capex=r), 0.0, 0.99)
    r_opex = _raiz(lambda r: van(reduccion_opex=r), 0.0, 1.0)
    tir_base = tir()
    filas += [
        {"tipo": "reduccion_capex_van0", "valor": r_capex},
        {"tipo": "reduccion_opex_van0", "valor": r_opex},
        {"tipo": "tir_real", "valor": tir_base},
    ]
    # Chequeo: el VAN a la TIR debe ser ~0.
    filas.append({"tipo": "chequeo_van_a_la_tir", "valor": van(tasa=tir_base)})
    # Cuando una palanca sola no alcanza (NaN arriba): VAN en su extremo.
    filas += [
        {"tipo": "van_con_capex_menos_99pct", "valor": van(reduccion_capex=0.99)},
        {"tipo": "van_con_opex_menos_100pct", "valor": van(reduccion_opex=1.0)},
    ]
    # Precio en verde que da VAN = 0, para cada reducción de CAPEX de la grilla.
    for r in REDUCCIONES_CAPEX:
        filas.append({
            "tipo": "precio_verde_van0",
            "reduccion_capex": r,
            "valor": _raiz(lambda p: van(reduccion_capex=r, precio_verde=p), 1.0, 30.0),
        })

    for k, v in ctx.items():
        filas.append({"tipo": f"contexto_{k}", "valor": v})

    for r in REDUCCIONES_CAPEX:
        for p in PRECIOS_VERDE:
            filas.append({
                "tipo": "grilla_capex_precio",
                "reduccion_capex": r,
                "precio_verde_usd_kg": p,
                "valor": van(reduccion_capex=r, precio_verde=p),
                "tir_real": tir(reduccion_capex=r, precio_verde=p),
            })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = palancas()
    SALIDA.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA, index=False)

    pd.set_option("display.width", 200)
    print(df[df["tipo"] != "grilla_capex_precio"][["tipo", "valor"]].to_string(index=False))
    g = df[df["tipo"] == "grilla_capex_precio"]
    print("\nVAN (USD), filas: reducción de CAPEX; columnas: USD/kg verde")
    print(g.pivot(index="reduccion_capex", columns="precio_verde_usd_kg", values="valor")
          .map("{:,.0f}".format).to_string())
    print("\nTIR real")
    print(g.pivot(index="reduccion_capex", columns="precio_verde_usd_kg", values="tir_real")
          .map("{:.1%}".format).to_string())
    print(f"\nGuardado en {SALIDA}")
