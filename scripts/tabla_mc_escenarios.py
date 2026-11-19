"""Tabla de Monte Carlo por escenario de precio y especificación del AR(1).

Para cada escenario (pesimista / base / optimista) y cada especificación del
precio (AR(1) en niveles, la del motor; AR(1) sobre retornos, estrés) corre
`run_monte_carlo` con los defaults del motor (caso base del Excel, 120 ha,
tasa real, OPEX estocástico, clima, vecería, quebrantos) y resume el VAN.

Columnas: VAN determinístico (`ParametrosMC(deterministico=True)`, el del
Excel), VAN medio con su IC95% (±1,96·desvío/√n), percentiles 5/10/50/90/95,
desvío, P(VAN<0), TIR mediana (sin las simulaciones sin TIR definida) y
cuántas no la tienen, VAN mediano, CVaR 10% (promedio del 10% peor) y precio
de equilibrio mediano del año 12 (USD por kg vendido).

Uso:
    python scripts/tabla_mc_escenarios.py [n_simulaciones]

Salida: outputs/tabla_mc_escenarios.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    ParametrosMC,
    precio_equilibrio,
    resumen_financiero,
    run_monte_carlo,
)
from src.precio_estocastico import precio_retornos_estres  # noqa: E402

SALIDA = RAIZ / "outputs" / "tabla_mc_escenarios.csv"
ESCENARIOS = ("pesimista", "base", "optimista")
SEMILLA = 42


def _van_deterministico(escenario: str) -> float:
    params = ParametrosMC(n_simulaciones=1, escenario=escenario, deterministico=True)
    costos = ParametrosCostos(hectareas=params.hectareas)
    return float(resumen_financiero(run_monte_carlo(params, costos), costos)["van_neto_usd"].iloc[0])


def fila(escenario: str, especificacion: str, n: int) -> dict:
    params = ParametrosMC(n_simulaciones=n, escenario=escenario, semilla=SEMILLA)
    costos = ParametrosCostos(hectareas=params.hectareas)
    precio_params = precio_retornos_estres(escenario) if especificacion == "retornos" else None
    df = run_monte_carlo(params, costos, precio_params)
    resumen = resumen_financiero(df, costos)
    van = resumen["van_neto_usd"].to_numpy()
    tir = resumen["tir"].to_numpy()
    p5, p10, p50, p90, p95 = np.percentile(van, [5, 10, 50, 90, 95])
    n_peores = max(1, int(np.ceil(0.10 * n)))
    return {
        "escenario": escenario,
        "precio": "niveles" if especificacion == "niveles" else "retornos (estrés)",
        "van_determinístico": _van_deterministico(escenario),
        "van_medio": van.mean(),
        "ic95": 1.96 * van.std(ddof=1) / np.sqrt(n),
        "p10": p10, "p50": p50, "p90": p90, "p5": p5, "p95": p95,
        "std": van.std(ddof=1),
        "p_van_neg": float((van < 0).mean()),
        "tir_mediana": float(np.nanmedian(tir)) if np.isfinite(tir).any() else float("nan"),
        "n_tir_nan": int(np.isnan(tir).sum()),
        "van_mediano": float(np.median(van)),
        "cvar_10": float(np.sort(van)[:n_peores].mean()),
        "precio_equilibrio_anio12_mediano_usd_kg_vendido": float(precio_equilibrio(df, 12).median()),
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    inicio = time.time()
    filas = []
    for escenario in ESCENARIOS:
        for especificacion in ("niveles", "retornos"):
            filas.append(fila(escenario, especificacion, n))
            print(f"{escenario}/{especificacion} ({time.time() - inicio:.0f}s)")
    tabla = pd.DataFrame(filas)
    SALIDA.parent.mkdir(exist_ok=True)
    tabla.to_csv(SALIDA, index=False)
    pd.set_option("display.width", 250)
    print(tabla.to_string(index=False))
    print(f"\nGuardado en {SALIDA}")
