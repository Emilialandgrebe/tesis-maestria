"""Prueba predictiva previa del precio bayesiano y sensibilidad del VAN a w.

1. Predictiva previa (solo precio, 20.000 sendas antitéticas): P5 / P50 / P95
   del precio en los años 7, 12 y 20 y la dispersión P95/P5 del año 20, para
   el precio bayesiano con w = 1 / 0,5 / 0 y, como comparación, el AR(1)
   actual en niveles y sobre retornos (escenario base, 7 USD/kg seco).
   Referencia histórica: P95/P5 y máx/mín del nivel real de FRED 1991-2026.
   Salida: outputs/precio_bayesiano_predictiva.csv

2. Sensibilidad a w (Monte Carlo completo, `run_monte_carlo_antitetico`,
   10.000 simulaciones, caso base seco, modo bayesiano) con w = 0; 0,25;
   0,5; 0,75; 1. Salida: outputs/precio_bayesiano_sensibilidad_w.csv

Uso:
    python scripts/precio_bayesiano_predictiva.py [n_mc]
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
from src.monte_carlo import ParametrosMC, resumen_financiero, run_monte_carlo_antitetico  # noqa: E402
from src.precio_bayesiano import (  # noqa: E402
    ParametrosPrecioBayes,
    cargar_posterior,
    simulate_prices_bayes_antitetico,
)
from src.precio_estocastico import (  # noqa: E402
    ParametrosPrecioAR1,
    precio_retornos_estres,
    simulate_prices_ar1_antitetico,
)

N_AÑOS = 20
N_PRECIO = 20_000
SEMILLA = 42
PESOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SALIDA_PRED = RAIZ / "outputs" / "precio_bayesiano_predictiva.csv"
SALIDA_W = RAIZ / "outputs" / "precio_bayesiano_sensibilidad_w.csv"


def _resumen_precios(nombre: str, precios: np.ndarray) -> dict:
    fila = {"modelo": nombre}
    for año in (7, 12, 20):
        p5, p50, p95 = np.percentile(precios[:, año - 1], [5, 50, 95])
        fila.update({f"p5_a{año}": p5, f"p50_a{año}": p50, f"p95_a{año}": p95})
    fila["media_a20"] = precios[:, 19].mean()
    fila["p95_p5_a20"] = fila["p95_a20"] / fila["p5_a20"]
    return fila


def predictiva() -> pd.DataFrame:
    filas = []
    for w, etiqueta in ((1.0, "pesimista"), (0.5, "base"), (0.0, "optimista")):
        precios = simulate_prices_bayes_antitetico(
            N_PRECIO, N_AÑOS, ParametrosPrecioBayes(peso_exportacion=w), np.random.default_rng(SEMILLA)
        )
        filas.append(_resumen_precios(f"bayesiano w={w:g} ({etiqueta})", precios))
    for nombre, pp in (
        ("AR(1) niveles, base 7", ParametrosPrecioAR1(escenario="base", venta_verde=False)),
        ("AR(1) retornos (estrés), base 7", precio_retornos_estres("base", venta_verde=False)),
    ):
        precios = simulate_prices_ar1_antitetico(N_PRECIO, N_AÑOS, pp, np.random.default_rng(SEMILLA))
        filas.append(_resumen_precios(nombre, precios))

    # Referencia histórica (índice real de FRED, 36 años).
    from scripts.calibrar_precio_bayesiano import serie_log_real

    nivel = np.exp(serie_log_real().to_numpy())
    p5, p95 = np.percentile(nivel, [5, 95])
    filas.append({"modelo": "histórico FRED real 1991-2026 (índice)",
                  "p95_p5_a20": p95 / p5, "max_min": nivel.max() / nivel.min()})
    post = cargar_posterior()
    df = pd.DataFrame(filas)
    df.attrs["p_phi_095"] = float((post["phi"] > 0.95).mean())
    return df


def sensibilidad_w(n: int) -> pd.DataFrame:
    filas = []
    inicio = time.time()
    for w in PESOS:
        # Sin el tipo de cambio estocástico: aísla el efecto de w (ver
        # scripts/tabla_mc_escenarios.py para la comparación con y sin).
        params = ParametrosMC(n_simulaciones=n, modo_precio="bayesiano", semilla=SEMILLA,
                              factor_ars_estocastico=False)
        costos = ParametrosCostos(hectareas=params.hectareas)
        df = run_monte_carlo_antitetico(params, costos, ParametrosPrecioBayes(peso_exportacion=w))
        r = resumen_financiero(df, costos)
        van = r["van_neto_usd"].to_numpy()
        tir = r["tir"].to_numpy()
        p12 = df.loc[df["año"] == 12, "precio_usd_kg"].to_numpy()
        n_peores = max(1, int(np.ceil(0.10 * n)))
        filas.append({
            "peso_exportacion_w": w,
            "nivel_esperado_usd_kg": 5 * w + 9 * (1 - w),
            "van_medio": van.mean(),
            "van_mediano": float(np.median(van)),
            "van_p5": float(np.percentile(van, 5)),
            "van_p95": float(np.percentile(van, 95)),
            "std": van.std(ddof=1),
            "p_van_neg": float((van < 0).mean()),
            "cvar_10": float(np.sort(van)[:n_peores].mean()),
            "tir_mediana": float(np.nanmedian(tir)),
            "precio_a12_p5": float(np.percentile(p12, 5)),
            "precio_a12_p50": float(np.median(p12)),
            "precio_a12_p95": float(np.percentile(p12, 95)),
        })
        print(f"w={w:g} ({time.time() - inicio:.0f}s)")
    return pd.DataFrame(filas)


if __name__ == "__main__":
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    pd.set_option("display.width", 250)
    pred = predictiva()
    SALIDA_PRED.parent.mkdir(exist_ok=True)
    pred.to_csv(SALIDA_PRED, index=False)
    print(pred.round(2).to_string(index=False))
    print(f"P(phi > 0,95) en la posterior: {pred.attrs['p_phi_095']:.3f}\n")
    sens = sensibilidad_w(n_mc)
    sens.to_csv(SALIDA_W, index=False)
    print(sens.round(3).to_string(index=False))
    print(f"\nGuardado en {SALIDA_PRED} y {SALIDA_W}")
