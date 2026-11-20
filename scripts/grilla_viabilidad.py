"""Grilla de viabilidad: precio × rendimiento a plena producción.

Dos bases de venta (argumento `base`):

- `verde` (default, caso base desde 2026-11-20): venta del pistacho recién
  cosechado, con pelón, a la procesadora (bloque 9). Precio 2-8 USD/kg
  verde, rendimiento 2.000-5.000 kg verdes/ha.
  Salida: outputs/grilla_viabilidad_verde.csv
- `seco` (alternativa): venta en seco con cáscara a precio al productor
  (Supuestos bloques 6 y 8). Precio 4-12 USD/kg seco, rendimiento
  2.000-4.000 kg secos con cáscara/ha.
  Salida: outputs/grilla_viabilidad_seco.csv

En cada celda corre el Monte Carlo completo (`run_monte_carlo_antitetico`,
motor por defecto: precio AR(1) en niveles, clima, vecería, falla de plantas
y OPEX estocástico, quebrantos) y guarda P(VAN<0) y el VAN mediano, más el
VAN medio, P10 y P90.

- Precio: nivel plano de los años 7-20 al que revierte el AR(1),
  `nivel_mult = precio / precio base del Excel` (4 USD/kg verde o 7 USD/kg
  seco).
- Rendimiento: kg/ha a plena producción (el 3.000 del Excel,
  Supuestos!B20), escalando `ParametrosMC.rendimiento_plena`; la curva de
  maduración y los factores estocásticos no cambian.

Todas las celdas usan la misma semilla (números aleatorios comunes): las
diferencias entre celdas son efecto del precio y el rendimiento, no ruido de
muestreo.

Uso:
    python scripts/grilla_viabilidad.py [n_simulaciones] [verde|seco]
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
from src.modelo_v2 import cargar_modelo_v2  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    RENDIMIENTO_IDEAL_KG_HA,
    RENDIMIENTO_VALIDADO_KG_HA,
    ParametrosMC,
    resumen_financiero,
    run_monte_carlo_antitetico,
)
from src.precio_estocastico import ParametrosPrecioAR1  # noqa: E402

BASES = {
    "verde": {
        "venta_verde": True,
        "precios": np.arange(2.0, 8.0 + 1e-9, 0.5),  # USD/kg verde
        "rendimientos": np.arange(2_000, 5_000 + 1, 250),  # kg verdes/ha a plena producción
        "col_precio": "precio_verde_usd_kg",
        "col_rend": "rendimiento_kg_verde_ha",
        "unidad_precio": "USD/kg verde",
        "unidad_rend": "kg verdes/ha",
    },
    "seco": {
        "venta_verde": False,
        "precios": np.arange(4.0, 12.0 + 1e-9, 0.5),  # USD/kg seco al productor
        "rendimientos": np.arange(2_000, 4_000 + 1, 250),  # kg secos/ha a plena producción
        "col_precio": "precio_seco_usd_kg",
        "col_rend": "rendimiento_kg_seco_ha",
        "unidad_precio": "USD/kg seco",
        "unidad_rend": "kg secos con cáscara/ha",
    },
}


def _precio_base(n_años: int, venta_verde: bool) -> float:
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", n_años, venta_verde=venta_verde)
    return float(trayectoria[trayectoria > 0][0])


def celda(
    precio: float, rendimiento: float, n_simulaciones: int, base: str = "verde", semilla: int = 42
) -> dict:
    cfg = BASES[base]
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        escenario="base",
        semilla=semilla,
        rendimiento_plena=RENDIMIENTO_IDEAL_KG_HA * rendimiento / RENDIMIENTO_VALIDADO_KG_HA,
    )
    costos = ParametrosCostos(hectareas=params.hectareas, venta_verde=cfg["venta_verde"])
    precio_params = ParametrosPrecioAR1(
        escenario="base",
        venta_verde=cfg["venta_verde"],
        nivel_mult=precio / _precio_base(params.n_años, cfg["venta_verde"]),
    )
    van = resumen_financiero(
        run_monte_carlo_antitetico(params, costos, precio_params), costos
    )["van_neto_usd"].to_numpy()
    return {
        cfg["col_precio"]: precio,
        cfg["col_rend"]: rendimiento,
        "p_van_negativo": float((van < 0).mean()),
        "van_mediano_usd": float(np.median(van)),
        "van_medio_usd": float(van.mean()),
        "van_p10_usd": float(np.percentile(van, 10)),
        "van_p90_usd": float(np.percentile(van, 90)),
    }


def grilla(n_simulaciones: int = 5_000, base: str = "verde") -> pd.DataFrame:
    precios = BASES[base]["precios"]
    rendimientos = BASES[base]["rendimientos"]
    filas = []
    inicio = time.time()
    total = len(precios) * len(rendimientos)
    for rendimiento in rendimientos:
        for precio in precios:
            filas.append(celda(float(precio), float(rendimiento), n_simulaciones, base))
        print(f"{len(filas)}/{total} celdas ({time.time() - inicio:.0f}s)")
    return pd.DataFrame(filas)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000
    base = sys.argv[2] if len(sys.argv) > 2 else "verde"
    cfg = BASES[base]
    salida = RAIZ / "outputs" / f"grilla_viabilidad_{base}.csv"
    df = grilla(n, base)
    salida.parent.mkdir(exist_ok=True)
    df.to_csv(salida, index=False)

    pd.set_option("display.width", 250)
    for col, fmt in (("p_van_negativo", "{:.0%}"), ("van_mediano_usd", "{:,.0f}")):
        tabla = df.pivot(index=cfg["col_rend"], columns=cfg["col_precio"], values=col)
        print(f"\n{col} (filas: {cfg['unidad_rend']}; columnas: {cfg['unidad_precio']})")
        print(tabla.map(fmt.format).to_string())
    print(f"\nGuardado en {salida}")
