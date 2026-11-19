"""Grilla de viabilidad: precio en verde × rendimiento a plena producción.

Caso base (venta en verde a la procesadora, Supuestos bloque 9). En cada
celda corre el Monte Carlo completo (`run_monte_carlo_antitetico`, motor por
defecto: precio AR(1) en niveles, clima, vecería, falla de plantas y OPEX
estocástico, quebrantos) y guarda P(VAN<0) y el VAN mediano, más el VAN
medio, P10 y P90.

- Precio en verde: nivel plano de los años 7-20 al que revierte el AR(1),
  `nivel_mult = precio / precio en verde base` (4 USD/kg verde).
- Rendimiento: kg verdes/ha a plena producción (el 3.000 del Excel,
  Supuestos!B20), escalando `ParametrosMC.rendimiento_plena`; la curva de
  maduración y los factores estocásticos no cambian.

Todas las celdas usan la misma semilla (números aleatorios comunes): las
diferencias entre celdas son efecto del precio y el rendimiento, no ruido de
muestreo.

Uso:
    python scripts/grilla_viabilidad.py [n_simulaciones]

Salida: outputs/grilla_viabilidad_verde.csv
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

PRECIOS_VERDE = np.arange(1.5, 10.0 + 1e-9, 0.5)  # USD/kg verde
RENDIMIENTOS = np.arange(2_000, 4_000 + 1, 250)  # kg verdes/ha a plena producción
SALIDA = RAIZ / "outputs" / "grilla_viabilidad_verde.csv"


def _precio_verde_base(n_años: int) -> float:
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", n_años, venta_verde=True)
    return float(trayectoria[trayectoria > 0][0])


def celda(precio: float, rendimiento: float, n_simulaciones: int, semilla: int = 42) -> dict:
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        escenario="base",
        semilla=semilla,
        rendimiento_plena=RENDIMIENTO_IDEAL_KG_HA * rendimiento / RENDIMIENTO_VALIDADO_KG_HA,
    )
    costos = ParametrosCostos(hectareas=params.hectareas, venta_verde=True)
    precio_params = ParametrosPrecioAR1(
        escenario="base", venta_verde=True, nivel_mult=precio / _precio_verde_base(params.n_años)
    )
    van = resumen_financiero(
        run_monte_carlo_antitetico(params, costos, precio_params), costos
    )["van_neto_usd"].to_numpy()
    return {
        "precio_verde_usd_kg": precio,
        "rendimiento_kg_verde_ha": rendimiento,
        "p_van_negativo": float((van < 0).mean()),
        "van_mediano_usd": float(np.median(van)),
        "van_medio_usd": float(van.mean()),
        "van_p10_usd": float(np.percentile(van, 10)),
        "van_p90_usd": float(np.percentile(van, 90)),
    }


def grilla(n_simulaciones: int = 5_000) -> pd.DataFrame:
    filas = []
    inicio = time.time()
    total = len(PRECIOS_VERDE) * len(RENDIMIENTOS)
    for rendimiento in RENDIMIENTOS:
        for precio in PRECIOS_VERDE:
            filas.append(celda(float(precio), float(rendimiento), n_simulaciones))
        print(f"{len(filas)}/{total} celdas ({time.time() - inicio:.0f}s)")
    return pd.DataFrame(filas)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000
    df = grilla(n)
    SALIDA.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA, index=False)

    pd.set_option("display.width", 250)
    for col, fmt in (("p_van_negativo", "{:.0%}"), ("van_mediano_usd", "{:,.0f}")):
        tabla = df.pivot(index="rendimiento_kg_verde_ha", columns="precio_verde_usd_kg", values=col)
        print(f"\n{col} (filas: kg verdes/ha; columnas: USD/kg verde)")
        print(tabla.map(fmt.format).to_string())
    print(f"\nGuardado en {SALIDA}")
