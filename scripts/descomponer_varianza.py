"""Descomposición de la varianza del VAN por fuente estocástica.

Segunda capa de incertidumbre (riesgo intrínseco): con los supuestos FIJOS
en su valor por defecto, cuánta de la dispersión del VAN viene de cada
fuente de ruido dentro de la simulación. Es una pregunta distinta de Sobol
(`src/sensibilidad.py`), que barre supuestos inciertos (tasa, hectáreas,
CAPEX, drift...) y mide cuánto mueven el VAN medio o P(VAN<0).

Corridas (escenario base por defecto, `run_monte_carlo_antitetico`):

- todo activo;
- "sin X": la fuente X fijada en su valor esperado y el resto activo;
- "solo X": X activa y el resto fijo;
- todo fijo (rendimiento y precio en su media, OPEX sin ruido).

Todas usan la misma semilla y `ParametrosMC.fuentes_fijas` no cambia el
consumo de `rng`, así que las fuentes activas reciben los mismos números
en todas las corridas (números aleatorios comunes): las diferencias entre
corridas son efecto de la fuente, no ruido de muestreo.

Dos fracciones por fuente:

- fraccion_al_quitar = 1 - Var(sin X) / Var(todo): varianza que se pierde
  al apagar X, incluidas sus interacciones (análoga a un ST de Sobol).
- fraccion_sola = Var(solo X) / Var(todo): varianza que X genera por sí
  sola (análoga a un S1).

Si las fuentes no interactuaran, ambas coincidirían y sumarían 1. El VAN
es producto de rendimiento por precio y Ganancias es convexa (max(0, EBIT)),
así que hay interacción: la diferencia entre ambas columnas la mide.

IC95% por bootstrap sobre los PARES antitéticos (fila i con fila i + n/2),
para no romper la estructura del muestreo.

Uso:
    python scripts/descomponer_varianza.py [escenario] [n_simulaciones]
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
    FUENTES_ESTOCASTICAS,
    ParametrosMC,
    resumen_financiero,
    run_monte_carlo_antitetico,
)

FUENTES = (*FUENTES_ESTOCASTICAS, "opex")
N_BOOTSTRAP = 1_000
SALIDA = RAIZ / "outputs"


def _van(params: ParametrosMC, apagadas: set[str]) -> np.ndarray:
    corrida = replace(
        params,
        fuentes_fijas=frozenset(apagadas - {"opex"}),
        capex_opex_estocastico="opex" not in apagadas,
    )
    costos = ParametrosCostos(hectareas=corrida.hectareas)
    resumen = resumen_financiero(run_monte_carlo_antitetico(corrida, costos), costos)
    return resumen["van_neto_usd"].to_numpy()


def descomponer(escenario: str = "base", n_simulaciones: int = 10_000, semilla: int = 42):
    """Retorna (corridas, fracciones): una fila por corrida con media, desvío
    y P(VAN<0), y una fila por fuente con las dos fracciones de varianza."""
    params = ParametrosMC(n_simulaciones=n_simulaciones, escenario=escenario, semilla=semilla)
    todas = set(FUENTES)

    vans: dict[str, np.ndarray] = {"todo activo": _van(params, set())}
    for f in FUENTES:
        vans[f"sin {f}"] = _van(params, {f})
    for f in FUENTES:
        vans[f"solo {f}"] = _van(params, todas - {f})
    vans["todo fijo"] = _van(params, todas)

    corridas = pd.DataFrame([
        {
            "corrida": nombre,
            "van_medio_usd": v.mean(),
            "van_std_usd": v.std(ddof=1),
            "prob_van_negativo": (v < 0).mean(),
        }
        for nombre, v in vans.items()
    ])

    # Bootstrap sobre pares antitéticos (mismos índices para todas las corridas)
    n_pares = n_simulaciones // 2
    rng = np.random.default_rng(semilla)
    pares = rng.integers(0, n_pares, (N_BOOTSTRAP, n_pares))
    idx = np.concatenate([pares, pares + n_pares], axis=1)

    def _var(v: np.ndarray, muestras: np.ndarray | None = None) -> np.ndarray:
        return v.var(ddof=1) if muestras is None else v[muestras].var(axis=1, ddof=1)

    var_total, var_total_b = _var(vans["todo activo"]), _var(vans["todo activo"], idx)
    filas = []
    for f in FUENTES:
        quitar = 1 - _var(vans[f"sin {f}"]) / var_total
        quitar_b = 1 - _var(vans[f"sin {f}"], idx) / var_total_b
        sola = _var(vans[f"solo {f}"]) / var_total
        sola_b = _var(vans[f"solo {f}"], idx) / var_total_b
        filas.append({
            "fuente": f,
            "fraccion_al_quitar": quitar,
            "fraccion_al_quitar_ic95_inf": np.quantile(quitar_b, 0.025),
            "fraccion_al_quitar_ic95_sup": np.quantile(quitar_b, 0.975),
            "fraccion_sola": sola,
            "fraccion_sola_ic95_inf": np.quantile(sola_b, 0.025),
            "fraccion_sola_ic95_sup": np.quantile(sola_b, 0.975),
        })
    fracciones = pd.DataFrame(filas)
    return corridas, fracciones


if __name__ == "__main__":
    escenario = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

    corridas, fracciones = descomponer(escenario, n)

    SALIDA.mkdir(exist_ok=True)
    corridas.to_csv(SALIDA / f"descomposicion_varianza_{escenario}_corridas.csv", index=False)
    fracciones.to_csv(SALIDA / f"descomposicion_varianza_{escenario}_fracciones.csv", index=False)

    pd.set_option("display.width", 160)
    print(corridas.to_string(index=False))
    print()
    print(fracciones.to_string(index=False))
    print(f"\nSuma fraccion_al_quitar: {fracciones['fraccion_al_quitar'].sum():.3f}  "
          f"suma fraccion_sola: {fracciones['fraccion_sola'].sum():.3f}")
    print(f"Guardado en {SALIDA}")
