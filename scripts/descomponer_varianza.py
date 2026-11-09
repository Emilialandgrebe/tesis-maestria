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

Tercer argumento, especificación del precio:

- `niveles` (default): la del motor, AR(1) en niveles que revierte a la
  trayectoria del Excel.
- `retornos`: escenario de estrés, AR(1) sobre retornos sin reversión
  (`precio_retornos_estres()`).
- `niveles_plano`: niveles con trayectoria REAL CONSTANTE, el promedio de
  la del Excel en los años con precio (7-20).
- `niveles_plano_8`: niveles con trayectoria constante en el precio del
  año 7 (sin la suba real posterior).

Las dos planas miden cuánto del resultado depende de la suba real de la
trayectoria.

Cuarto argumento opcional, escenario de estrés climático
(`ParametrosMC.estres_climatico`): `puntual` o `extremo` (pendiente de Sen
de ERA5-Land o extremo de su IC95%). Sin él, clima sin tendencia.

Uso:
    python scripts/descomponer_varianza.py [escenario] [n_simulaciones] [niveles|retornos|niveles_plano|niveles_plano_8] [puntual|extremo]
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
from src.modelo_v2 import cargar_modelo_v2  # noqa: E402
from src.precio_estocastico import ParametrosPrecioAR1, precio_retornos_estres  # noqa: E402

FUENTES = (*FUENTES_ESTOCASTICAS, "opex")
N_BOOTSTRAP = 1_000
SALIDA = RAIZ / "outputs"


def _van(
    params: ParametrosMC, apagadas: set[str], precio_params: ParametrosPrecioAR1
) -> np.ndarray:
    corrida = replace(
        params,
        fuentes_fijas=frozenset(apagadas - {"opex"}),
        capex_opex_estocastico="opex" not in apagadas,
    )
    costos = ParametrosCostos(hectareas=corrida.hectareas)
    resumen = resumen_financiero(
        run_monte_carlo_antitetico(corrida, costos, precio_params), costos
    )
    return resumen["van_neto_usd"].to_numpy()


def _precio(escenario: str, especificacion: str, n_años: int) -> ParametrosPrecioAR1:
    if especificacion == "niveles":
        return ParametrosPrecioAR1(escenario=escenario)
    if especificacion == "retornos":
        return precio_retornos_estres(escenario)
    excel = cargar_modelo_v2().trayectoria_precio(escenario, n_años)
    con_precio = excel > 0
    if especificacion == "niveles_plano":
        nivel = excel[con_precio].mean()
    elif especificacion == "niveles_plano_8":
        nivel = excel[con_precio][0]
    else:
        raise ValueError(f"especificación desconocida: {especificacion!r}")
    return ParametrosPrecioAR1(
        escenario=escenario, trayectoria=tuple(np.where(con_precio, nivel, 0.0))
    )


def descomponer(
    escenario: str = "base",
    n_simulaciones: int = 10_000,
    semilla: int = 42,
    especificacion: str = "niveles",
    estres_variante: str | None = None,
):
    """Retorna (corridas, fracciones): una fila por corrida con media, desvío
    y P(VAN<0), y una fila por fuente con las dos fracciones de varianza."""
    params = ParametrosMC(
        n_simulaciones=n_simulaciones, escenario=escenario, semilla=semilla,
        estres_climatico=estres_variante is not None,
        estres_variante=estres_variante or "puntual",
    )
    precio_params = _precio(escenario, especificacion, params.n_años)
    todas = set(FUENTES)

    vans: dict[str, np.ndarray] = {"todo activo": _van(params, set(), precio_params)}
    for f in FUENTES:
        vans[f"sin {f}"] = _van(params, {f}, precio_params)
    for f in FUENTES:
        vans[f"solo {f}"] = _van(params, todas - {f}, precio_params)
    vans["todo fijo"] = _van(params, todas, precio_params)

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
    especificacion = sys.argv[3] if len(sys.argv) > 3 else "niveles"
    estres = sys.argv[4] if len(sys.argv) > 4 else None

    corridas, fracciones = descomponer(
        escenario, n, especificacion=especificacion, estres_variante=estres
    )

    sufijo = escenario if especificacion == "niveles" else f"{escenario}_precio_{especificacion}"
    if estres:
        sufijo += f"_estres_{estres}"
    SALIDA.mkdir(exist_ok=True)
    corridas.to_csv(SALIDA / f"descomposicion_varianza_{sufijo}_corridas.csv", index=False)
    fracciones.to_csv(SALIDA / f"descomposicion_varianza_{sufijo}_fracciones.csv", index=False)

    pd.set_option("display.width", 160)
    print(corridas.to_string(index=False))
    print()
    print(fracciones.to_string(index=False))
    print(f"\nSuma fraccion_al_quitar: {fracciones['fraccion_al_quitar'].sum():.3f}  "
          f"suma fraccion_sola: {fracciones['fraccion_sola'].sum():.3f}")
    print(f"Guardado en {SALIDA}")
