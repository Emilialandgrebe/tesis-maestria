"""`ParametrosMC.fuentes_fijas`: apagar fuentes estocásticas por su media
(scripts/descomponer_varianza.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.modelo_v2 import cargar_modelo_v2  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    FUENTES_ESTOCASTICAS,
    ParametrosMC,
    run_monte_carlo_antitetico,
)

N = 2_000


def _corrida(**kwargs):
    df = run_monte_carlo_antitetico(ParametrosMC(n_simulaciones=N, **kwargs))
    return df.pivot(index="simulacion", columns="año")


def test_todo_fijo_no_tiene_varianza():
    df = _corrida(fuentes_fijas=frozenset(FUENTES_ESTOCASTICAS), capex_opex_estocastico=False)
    assert np.allclose(df["van_neto_usd"].std(), 0.0, atol=1e-6)


def test_precio_fijo_es_la_trayectoria_del_excel():
    df = _corrida(fuentes_fijas=frozenset({"precio"}))
    trayectoria = cargar_modelo_v2().trayectoria_precio("base", 20)
    np.testing.assert_allclose(df["precio_usd_kg"].to_numpy(), np.tile(trayectoria, (N, 1)))


def test_fijar_una_fuente_no_cambia_los_sorteos_de_las_otras():
    """Números aleatorios comunes: con la vecería fija, el precio es el
    mismo realización a realización."""
    activo = _corrida()
    sin_veceria = _corrida(fuentes_fijas=frozenset({"veceria"}))
    np.testing.assert_array_equal(activo["precio_usd_kg"], sin_veceria["precio_usd_kg"])


@pytest.mark.parametrize("fuente", ["clima", "veceria", "plantas"])
def test_fijar_un_factor_conserva_el_rendimiento_medio(fuente):
    activo = _corrida()["rendimiento_kg_ha"].to_numpy()[:, 11:].mean()
    fijo = _corrida(fuentes_fijas=frozenset({fuente}))["rendimiento_kg_ha"].to_numpy()[:, 11:].mean()
    assert abs(fijo / activo - 1) < 0.01


def test_fuente_desconocida():
    with pytest.raises(ValueError, match="fuentes_fijas"):
        run_monte_carlo_antitetico(ParametrosMC(n_simulaciones=10, fuentes_fijas=frozenset({"lluvia"})))
