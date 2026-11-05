"""Carga de los parámetros del modelo financiero v2 desde data/modelo_v2/.

Los CSV los genera `scripts/importar_modelo_v2.py` a partir de
`data/referencia/Modelo financiero - Pistacho Mendoza v2.xlsx`. Este módulo
es el único punto del motor que lee esos archivos: `src/costos.py`,
`src/monte_carlo.py` y `src/precio_estocastico.py` toman de acá supuestos,
CAPEX, OPEX y las series anuales (curva de rendimiento, precios por
escenario, rampa de fertilización).

Todas las series anuales se devuelven indexadas por año del proyecto 1..20
(el año 0 del Excel solo tiene CAPEX, que el motor maneja aparte).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DIR_MODELO_V2 = Path(__file__).resolve().parents[1] / "data" / "modelo_v2"

_COLUMNA_PRECIO = {
    "pesimista": "precio_pesimista_usd_kg",
    "base": "precio_base_usd_kg",
    "optimista": "precio_optimista_usd_kg",
}


@dataclass(frozen=True)
class ModeloV2:
    """Parámetros del Excel v2, tal como los exportó el importador."""

    supuestos: dict[str, float]
    capex: pd.DataFrame
    opex: pd.DataFrame
    anual: pd.DataFrame

    @property
    def n_años(self) -> int:
        """Horizonte del Excel (años con flujo operativo, sin contar el año 0)."""
        return int(self.anual["anio"].max())

    def serie(self, columna: str, n_años: int) -> np.ndarray:
        """Columna de anual.csv para los años 1..n_años, forma (n_años,)."""
        if n_años > self.n_años:
            raise ValueError(
                f"n_años={n_años} supera el horizonte del Excel ({self.n_años} años): "
                "no hay curva ni precios definidos más allá."
            )
        valores = self.anual.set_index("anio")[columna]
        return valores.loc[1:n_años].to_numpy(dtype=float)

    def curva_excel(self, n_años: int) -> np.ndarray:
        """% de plena producción por año, tal cual el Excel (vecería incluida)."""
        return self.serie("rendimiento_pct_plena", n_años)

    def trayectoria_precio(self, escenario: str, n_años: int) -> np.ndarray:
        """Precio USD/kg (USD constantes 2026) por año para un escenario."""
        if escenario not in _COLUMNA_PRECIO:
            raise ValueError(f"escenario debe ser uno de {list(_COLUMNA_PRECIO)}, recibido {escenario!r}")
        return self.serie(_COLUMNA_PRECIO[escenario], n_años)

    def rampa_fertilizacion(self, n_años: int) -> np.ndarray:
        """Fracción de la dosis plena de fertilización/fitosanidad por año."""
        return self.serie("rampa_fertilizacion", n_años)


@lru_cache(maxsize=None)
def cargar_modelo_v2(directorio: Path = DIR_MODELO_V2) -> ModeloV2:
    """Lee los cuatro CSV de parámetros. Cacheado: se lee una vez por proceso."""
    supuestos = pd.read_csv(directorio / "supuestos.csv")
    return ModeloV2(
        supuestos=dict(zip(supuestos["parametro"], supuestos["valor"].astype(float))),
        capex=pd.read_csv(directorio / "capex.csv"),
        opex=pd.read_csv(directorio / "opex.csv"),
        anual=pd.read_csv(directorio / "anual.csv"),
    )
