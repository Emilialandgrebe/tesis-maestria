"""Carga de los parámetros del modelo financiero v2 desde data/modelo_v2/.

Los CSV los genera `scripts/importar_modelo_v2.py` a partir de
`data/referencia/Modelo financiero - Pistacho Mendoza v2.xlsx`. Este módulo
es el único punto del motor que lee esos archivos: `src/costos.py`,
`src/monte_carlo.py` y `src/precio_estocastico.py` toman de acá supuestos,
CAPEX, OPEX y las series anuales (curva de rendimiento, precios por
escenario, rampa de fertilización).

Hay dos juegos de precios por escenario, planos en los años 7-20, en USD
constantes 2026:

- SECO con cáscara al productor (bloque 6; caso base desde 2026-11-19,
  venta_verde = 0): 5 / 7 / 9 USD/kg seco, junto con el bloque 8
  (conversión 1,00 y proporción al productor apagada, `src/costos.py`);
- en VERDE (bloque 9; alternativa, venta_verde = 1): lo que paga la
  procesadora por kg verde recién cosechado, 2,5 / 4 / 6 USD/kg verde.

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
    # Trayectoria creciente del plan original (8 -> 11,5 USD/kg). Solo
    # referencia: ningún caller la usa por defecto; sirve para reproducir el
    # plan original (tests/test_consistencia_excel.py).
    "plan_original": "precio_plan_original_usd_kg",
}
_COLUMNA_PRECIO_VERDE = {
    "pesimista": "precio_verde_pesimista_usd_kg",
    "base": "precio_verde_base_usd_kg",
    "optimista": "precio_verde_optimista_usd_kg",
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

    @property
    def venta_verde(self) -> bool:
        """Supuestos!B78: True = venta en verde (alternativa); False = venta
        en seco con cáscara (caso base desde 2026-11-19)."""
        return bool(self.supuestos["venta_verde"])

    def trayectoria_precio(
        self, escenario: str, n_años: int, venta_verde: bool | None = None
    ) -> np.ndarray:
        """Precio de referencia por año (USD constantes 2026).

        `venta_verde=True`: precio en verde, USD/kg verde (bloque 9).
        `venta_verde=False`: precio de mercado del procesado, USD/kg seco
        (bloque 6). None (default) sigue el caso base del Excel (B78).
        `escenario`: "pesimista", "base", "optimista" o, solo con precio de
        mercado, "plan_original" (trayectoria vieja, solo referencia).
        """
        if venta_verde is None:
            venta_verde = self.venta_verde
        columnas = _COLUMNA_PRECIO_VERDE if venta_verde else _COLUMNA_PRECIO
        if escenario not in columnas:
            base = "en verde" if venta_verde else "de mercado"
            raise ValueError(
                f"escenario con precio {base} debe ser uno de {list(columnas)}, recibido {escenario!r}"
            )
        return self.serie(columnas[escenario], n_años)

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
