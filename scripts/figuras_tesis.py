"""Figuras de la tesis (caso base: venta en verde a la procesadora).

- outputs/figuras/grilla_viabilidad_verde.png: mapa de calor de P(VAN < 0)
  sobre la grilla precio en verde x rendimiento a plena producción
  (outputs/grilla_viabilidad_verde.csv, de scripts/grilla_viabilidad.py), con
  la curva de VAN mediano = 0, el caso base (4 USD/kg verde, 3.000 kg/ha) y
  el rango de precios del Excel (2,5-5,5 USD/kg verde).
- outputs/figuras/tornado_sobol_bayes.png: índices de Sobol totales (ST, con
  su IC bootstrap ST_conf) del VAN medio y del CVaR 10 % con el precio
  bayesiano (data/processed/sobol_indices_bayes.parquet).

Formato numérico argentino (coma decimal), 300 dpi.

Uso:
    python scripts/figuras_tesis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import ticker  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DIR_FIGURAS = RAIZ / "outputs" / "figuras"
DPI = 300

NOMBRE_PARAMETRO = {
    "tasa_descuento": "Tasa de descuento real",
    "capex_extra_pct": "Recargo de CAPEX",
    "peso_exportacion": "Peso de la paridad de exportación (w)",
    "factor_ars": "Tipo de cambio real (costos en pesos)",
    "p_bajo_si_alto": "Vecería: P(año bajo | alto)",
    "precision_factor_frio": "Precisión del factor de frío",
    "precision_factor_calor": "Precisión del factor de calor",
}
NOMBRE_METRICA = {"van_neto_medio_usd": "VAN medio", "cvar_10_usd": "CVaR 10 %"}


def _coma(x: float, decimales: int) -> str:
    return f"{x:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def figura_grilla(ruta_csv: Path, salida: Path) -> None:
    df = pd.read_csv(ruta_csv)
    p = df.pivot(index="rendimiento_kg_verde_ha", columns="precio_verde_usd_kg", values="p_van_negativo")
    van = df.pivot(index="rendimiento_kg_verde_ha", columns="precio_verde_usd_kg", values="van_mediano_usd")
    precios, rend = p.columns.to_numpy(), p.index.to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    paso_p, paso_r = precios[1] - precios[0], rend[1] - rend[0]
    extent = [precios[0] - paso_p / 2, precios[-1] + paso_p / 2, rend[0] - paso_r / 2, rend[-1] + paso_r / 2]
    im = ax.imshow(p.to_numpy() * 100, origin="lower", aspect="auto", extent=extent,
                   cmap="RdYlGn_r", vmin=0, vmax=100)
    for i, r in enumerate(rend):
        for j, pr in enumerate(precios):
            v = p.iloc[i, j] * 100
            ax.text(pr, r, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 85 or v < 15 else "black")
    cs = ax.contour(precios, rend, van.to_numpy(), levels=[0], colors="black", linewidths=1.8)
    ax.clabel(cs, fmt={0: "VAN mediano = 0"}, fontsize=8)
    for x in (2.5, 5.5):
        ax.axvline(x, color="#1f3b73", linestyle="--", linewidth=1.3,
                   label="Rango de precios del Excel (2,5–5,5 USD/kg verde)" if x == 2.5 else None)
    ax.add_patch(plt.Rectangle((4.0 - paso_p / 2, 3000 - paso_r / 2), paso_p, paso_r, fill=False,
                               edgecolor="#1f3b73", linewidth=2.2,
                               label="Caso base (4 USD/kg verde; 3.000 kg/ha)"))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(VAN < 0) (%)")
    ax.set_xlabel("Precio en verde de largo plazo (USD/kg verde, constantes 2026)")
    ax.set_ylabel("Rendimiento a plena producción (kg verdes/ha)")
    ax.set_title("Probabilidad de VAN negativo según precio y rendimiento\n"
                 "(venta en verde a la procesadora, 120 ha, Monte Carlo de 5.000 simulaciones por celda)",
                 fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _coma(x, 0)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: _coma(y, 0)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(salida, dpi=DPI)
    plt.close(fig)


def figura_tornado(ruta_parquet: Path, salida: Path) -> None:
    sobol = pd.read_parquet(ruta_parquet)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, (metrica, nombre) in zip(ejes, NOMBRE_METRICA.items()):
        sub = sobol[sobol["metrica"] == metrica].sort_values("ST")
        etiquetas = sub["parametro"].map(lambda q: NOMBRE_PARAMETRO.get(q, q))
        ax.barh(etiquetas, sub["ST"], xerr=sub["ST_conf"].to_numpy(), color="#4C72B0",
                error_kw=dict(ecolor="#555555", lw=1, capsize=3))
        for y, (st, conf) in enumerate(zip(sub["ST"], sub["ST_conf"])):
            ax.text(st + conf + 0.01, y, _coma(st, 2), va="center", fontsize=8)
        ax.set_xlim(0, max(0.8, float((sub["ST"] + sub["ST_conf"]).max()) + 0.1))
        ax.set_xlabel("Índice de Sobol total (ST)")
        ax.set_title(f"{nombre}  (ΣS1 = {_coma(sub['S1'].sum(), 2)})")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _coma(x, 1)))
    fig.suptitle("Sensibilidad global con el precio bayesiano (venta en verde, 120 ha; "
                 "N = 256, 2.304 evaluaciones de 1.000 simulaciones; barras: IC 95 % bootstrap)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_grilla(RAIZ / "outputs" / "grilla_viabilidad_verde.csv", DIR_FIGURAS / "grilla_viabilidad_verde.png")
    figura_tornado(RAIZ / "data" / "processed" / "sobol_indices_bayes.parquet",
                   DIR_FIGURAS / "tornado_sobol_bayes.png")
    print(f"Guardado en {DIR_FIGURAS}")
