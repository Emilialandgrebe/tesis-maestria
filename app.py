"""Interfaz Streamlit sobre el motor de Monte Carlo del proyecto de pistacho.

Versión mínima (Fase 4, primera iteración): expone en un sidebar los
parámetros dominantes de `ParametrosMC` y muestra la distribución del VAN al
año 20 con sus métricas destacadas. Esta app es SOLO consumidora del motor
(`src/monte_carlo.py`, `src/costos.py`); no duplica ni modifica su lógica.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# La raíz del repo debe estar en sys.path para que `import src.*` funcione
# tanto si se ejecuta `streamlit run app.py` desde la raíz como desde otro cwd.
_RAIZ = Path(__file__).resolve().parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.monte_carlo import ParametrosMC, resumen_financiero, run_monte_carlo  # noqa: E402

SEMILLA = 42  # fija, igual que el default del motor (reproducibilidad)


# ---------------------------------------------------------------------------
# Simulación cacheada
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def correr_simulacion(
    hectareas: float,
    escenario: str,
    tasa_descuento: float,
    capex_opex_estocastico: bool,
    correlacionar_frio_calor: bool,
    n_simulaciones: int,
    semilla: int,
):
    """Corre `run_monte_carlo()` y devuelve el resumen por simulación.

    Todos los argumentos son escalares/booleanos: forman parte de la key de
    cache, así que Streamlit no recalcula las N iteraciones si el sidebar no
    cambió respecto de la última corrida.
    """
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=hectareas,
        escenario=escenario,
        tasa_descuento=tasa_descuento,
        capex_opex_estocastico=capex_opex_estocastico,
        correlacionar_frio_calor=correlacionar_frio_calor,
        semilla=semilla,
    )
    costos = ParametrosCostos(hectareas=hectareas)
    df = run_monte_carlo(params, costos)
    # `resumen_financiero` necesita el MISMO objeto `costos` usado en la corrida
    # (run_monte_carlo sincroniza costos.hectareas con params.hectareas in-place).
    return resumen_financiero(df, costos)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Monte Carlo — Pistacho Jocolí", layout="wide")
st.title("Simulación de Monte Carlo — Pistacho Jocolí")
st.caption(
    "VAN al año 20 sobre el motor cerrado (`src/monte_carlo.py`). "
    "Primera versión mínima de la interfaz."
)

with st.sidebar:
    st.header("Parámetros")

    hectareas = st.slider(
        "Hectáreas", min_value=5, max_value=100, value=50, step=5,
        help="Superficie del proyecto. Escala lineal de CAPEX/OPEX (sin economías de escala).",
    )

    escenario = st.selectbox(
        "Escenario de precio", options=["pesimista", "base", "optimista"], index=1,
        help="Distribución triangular de precio USD/kg (ESCENARIOS_PRECIO en el motor).",
    )

    tasa_descuento = st.slider(
        "Tasa de descuento", min_value=0.02, max_value=0.20, value=0.08, step=0.005,
        format="%.3f",
        help="Variable dominante en el análisis de Sobol (S_T ≈ 0,89–0,93).",
    )

    capex_opex_estocastico = st.checkbox(
        "CAPEX/OPEX estocástico", value=True,
        help="Variabilidad de riego, pozo y multiplicador de OPEX. Default del motor oficial: activado.",
    )

    correlacionar_frio_calor = st.checkbox(
        "Correlacionar frío / calor", value=False,
        help="Cópula gaussiana frío↔calor (r≈-0,29, no significativa al 5%). Default: desactivado.",
    )

    st.divider()

    n_simulaciones = st.slider(
        "Nº de simulaciones", min_value=2000, max_value=10000, value=3000, step=1000,
        help="Más iteraciones = menos ruido de muestreo, pero más lento. Subir a 10.000 para precisión.",
    )

resumen = correr_simulacion(
    hectareas=float(hectareas),
    escenario=escenario,
    tasa_descuento=float(tasa_descuento),
    capex_opex_estocastico=bool(capex_opex_estocastico),
    correlacionar_frio_calor=bool(correlacionar_frio_calor),
    n_simulaciones=int(n_simulaciones),
    semilla=SEMILLA,
)

van = resumen["van_neto_usd"].to_numpy()
tir = resumen["tir"].to_numpy()

p10, p50, p90 = np.percentile(van, [10, 50, 90])
van_medio = float(np.mean(van))
prob_van_neg = float(np.mean(van < 0))
tir_media = float(np.nanmean(tir))
n_tir_nan = int(np.isnan(tir).sum())

col1, col2, col3 = st.columns(3)
col1.metric("VAN medio (año 20)", f"US$ {van_medio:,.0f}")
col2.metric("P(VAN < 0)", f"{prob_van_neg:.1%}")
col3.metric("TIR media", f"{tir_media:.2%}")

st.markdown(
    f"**P10:** US$ {p10:,.0f}  ·  **P50:** US$ {p50:,.0f}  ·  **P90:** US$ {p90:,.0f}"
)
if n_tir_nan:
    st.caption(
        f"{n_tir_nan} de {len(tir)} simulaciones sin TIR definida (flujo sin cambio "
        "de signo); se excluyen del promedio."
    )

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(van / 1e6, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
for valor, etiqueta, color in (
    (p10, "P10", "#C44E52"),
    (p50, "P50", "#55A868"),
    (p90, "P90", "#C44E52"),
):
    ax.axvline(valor / 1e6, color=color, linestyle="--", linewidth=1.5)
    ax.text(
        valor / 1e6, ax.get_ylim()[1] * 0.95, f" {etiqueta}",
        color=color, fontsize=9, va="top", ha="left",
    )
ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)
ax.set_xlabel("VAN al año 20 (millones de USD)")
ax.set_ylabel("Frecuencia")
ax.set_title(
    f"Distribución del VAN — {escenario}, {hectareas} ha, "
    f"tasa {tasa_descuento:.1%}, N={n_simulaciones:,}"
)
fig.tight_layout()
st.pyplot(fig)
