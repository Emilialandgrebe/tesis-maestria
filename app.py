"""Interfaz Streamlit sobre el motor de Monte Carlo del proyecto de pistacho.

Fase 4. Esta app es SOLO consumidora del motor (`src/monte_carlo.py`,
`src/costos.py`); no duplica ni modifica su lógica.

Secciones:
- Distribución del VAN al año 20 (histograma + métricas).
- Trayectoria por año (fan charts P10/P50/P90 de producción, ingreso y VAN).
- Sensibilidad de Sobol (tornado de ST, pre-calculado en
  `data/processed/sobol_indices.parquet` -- NO se recalcula en vivo).
- Comparación de los tres escenarios de precio a los parámetros del sidebar.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# La raíz del repo debe estar en sys.path para que `import src.*` funcione
# tanto si se ejecuta `streamlit run app.py` desde la raíz como desde otro cwd.
_RAIZ = Path(__file__).resolve().parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.monte_carlo import ParametrosMC, resumen_financiero, run_monte_carlo  # noqa: E402

SEMILLA = 42  # fija, igual que el default del motor (reproducibilidad)
ESCENARIOS = ["pesimista", "base", "optimista"]
COLOR_ESCENARIO = {"pesimista": "#C44E52", "base": "#4C72B0", "optimista": "#55A868"}
RUTA_SOBOL = _RAIZ / "data" / "processed" / "sobol_indices.parquet"


# ---------------------------------------------------------------------------
# Simulación cacheada
# ---------------------------------------------------------------------------

def _percentiles_por_año(df: pd.DataFrame, col: str, prefijo: str) -> pd.DataFrame:
    """P10/P50/P90 de `col` agrupando por año del proyecto."""
    g = df.groupby("año")[col]
    return pd.DataFrame({
        f"{prefijo}_p10": g.quantile(0.10),
        f"{prefijo}_p50": g.quantile(0.50),
        f"{prefijo}_p90": g.quantile(0.90),
    })


@st.cache_data(show_spinner=False)
def simular(
    hectareas: float,
    escenario: str,
    tasa_descuento: float,
    capex_opex_estocastico: bool,
    correlacionar_frio_calor: bool,
    n_simulaciones: int,
    semilla: int,
) -> dict:
    """Corre `run_monte_carlo()` una vez y devuelve lo que necesita la UI.

    Todos los argumentos son escalares/booleanos: forman la key de cache, así
    que Streamlit no recalcula si la combinación ya se corrió. La sección de
    comparación de escenarios reusa esta misma función (una entrada de cache
    por escenario), por lo que el escenario seleccionado en el sidebar se
    calcula una sola vez y lo comparten todas las secciones.

    Devuelve un dict con:
      - ``van``  : np.ndarray (VAN neto al año 20, uno por simulación)
      - ``tir``  : np.ndarray (TIR por simulación; puede tener NaN)
      - ``fan``  : DataFrame indexado por año, con columnas
                   ``prod_p{10,50,90}`` (kg/ha), ``ingreso_p{10,50,90}`` (USD)
                   y ``van_p{10,50,90}`` (USD acumulado descontado).
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
    resumen = resumen_financiero(df, costos)

    fan = pd.concat(
        [
            _percentiles_por_año(df, "rendimiento_kg_ha", "prod"),
            _percentiles_por_año(df, "ingreso_usd", "ingreso"),
            _percentiles_por_año(df, "van_neto_usd", "van"),
        ],
        axis=1,
    )
    return {
        "van": resumen["van_neto_usd"].to_numpy(),
        "tir": resumen["tir"].to_numpy(),
        "fan": fan,
    }


@st.cache_data(show_spinner=False)
def cargar_sobol(ruta: str) -> pd.DataFrame:
    """Índices de Sobol pre-calculados (`src/sensibilidad.py`)."""
    return pd.read_parquet(ruta)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Monte Carlo — Pistacho Jocolí", layout="wide")
st.title("Simulación de Monte Carlo — Pistacho Jocolí")
st.caption("Interfaz sobre el motor cerrado (`src/monte_carlo.py`). Fase 4, 2ª iteración.")

with st.sidebar:
    st.header("Parámetros")

    hectareas = st.slider(
        "Hectáreas", min_value=5, max_value=300, value=50, step=5,
        help=(
            "Superficie del proyecto. Escala lineal de CAPEX/OPEX/ingresos "
            "(sin economías de escala). Nota: el modelo ML surrogate y el "
            "Sobol pre-calculado sólo cubren hasta 100 ha; el motor simula "
            "cualquier superficie sin problema."
        ),
    )

    escenario = st.selectbox(
        "Escenario de precio", options=ESCENARIOS, index=1,
        help="Distribución triangular de precio USD/kg (ESCENARIOS_PRECIO en el motor).",
    )

    tasa_descuento = st.slider(
        "Tasa de descuento", min_value=0.02, max_value=0.20, value=0.08, step=0.005,
        format="%.3f",
        help="Variable dominante en el análisis de Sobol (S_T ≈ 0,86–0,92).",
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

_args_comunes = dict(
    hectareas=float(hectareas),
    tasa_descuento=float(tasa_descuento),
    capex_opex_estocastico=bool(capex_opex_estocastico),
    correlacionar_frio_calor=bool(correlacionar_frio_calor),
    n_simulaciones=int(n_simulaciones),
    semilla=SEMILLA,
)

with st.spinner("Corriendo Monte Carlo…"):
    res = simular(escenario=escenario, **_args_comunes)

van = res["van"]
tir = res["tir"]

tab_van, tab_fan, tab_sobol, tab_cmp = st.tabs(
    ["Distribución del VAN", "Trayectoria por año", "Sensibilidad (Sobol)", "Comparar escenarios"]
)


# ---------------------------------------------------------------------------
# Tab 1 — distribución del VAN al año 20
# ---------------------------------------------------------------------------

with tab_van:
    p10, p50, p90 = np.percentile(van, [10, 50, 90])
    van_medio = float(np.mean(van))
    prob_van_neg = float(np.mean(van < 0))
    tir_media = float(np.nanmean(tir))
    n_tir_nan = int(np.isnan(tir).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("VAN medio (año 20)", f"US$ {van_medio:,.0f}")
    c2.metric("P(VAN < 0)", f"{prob_van_neg:.1%}")
    c3.metric("TIR media", f"{tir_media:.2%}")

    st.markdown(
        f"**P10:** US$ {p10:,.0f}  ·  **P50:** US$ {p50:,.0f}  ·  **P90:** US$ {p90:,.0f}"
    )
    if n_tir_nan:
        st.caption(
            f"{n_tir_nan} de {len(tir)} simulaciones sin TIR definida (flujo sin "
            "cambio de signo); se excluyen del promedio."
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


# ---------------------------------------------------------------------------
# Tab 2 — fan charts por año
# ---------------------------------------------------------------------------

def _fan_chart(ax, fan: pd.DataFrame, prefijo: str, titulo: str, ylabel: str, escala: float = 1.0):
    años = fan.index.to_numpy()
    ax.fill_between(
        años, fan[f"{prefijo}_p10"] / escala, fan[f"{prefijo}_p90"] / escala,
        alpha=0.25, color="#4C72B0", label="P10–P90",
    )
    ax.plot(años, fan[f"{prefijo}_p50"] / escala, color="#4C72B0", linewidth=2, label="P50")
    ax.set_title(titulo)
    ax.set_xlabel("Año del proyecto")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


with tab_fan:
    st.caption(
        "Banda P10–P90 y mediana P50 por año, sobre las "
        f"{n_simulaciones:,} simulaciones del escenario **{escenario}**. "
        "El VAN es acumulado y descontado; producción e ingreso son anuales."
    )
    fan = res["fan"]
    fcol1, fcol2, fcol3 = st.columns(3)
    for columna, (prefijo, titulo, ylabel, escala) in zip(
        (fcol1, fcol2, fcol3),
        (
            ("prod", "Producción", "kg/ha", 1.0),
            ("ingreso", "Ingreso bruto anual", "millones USD", 1e6),
            ("van", "VAN neto acumulado", "millones USD", 1e6),
        ),
    ):
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        _fan_chart(ax, fan, prefijo, titulo, ylabel, escala)
        if prefijo == "van":
            ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
        fig.tight_layout()
        columna.pyplot(fig)


# ---------------------------------------------------------------------------
# Tab 3 — tornado de Sobol (pre-calculado)
# ---------------------------------------------------------------------------

with tab_sobol:
    if not RUTA_SOBOL.exists():
        st.warning(
            f"No se encontró `{RUTA_SOBOL.relative_to(_RAIZ)}`. "
            "Generá los índices con `src/sensibilidad.py` antes de usar esta sección."
        )
    else:
        sobol = cargar_sobol(str(RUTA_SOBOL))
        disponibles = list(sobol["escenario"].unique())
        if escenario in disponibles:
            esc_sobol, nota = escenario, ""
        else:
            esc_sobol = "base" if "base" in disponibles else disponibles[0]
            nota = f" · escenario '{escenario}' no pre-calculado, se muestra '{esc_sobol}'"

        sub = sobol[sobol["escenario"] == esc_sobol].sort_values("ST")

        st.caption(
            "Índices de Sobol **totales (ST)** del VAN medio, pre-calculados con "
            "`src/sensibilidad.py` (no se recalculan en vivo). Barras de error = "
            "IC bootstrap (`ST_conf`)."
        )
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh(
            sub["parametro"], sub["ST"],
            xerr=sub["ST_conf"].to_numpy(), color="#4C72B0",
            error_kw=dict(ecolor="#555555", lw=1),
        )
        ax.set_xlabel("Índice de Sobol total (ST)")
        ax.set_title(f"Sensibilidad del VAN medio — escenario {esc_sobol}{nota}")
        fig.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            sub.sort_values("ST", ascending=False)[
                ["parametro", "S1", "S1_conf", "ST", "ST_conf"]
            ].reset_index(drop=True),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 4 — comparación de escenarios lado a lado
# ---------------------------------------------------------------------------

with tab_cmp:
    st.caption(
        f"Los tres escenarios de precio a los parámetros actuales del sidebar "
        f"({hectareas} ha, tasa {tasa_descuento:.1%}, "
        f"CAPEX/OPEX {'estocástico' if capex_opex_estocastico else 'fijo'}, "
        f"N={n_simulaciones:,}). Cada corrida se cachea por separado."
    )
    with st.spinner("Corriendo los tres escenarios…"):
        cmp = {e: simular(escenario=e, **_args_comunes) for e in ESCENARIOS}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e in ESCENARIOS:
        v = cmp[e]["van"] / 1e6
        ax.hist(
            v, bins=60, density=True, histtype="step", linewidth=1.8,
            color=COLOR_ESCENARIO[e], label=e,
        )
        ax.axvline(float(np.mean(v)), color=COLOR_ESCENARIO[e], linestyle=":", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("VAN al año 20 (millones de USD)")
    ax.set_ylabel("Densidad")
    ax.set_title("VAN por escenario de precio (líneas punteadas = media)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    tabla = pd.DataFrame(
        {
            "escenario": ESCENARIOS,
            "VAN medio (USD)": [float(np.mean(cmp[e]["van"])) for e in ESCENARIOS],
            "P(VAN<0)": [float(np.mean(cmp[e]["van"] < 0)) for e in ESCENARIOS],
            "TIR media": [float(np.nanmean(cmp[e]["tir"])) for e in ESCENARIOS],
        }
    )
    st.dataframe(
        tabla.style.format(
            {"VAN medio (USD)": "{:,.0f}", "P(VAN<0)": "{:.1%}", "TIR media": "{:.2%}"}
        ),
        use_container_width=True,
    )
