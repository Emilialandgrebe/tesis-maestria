"""Interfaz Streamlit sobre el motor de Monte Carlo del proyecto de pistacho.

Fase 4. Esta app es SOLO consumidora del motor (`src/monte_carlo.py`,
`src/costos.py`); no duplica ni modifica su lógica.

Secciones:
- Distribución del VAN al año 20 (histograma + métricas).
- Trayectoria por año (fan charts P10/P50/P90 de producción, precio, ingreso y VAN).
- Sensibilidad de Sobol (tornados de ST del VAN medio y del CVaR 10 %,
  pre-calculados con el precio bayesiano en
  `data/processed/sobol_indices_bayes.parquet`, superficie fija en 120 ha --
  NO se recalcula en vivo).
- Comparación de los tres escenarios de precio a los parámetros del sidebar.

El sidebar incluye los dos interruptores fiscales del Excel: traslado de
quebrantos (encendido por defecto, caso base) y activación del OPEX
pre-productivo como costo de implantación (apagado por defecto,
sensibilidad). Aplican a todas las pestañas salvo Sobol, que está
pre-calculado con el caso base (quebrantos sí, activación no, sin estrés
climático). También incluye el escenario de estrés
climático (tendencia de frío y calor de ERA5-Land, sin renormalizar).

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import matplotlib.pyplot as plt
from matplotlib import ticker
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
from src.modelo_v2 import cargar_modelo_v2  # noqa: E402
from src.precio_estocastico import precio_retornos_estres  # noqa: E402

SEMILLA = 42  # fija, igual que el default del motor (reproducibilidad)
ESCENARIOS = ["pesimista", "base", "optimista"]
COLOR_ESCENARIO = {"pesimista": "#C44E52", "base": "#4C72B0", "optimista": "#55A868"}
RUTA_SOBOL = _RAIZ / "data" / "processed" / "sobol_indices_bayes.parquet"
_NOMBRE_PRECIO = {
    "ar1": "AR(1) en niveles",
    "ar1_retornos": "AR(1) retornos (estrés)",
    "triangular": "triangular",
    "bayesiano": "bayesiano",
}
# Etiqueta del sidebar -> variante de ParametrosMC (None = sin estrés).
_ESTRES_CLIMATICO = {
    "Sin estrés": None,
    "Tendencia de Sen (puntual)": "puntual",
    "Extremo del IC95%": "extremo",
    "Solo frío (extremo)": "solo_frio_extremo",
}
_NOMBRE_PARAMETRO = {
    "tasa_descuento": "Tasa de descuento real",
    "precio_nivel_mult": "Nivel del precio (× nivel base)",
    "precio_drift_anual": "Drift real del precio",
    "hectareas": "Hectáreas",
    "capex_extra_pct": "Recargo de CAPEX",
    "p_bajo_si_alto": "Vecería: P(año bajo | alto)",
    "precision_factor_frio": "Precisión del factor de frío",
    "precision_factor_calor": "Precisión del factor de calor",
    "peso_exportacion": "Peso de la paridad de exportación (w)",
    "factor_ars": "Tipo de cambio real (costos en pesos)",
}
_NOMBRE_METRICA = {"van_neto_medio_usd": "VAN medio", "cvar_10_usd": "CVaR 10%"}


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


UMBRAL_PERDIDA_USD = 500_000


# ---------------------------------------------------------------------------
# Formato numérico argentino: punto de miles, coma decimal, signo menos "−"
# ---------------------------------------------------------------------------

def _num(x: float, dec: int = 0) -> str:
    """12345.678 -> "12.345,68" (dec=2); negativos con "−"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    texto = f"{abs(x):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"−{texto}" if x < 0 and texto.strip("0,.") else texto


def _pct(x: float, dec: int = 1) -> str:
    """0.6235 -> "62,4%"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{_num(100 * x, dec)}%"


def _usd(x: float) -> str:
    """-2836458 -> "US$ −2.836.458"."""
    return f"US$ {_num(x, 0)}"


def _usd_m(x: float) -> str:
    """-2836458 -> "US$ −2,84 M"."""
    return f"US$ {_num(x / 1e6, 2)} M"


class _FormatoEjeAR(ticker.Formatter):
    """Ticks de matplotlib con formato argentino. Los decimales salen del
    paso entre ticks (p. ej. paso 0,5 -> 1 decimal)."""

    def __call__(self, x, pos=None):
        locs = [v for v in self.locs if np.isfinite(v)]
        paso = min((abs(b - a) for a, b in zip(locs, locs[1:]) if b != a), default=1.0)
        # Menos decimales que representan el paso exacto (2,5 -> 1; 0,25 -> 2).
        dec = next((d for d in range(5) if abs(round(paso, d) - paso) < paso * 1e-6), 4)
        return _num(0.0 if abs(x) < paso * 1e-9 else x, dec)


def _mostrar(fig, destino=st) -> None:
    """Aplica el formato argentino a los ejes numéricos y muestra la figura."""
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if isinstance(eje.get_major_formatter(), ticker.ScalarFormatter):
                eje.set_major_formatter(_FormatoEjeAR())
    fig.tight_layout()
    destino.pyplot(fig)


def _metricas_van(van: np.ndarray, tir: np.ndarray) -> dict:
    """Métricas de resumen de una corrida, compartidas por las pestañas
    Distribución del VAN y Comparar escenarios.

    - TIR mediana: sin las simulaciones con TIR no definida (NaN).
    - CVaR 10%: promedio del VAN en el 10% de simulaciones con menor VAN
      (las ceil(0,1·n) peores).
    """
    n_peores = max(1, int(np.ceil(0.10 * len(van))))
    return {
        "van_medio": float(np.mean(van)),
        "van_mediano": float(np.median(van)),
        "prob_van_neg": float(np.mean(van < 0)),
        "prob_perdida_grande": float(np.mean(van < -UMBRAL_PERDIDA_USD)),
        "cvar_10": float(np.mean(np.sort(van)[:n_peores])),
        "tir_mediana": float(np.nanmedian(tir)) if np.isfinite(tir).any() else float("nan"),
        "n_tir_nan": int(np.isnan(tir).sum()),
    }


def _ayuda_tir(n_tir_nan: int, n: int) -> str:
    if n_tir_nan == 0:
        return f"Mediana de la TIR real. Las {_num(n)} simulaciones tienen TIR definida."
    return (
        f"Mediana de la TIR real. {_num(n_tir_nan)} de {_num(n)} simulaciones no tienen "
        "TIR definida (no hay tasa entre −99% y 1.000% que anule el VAN) y se "
        "excluyen de la mediana."
    )


@st.cache_data(show_spinner=False)
def simular(
    hectareas: float,
    escenario: str,
    tasa_descuento: float,
    capex_opex_estocastico: bool,
    correlacionar_frio_calor: bool,
    modo_precio: str,
    n_simulaciones: int,
    semilla: int,
    quebrantos: bool = True,
    activar_opex_pre: bool = False,
    estres_climatico: str | None = None,
    factor_ars_estocastico: bool = False,
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
        # "ar1_retornos" es el AR(1) de estrés (sobre retornos, sin
        # reversión): el mismo generador "ar1" del motor con otros parámetros.
        modo_precio="ar1" if modo_precio == "ar1_retornos" else modo_precio,
        semilla=semilla,
        estres_climatico=estres_climatico is not None,
        estres_variante=estres_climatico or "puntual",
        factor_ars_estocastico=factor_ars_estocastico,
    )
    costos = ParametrosCostos(
        hectareas=hectareas, quebrantos=quebrantos, activar_opex_pre=activar_opex_pre
    )
    precio_params = precio_retornos_estres(escenario) if modo_precio == "ar1_retornos" else None
    df = run_monte_carlo(params, costos, precio_params)
    # `resumen_financiero` necesita el MISMO objeto `costos` usado en la corrida
    # (run_monte_carlo sincroniza costos.hectareas con params.hectareas in-place).
    resumen = resumen_financiero(df, costos)

    fan = pd.concat(
        [
            _percentiles_por_año(df, "rendimiento_kg_ha", "prod"),
            _percentiles_por_año(df, "precio_usd_kg", "precio"),
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
st.subheader("Caso base: venta del pistacho seco con cáscara, a precio al productor")
st.caption(
    "Interfaz sobre el motor (`src/monte_carlo.py`): 120 ha, USD constantes "
    "2026, tasa real, precio AR(1) en niveles calibrado con FRED."
)

with st.sidebar:
    st.header("Parámetros")

    hectareas = st.slider(
        "Hectáreas", min_value=5, max_value=300, value=120, step=5,
        help=(
            "Superficie del proyecto. CAPEX/OPEX escalan con las reglas del "
            "Excel v2 (por hectárea, por planta, 1 sistema solar cada 50 ha, "
            "1 operario cada 8 ha). Nota: el modelo ML surrogate cubre "
            "60-180 ha y el Sobol pre-calculado usa 120 ha fijas; el motor "
            "simula cualquier superficie sin problema."
        ),
    )

    escenario = st.selectbox(
        "Escenario de precio", options=ESCENARIOS, index=1,
        help=(
            "Nivel de precio ancla (USD/kg seco con cáscara, al productor) del "
            "proyecto: pesimista 5, base 7 y optimista 9 USD/kg seco (Excel v2, "
            "bloque 6). La finca paga la maquila de pelado y secado (0,40 "
            "USD/kg seco). La venta en verde queda como alternativa. El modelo "
            "probabilístico que genera la trayectoria de precio alrededor "
            "de ese ancla se elige abajo, en 'Modelo de precio'. En el modelo "
            "bayesiano el escenario fija el peso w de la paridad de "
            "exportación: pesimista w = 1 (nivel medio 5), base 0,5 (7) y "
            "optimista 0 (9)."
        ),
    )

    modo_precio = st.selectbox(
        "Modelo de precio", options=["ar1", "bayesiano", "ar1_retornos", "triangular"], index=0,
        format_func=lambda m: {
            "ar1": "AR(1) en niveles, revierte al nivel (principal)",
            "bayesiano": "Bayesiano: posterior de FRED y paridades exportación/importación",
            "ar1_retornos": "AR(1) sobre retornos, sin reversión (estrés)",
            "triangular": "Triangular independiente por año (naive)",
        }[m],
        help=(
            "AR(1) en niveles (default del motor): el precio oscila "
            "alrededor del nivel constante del Excel y vuelve a él (semivida "
            "~4 años), calibrado con la serie real deflactada de FRED "
            "(WPU01190106) -- ver src/precio_estocastico.py. "
            "Bayesiano (Paso 7, src/precio_bayesiano.py): mismo AR(1) en "
            "niveles, pero phi y sigma se sortean por simulación de la "
            "posterior de FRED (phi mediana 0,87, IC90 0,73–0,98: admite raíz "
            "unitaria) y el nivel de largo plazo es una mezcla de la paridad "
            "de exportación (~5 USD/kg) y la de importación (~9), con peso w "
            "según el escenario. "
            "AR(1) sobre retornos: el desvío no revierte y la dispersión del "
            "año 20 duplica el rango histórico; se usa como estrés. "
            "Triangular: ESCENARIOS_PRECIO, independiente por año sin "
            "memoria -- el modelo naive anterior, disponible acá solo como "
            "comparación explícita."
        ),
    )

    if modo_precio == "bayesiano":
        factor_ars_estocastico = st.checkbox(
            "Tipo de cambio real estocástico", value=True,
            help=(
                "Un sorteo por simulación, Uniforme(0,8; 1,2), constante en los "
                "20 años: shock permanente sobre los costos que se pagan en "
                "pesos (CAPEX, OPEX, sueldos y maquila, según moneda_pago). "
                "< 1 = devaluación real, > 1 = atraso cambiario. Activado por "
                "defecto en el modo bayesiano."
            ),
        )
    else:
        factor_ars_estocastico = False

    _TASA_EXCEL = 0.1423
    tasa_descuento = st.select_slider(
        "Tasa de descuento real",
        options=sorted({round(0.02 + 0.0025 * i, 4) for i in range(93)} | {_TASA_EXCEL}),
        value=_TASA_EXCEL,
        format_func=lambda t: _pct(t, 2),
        help=(
            "Tasa real en USD (hoja Tasa del Excel v2: 14,23%). En el Sobol "
            "con el precio bayesiano (caso seco, rango 10–18%, 120 ha fijas) "
            "es la segunda variable del VAN medio (S_T ≈ 0,32, detrás del peso "
            "de la paridad w, 0,56), pero casi no pesa en el CVaR 10% (≈ 0,04): "
            "en la cola mala dominan la inversión y los costos de los primeros "
            "años, que la tasa casi no descuenta."
        ),
    )

    capex_opex_estocastico = st.checkbox(
        "OPEX estocástico", value=True,
        help=(
            "Multiplicador de OPEX (±15% triangular). Default del motor "
            "oficial: activado. El nombre del parámetro sigue siendo "
            "`capex_opex_estocastico` en el motor por compatibilidad, pero "
            "desde la resincronización del CAPEX (Fase B) ya no agrega "
            "variabilidad de riego/pozo -- esos ítems tienen precio fijo "
            "real confirmado, ver notas/PLAN_TESIS.md."
        ),
    )

    correlacionar_frio_calor = st.checkbox(
        "Correlacionar frío / calor", value=False,
        help="Cópula gaussiana frío↔calor (r≈-0,29, no significativa al 5%). Default: desactivado.",
    )

    estres_etiqueta = st.radio(
        "Estrés climático", list(_ESTRES_CLIMATICO), index=0,
        help=(
            "Desplaza cada año las medias de horas de frío y tmax de verano "
            "según la tendencia de Sen de ERA5-Land 1990-2024, extrapolada "
            "desde 2024 (año 1 del proyecto = 2027), sin renormalizar el "
            "rendimiento ideal. Puntual: frío -2,65 h/año, calor +0,088 °C/año. "
            "Extremo del IC95%: frío -11,92 h/año, calor +0,131 °C/año. Solo "
            "frío (extremo): -11,92 h/año y calor sin tendencia. La "
            "tendencia de frío no es significativa (p = 0,23): se usa solo "
            "como estrés. En el motor el calor solo penaliza veranos más "
            "frescos que la media, así que la tendencia de calor sube el "
            "rendimiento; el efecto negativo viene del frío."
        ),
    )

    quebrantos = st.checkbox(
        "Traslado de quebrantos (5 años)", value=True,
        help=(
            "Las pérdidas fiscales (base imponible < 0) se descuentan de la "
            "base de Ganancias de los 5 años siguientes, las más antiguas "
            "primero (hoja Quebrantos del Excel). Se calcula por simulación. "
            "Default: activado (caso base del Excel)."
        ),
    )

    activar_opex_pre = st.checkbox(
        "Activar OPEX pre-productivo (20 años)", value=False,
        help=(
            "Sensibilidad: el OPEX de los años 1-6 no se deduce en el año sino "
            "que se activa como costo de implantación y se amortiza en 20 "
            "años desde la primera cosecha (criterio RT 22 / NIC 16). Solo "
            "cambia la base de Ganancias, no la caja. Default: desactivado; "
            "tratamiento fiscal a confirmar con el contador."
        ),
    )

    st.divider()

    n_simulaciones = st.select_slider(
        "Nº de simulaciones", options=list(range(2000, 10001, 1000)), value=3000,
        format_func=_num,
        help="Más iteraciones = menos ruido de muestreo, pero más lento. Subir a 10.000 para precisión.",
    )

_args_comunes = dict(
    hectareas=float(hectareas),
    tasa_descuento=float(tasa_descuento),
    capex_opex_estocastico=bool(capex_opex_estocastico),
    correlacionar_frio_calor=bool(correlacionar_frio_calor),
    modo_precio=modo_precio,
    n_simulaciones=int(n_simulaciones),
    semilla=SEMILLA,
    quebrantos=bool(quebrantos),
    activar_opex_pre=bool(activar_opex_pre),
    estres_climatico=_ESTRES_CLIMATICO[estres_etiqueta],
    factor_ars_estocastico=bool(factor_ars_estocastico),
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
    p5, p10, p50, p90, p95 = np.percentile(van, [5, 10, 50, 90, 95])
    m = _metricas_van(van, tir)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VAN medio (año 20)", _usd_m(m["van_medio"]), help=_usd(m["van_medio"]))
    c2.metric("VAN mediano (año 20)", _usd_m(m["van_mediano"]), help=_usd(m["van_mediano"]))
    c3.metric(
        "Probabilidad de VAN negativo", _pct(m["prob_van_neg"], 1),
        help="Porcentaje de simulaciones en que el proyecto no alcanza la tasa exigida.",
    )
    c4.metric(
        "TIR mediana", _pct(m["tir_mediana"], 2),
        help=_ayuda_tir(m["n_tir_nan"], len(tir)),
    )

    tabla_riesgo = pd.DataFrame([{
        "P5": p5,
        "P10": p10,
        "P50": p50,
        "P90": p90,
        "P95": p95,
        f"P(pérdida > {_usd(UMBRAL_PERDIDA_USD)})": m["prob_perdida_grande"],
        "CVaR 10% (VAN medio del 10% peor)": m["cvar_10"],
    }])
    columnas_usd = [c for c in tabla_riesgo.columns if not c.startswith("P(pérdida")]
    st.dataframe(
        tabla_riesgo.style.format(
            {**{c: _usd for c in columnas_usd}, tabla_riesgo.columns[5]: _pct}
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Percentiles del VAN al año 20. Pérdida = VAN por debajo de "
        f"−US\\$ {_num(UMBRAL_PERDIDA_USD)}. CVaR 10%: promedio del VAN en el "
        "10% de simulaciones con peor resultado."
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
        f"tasa {_pct(tasa_descuento, 2)}, N={_num(n_simulaciones)}"
    )
    _mostrar(fig)


# ---------------------------------------------------------------------------
# Tab 2 — fan charts por año
# ---------------------------------------------------------------------------

def _fan_chart(
    ax, fan: pd.DataFrame, prefijo: str, titulo: str, ylabel: str,
    escala: float = 1.0, desde_año: int = 1,
):
    fan = fan.loc[fan.index >= desde_año]
    años = fan.index.to_numpy()
    ax.fill_between(
        años, fan[f"{prefijo}_p10"] / escala, fan[f"{prefijo}_p90"] / escala,
        alpha=0.25, color="#4C72B0", label="P10–P90",
    )
    ax.plot(años, fan[f"{prefijo}_p50"] / escala, color="#4C72B0", linewidth=2, label="P50")
    ax.set_title(titulo)
    ax.set_xlabel("Año del proyecto")
    ax.set_ylabel(ylabel)


with tab_fan:
    st.caption(
        "Banda P10–P90 y mediana P50 por año, sobre las "
        f"{_num(n_simulaciones)} simulaciones del escenario **{escenario}**. "
        "El VAN es acumulado y descontado; producción, precio e ingreso son anuales."
    )
    fan = res["fan"]
    trayectoria_excel = cargar_modelo_v2().trayectoria_precio(escenario, len(fan))
    primer_año_precio = int(fan.index[np.argmax(trayectoria_excel > 0)])
    paneles = (
        ("prod", "Producción", "kg secos con cáscara/ha", 1.0),
        ("precio", "Precio al productor (seco con cáscara)", "USD/kg seco (constantes 2026)", 1.0),
        ("ingreso", "Ingreso bruto anual", "millones USD", 1e6),
        ("van", "VAN neto acumulado", "millones USD", 1e6),
    )
    for fila in (paneles[:2], paneles[2:]):
        for columna, (prefijo, titulo, ylabel, escala) in zip(st.columns(2), fila):
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            if prefijo == "precio":
                # Solo años con venta: antes de la primera cosecha no hay precio.
                _fan_chart(ax, fan, prefijo, titulo, ylabel, escala, desde_año=primer_año_precio)
                años = fan.index.to_numpy()
                con_precio = años >= primer_año_precio
                ax.plot(
                    años[con_precio], trayectoria_excel[con_precio], color="#C44E52",
                    linestyle="--", linewidth=1.5, label="Nivel del Excel",
                )
            else:
                _fan_chart(ax, fan, prefijo, titulo, ylabel, escala)
            if prefijo == "van":
                ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
            ax.legend(fontsize=8)
            _mostrar(fig, columna)


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

        st.caption(
            "Índices de Sobol **totales (ST)** pre-calculados con el precio "
            "bayesiano (`src/sensibilidad.py`, `analizar_sobol_bayes`; no se "
            "recalculan en vivo), sobre dos métricas de las mismas 2.304 "
            "evaluaciones (N = 256, 1.000 simulaciones cada una): el **VAN "
            "medio** y el **CVaR 10%** (promedio del VAN en el 10% peor de las "
            "simulaciones, una medida del riesgo de cola). Una sola corrida: el "
            "escenario solo fija el peso w de la paridad, así que w se barre "
            "en [0, 1]. Barras de error = IC bootstrap (`ST_conf`). La "
            "superficie está fija en 120 ha: es una decisión de inversión, no "
            "una incertidumbre."
        )
        st.info(
            "**Clima y vecería dan ≈ 0** porque Sobol varía sus PARÁMETROS "
            "(precisión de los factores de frío y calor, persistencia de la "
            "vecería), no el clima año a año. El clima y la vecería de cada año "
            "sí están dentro de cada simulación, y su efecto se promedia en las "
            "1.000 simulaciones de cada evaluación: su peso en la dispersión "
            "del VAN se ve en la descomposición de varianza (clima ~5%, "
            "vecería ~3% en el caso base), no en estos índices."
        )
        if modo_precio != "bayesiano":
            st.info(
                "Estos índices se calcularon con el precio bayesiano: no "
                "reflejan el 'Modelo de precio' elegido en el sidebar ni sus "
                "otras opciones (caso base: venta en seco, quebrantos sí, "
                "activación no, sin estrés climático)."
            )

        columnas = st.columns(2)
        tablas = []
        for columna, (metrica, nombre) in zip(columnas, _NOMBRE_METRICA.items()):
            sub = sobol[sobol["metrica"] == metrica].sort_values("ST")
            etiquetas = sub["parametro"].map(lambda p: _NOMBRE_PARAMETRO.get(p, p))
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.barh(
                etiquetas, sub["ST"],
                xerr=sub["ST_conf"].to_numpy(), color="#4C72B0",
                error_kw=dict(ecolor="#555555", lw=1),
            )
            ax.set_xlabel("Índice de Sobol total (ST)")
            ax.set_title(f"Sensibilidad del {nombre}")
            _mostrar(fig, columna)
            tablas.append(
                sub.assign(parametro=etiquetas)
                .set_index("parametro")[["S1", "ST", "ST_conf"]]
                .rename(columns={
                    "S1": f"S1 {nombre}", "ST": f"ST {nombre}", "ST_conf": f"IC ST {nombre} (±)",
                })
            )
        tabla_sobol = (
            pd.concat(tablas, axis=1)
            .sort_values(f"ST {_NOMBRE_METRICA['van_neto_medio_usd']}", ascending=False)
            .reset_index()
            .rename(columns={"parametro": "Parámetro"})
        )
        st.dataframe(
            tabla_sobol.style.format(lambda v: _num(v, 3), subset=tabla_sobol.columns[1:]),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 4 — comparación de escenarios lado a lado
# ---------------------------------------------------------------------------

with tab_cmp:
    st.caption(
        f"Los tres escenarios de precio a los parámetros actuales del sidebar "
        f"({_num(hectareas)} ha, tasa {_pct(tasa_descuento, 2)}, "
        f"OPEX {'estocástico' if capex_opex_estocastico else 'fijo'}, "
        f"modelo de precio {_NOMBRE_PRECIO[modo_precio]}, "
        f"{'con' if quebrantos else 'sin'} traslado de quebrantos, "
        f"{'con' if activar_opex_pre else 'sin'} activación del OPEX pre-productivo, "
        f"estrés climático: {estres_etiqueta}, "
        f"N={_num(n_simulaciones)}). Cada corrida se cachea por separado."
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
    _mostrar(fig)

    metricas_cmp = {e: _metricas_van(cmp[e]["van"], cmp[e]["tir"]) for e in ESCENARIOS}
    col_perdida = f"P(pérdida > {_usd(UMBRAL_PERDIDA_USD)})"
    tabla = pd.DataFrame(
        {
            "Escenario": ESCENARIOS,
            "VAN medio (USD)": [metricas_cmp[e]["van_medio"] for e in ESCENARIOS],
            "VAN mediano (USD)": [metricas_cmp[e]["van_mediano"] for e in ESCENARIOS],
            "Probabilidad de VAN negativo": [metricas_cmp[e]["prob_van_neg"] for e in ESCENARIOS],
            col_perdida: [metricas_cmp[e]["prob_perdida_grande"] for e in ESCENARIOS],
            "CVaR 10% (USD)": [metricas_cmp[e]["cvar_10"] for e in ESCENARIOS],
            "TIR mediana": [metricas_cmp[e]["tir_mediana"] for e in ESCENARIOS],
            "Sin TIR definida": [metricas_cmp[e]["n_tir_nan"] for e in ESCENARIOS],
        }
    )
    st.dataframe(
        tabla.style.format(
            {
                "VAN medio (USD)": _num,
                "VAN mediano (USD)": _num,
                "Probabilidad de VAN negativo": _pct,
                col_perdida: _pct,
                "CVaR 10% (USD)": _num,
                "TIR mediana": lambda v: _pct(v, 2),
                "Sin TIR definida": _num,
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Probabilidad de VAN negativo: porcentaje de simulaciones en que el "
        "proyecto no alcanza la tasa exigida. TIR mediana sin las simulaciones "
        "sin TIR definida (columna de la derecha)."
    )
