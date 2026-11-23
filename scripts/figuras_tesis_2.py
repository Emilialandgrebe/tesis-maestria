"""Figuras y tablas de la tesis, segunda tanda (caso base: venta en verde).

Solo lee el modelo y genera salidas; no cambia ningún supuesto.

Figuras (PNG, 300 dpi, coma decimal) en outputs/figuras/:
  01_precio_real_fred.png          serie anual del precio real (FRED deflactada)
  02_fan_chart_precio_verde.png    precio simulado en verde: AR(1) niveles y bayesiano
  03_posterior_phi_sigma.png       histogramas de la posterior de phi y sigma
  04_series_climaticas.png         horas de frío y tmax de verano, con Sen
  05_produccion_y_flujo.png        curva de producción y flujo de fondos (Excel)
  06_histogramas_van.png           VAN de los tres escenarios (AR(1))
  07_descomposicion_varianza.png   fracción de la varianza del VAN por fuente
  08_penalizaciones_frio_calor.png funciones de transferencia frío y calor

Tablas (CSV) en outputs/:
  clima_estadisticas_tendencias.csv  estadísticas + MK modificado + Sen
  estres_climatico_verde.csv         VAN medio, P(VAN<0) y CVaR 10 % por variante
  sobol_bayes_verde_s1.csv           S1 (y ST) del Sobol bayesiano en verde

Uso:
    python scripts/figuras_tesis_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymannkendall as mk  # noqa: E402
from matplotlib import ticker  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.costos import ParametrosCostos  # noqa: E402
from src.data.climate_features import calcular_deficit_hidrico, calcular_horas_frio  # noqa: E402
from src.data.climate_fetcher import fetch_climate_data  # noqa: E402
from src.data.correccion_sesgo import corregir_precipitacion, tmax_diaria_corregida  # noqa: E402
from src.monte_carlo import (  # noqa: E402
    _CALOR_VERANO_MEDIA,
    _CALOR_VERANO_STD,
    _HORAS_FRIO_MEDIA,
    _HORAS_FRIO_STD,
    PENDIENTES_ESTRES_CLIMATICO,
    ParametrosMC,
    _media_factor_calor,
    _media_factor_frio,
    resumen_financiero,
    run_monte_carlo,
)
from src.precio_bayesiano import (  # noqa: E402
    ParametrosPrecioBayes,
    cargar_posterior,
    simulate_prices_bayes_antitetico,
)
from src.precio_estocastico import ParametrosPrecioAR1, simulate_prices_ar1_antitetico  # noqa: E402

DIR_FIG = RAIZ / "outputs" / "figuras"
DIR_OUT = RAIZ / "outputs"
DPI = 300
SEMILLA = 42
N_MC = 10_000
AZUL, GRIS, ROJO, VERDE = "#2C5F8A", "#404040", "#B5432A", "#4A7A41"
ESCENARIOS = ("pesimista", "base", "optimista")
COLOR_ESC = {"pesimista": ROJO, "base": AZUL, "optimista": VERDE}


def coma(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def fmt_eje(d: int = 0, escala: float = 1.0):
    return ticker.FuncFormatter(lambda v, _: coma(v / escala, d))


def guardar(fig, nombre: str) -> Path:
    ruta = DIR_FIG / nombre
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return ruta


def fuente(fig, texto: str) -> None:
    fig.text(0.99, -0.01, texto, ha="right", va="top", fontsize=7.5, color="#555555")


# ---------------------------------------------------------------- 1. FRED
def fig_precio_fred() -> Path:
    from scripts.calibrar_precio_bayesiano import serie_log_real

    real = np.exp(serie_log_real())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(real.index, real.values, color=AZUL, marker="o", markersize=4, linewidth=1.8)
    ax.axhline(real.mean(), color=GRIS, linestyle=":", linewidth=1,
               label=f"Media 1991–2026 ({coma(real.mean(), 1)})")
    ax.set_xlabel("Año")
    ax.set_ylabel("Índice de precio real\n(dic-1991 = 100 nominal; USD de 2026)")
    ax.set_title("Precio real del pistacho al productor en EE.UU., serie anual 1991–2026")
    ax.yaxis.set_major_formatter(fmt_eje(0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)
    fuente(fig, "Fuente: FRED, serie WPU01190106 (PPI Farm Products: Pistachios, BLS), deflactada por "
                "CPI-U sin ajuste estacional (CPIAUCNS); último mes reportado de cada año.")
    return guardar(fig, "01_precio_real_fred.png")


# ---------------------------------------------------------------- 2. Fan chart
def _banda(ax, precios: np.ndarray, color: str, titulo: str) -> None:
    años = np.arange(1, precios.shape[1] + 1)
    venta = años >= 7
    a, p = años[venta], precios[:, venta]
    q = np.percentile(p, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(a, q[0], q[4], color=color, alpha=0.18, label="P5–P95")
    ax.fill_between(a, q[1], q[3], color=color, alpha=0.35, label="P25–P75")
    ax.plot(a, q[2], color=color, linewidth=2, label="Mediana")
    ax.plot(a, p.mean(axis=0), color="black", linestyle="--", linewidth=1.2, label="Media")
    ax.axvspan(0, 6.5, color="#DDDDDD", alpha=0.5)
    ax.text(3.25, 0.3, "Sin venta\n(años 1–6)", ha="center", va="bottom", fontsize=8, color="#555555")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(fmt_eje(1))
    ax.set_xlabel("Año del proyecto")
    ax.set_title(titulo, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def fig_fan_chart() -> Path:
    n = 20_000
    ar1 = simulate_prices_ar1_antitetico(
        n, 20, ParametrosPrecioAR1(escenario="base", venta_verde=True), np.random.default_rng(SEMILLA)
    )
    bayes = simulate_prices_bayes_antitetico(
        n, 20, ParametrosPrecioBayes(escenario="base", venta_verde=True), np.random.default_rng(SEMILLA)
    )
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    _banda(ejes[0], ar1, AZUL, "AR(1) en niveles (revierte al nivel de 4 USD/kg verde)")
    _banda(ejes[1], bayes, "#7A4A9A", "Bayesiano (phi y sigma de la posterior; nivel: mezcla 2,5 / 5,5, w = 0,5)")
    ejes[0].set_ylabel("Precio en verde (USD/kg verde, constantes 2026)")
    ejes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Precio simulado en verde, escenario base (20.000 trayectorias antitéticas)", fontsize=11)
    fig.tight_layout()
    return guardar(fig, "02_fan_chart_precio_verde.png")


# ---------------------------------------------------------------- 3. Posterior
def fig_posterior() -> Path:
    post = cargar_posterior()
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, nombre, color in (
        (ejes[0], "phi", "Persistencia φ", AZUL),
        (ejes[1], "sigma", "Volatilidad σ (desvío del shock log anual)", "#7A4A9A"),
    ):
        x = post[col]
        q05, q50, q95 = x.quantile([0.05, 0.5, 0.95])
        ax.hist(x, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(q50, color="black", linewidth=1.5, label=f"Mediana {coma(q50, 3)}")
        for q in (q05, q95):
            ax.axvline(q, color="black", linestyle="--", linewidth=1)
        ax.plot([], [], color="black", linestyle="--", label=f"IC 90 % [{coma(q05, 2)}; {coma(q95, 2)}]")
        ax.set_xlabel(nombre)
        ax.set_ylabel("Frecuencia (sorteos)")
        ax.xaxis.set_major_formatter(fmt_eje(2))
        ax.legend(fontsize=8)
    ejes[0].set_title(f"P(φ > 0,95) = {coma((post['phi'] > 0.95).mean() * 100, 1)} %", fontsize=10)
    fig.suptitle("Distribución a posteriori del AR(1) del log del precio real (4.000 sorteos)", fontsize=11)
    fuente(fig, "Prior: φ ~ U(0, 1], σ ~ HalfNormal(0,3); posterior conjugada exacta + remuestreo por "
                "importancia (scripts/calibrar_precio_bayesiano.py). Datos: FRED real 1991–2026.")
    fig.tight_layout()
    return guardar(fig, "03_posterior_phi_sigma.png")


# ---------------------------------------------------------------- 4 y 9. Clima
def series_climaticas() -> dict[str, pd.Series]:
    """Igual que el notebook 00: lluvia y tmax corregidas (normales SMN
    1991-2020, Mendoza Aero); horas de frío y ET0 sin corregir."""
    crudo = fetch_climate_data()
    df = corregir_precipitacion(crudo)
    tmax = tmax_diaria_corregida(crudo)
    ef = tmax[tmax.index.month.isin([1, 2])]
    año = df.index.year
    return {
        "horas_frio": calcular_horas_frio(df).astype(float),
        "tmax_verano": ef.groupby(ef.index.year).mean(),
        "precipitacion": df["precipitation"].groupby(año).sum(),
        "et0": df["et0_fao_evapotranspiration"].groupby(año).sum(),
        "deficit_hidrico": calcular_deficit_hidrico(df),
    }


ETIQUETAS_CLIMA = {
    "horas_frio": ("Horas de frío (< 7 °C, may–sep)", "h"),
    "tmax_verano": ("Tmax media de verano (ene–feb)", "°C"),
    "precipitacion": ("Precipitación anual", "mm"),
    "et0": ("ET0 anual (FAO-56)", "mm"),
    "deficit_hidrico": ("Déficit hídrico (ET0 − precipitación)", "mm"),
}


def tabla_clima(series: dict[str, pd.Series]) -> pd.DataFrame:
    filas = []
    for clave, s in series.items():
        s = s.loc[1990:2024]
        r = mk.hamed_rao_modification_test(s.values)
        sen = theilslopes(s.values, s.index.astype(float))
        nombre, unidad = ETIQUETAS_CLIMA[clave]
        filas.append({
            "variable": nombre, "unidad": unidad, "n_años": len(s),
            "media": s.mean(), "desvio": s.std(ddof=1), "minimo": s.min(), "maximo": s.max(),
            "año_minimo": int(s.idxmin()), "año_maximo": int(s.idxmax()),
            "mk_tau": r.Tau, "mk_z": r.z, "mk_p": r.p, "mk_tendencia": r.trend,
            "sen_por_año": sen.slope, "sen_por_decada": 10 * sen.slope,
            "sen_ic95_inf_decada": 10 * sen.low_slope, "sen_ic95_sup_decada": 10 * sen.high_slope,
        })
    return pd.DataFrame(filas)


def fig_clima(series: dict[str, pd.Series], tabla: pd.DataFrame) -> Path:
    fig, ejes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, clave, color in ((ejes[0], "horas_frio", AZUL), (ejes[1], "tmax_verano", ROJO)):
        s = series[clave].loc[1990:2024]
        x = s.index.astype(float)
        fila = tabla[tabla["variable"] == ETIQUETAS_CLIMA[clave][0]].iloc[0]
        sen = theilslopes(s.values, x)
        xc, yc = np.median(x), np.median(s.values)
        ax.plot(x, s.values, color=color, marker="o", markersize=4, linewidth=1.6, label="Serie anual")
        ax.fill_between(x, yc + sen.low_slope * (x - xc), yc + sen.high_slope * (x - xc),
                        color=GRIS, alpha=0.12, label="IC 95 % de la pendiente de Sen")
        unidad = ETIQUETAS_CLIMA[clave][1]
        d = 1 if clave == "horas_frio" else 2
        ax.plot(x, yc + sen.slope * (x - xc), color=GRIS, linestyle="--", linewidth=1.8,
                label=(f"Sen: {coma(fila['sen_por_decada'], d)} {unidad}/década; "
                       f"MK modificado p = {coma(fila['mk_p'], 3)}"))
        ax.axhline(s.mean(), color=color, linestyle=":", linewidth=0.9)
        ax.set_ylabel(f"{ETIQUETAS_CLIMA[clave][0]} ({unidad})")
        ax.yaxis.set_major_formatter(fmt_eje(0 if clave == "horas_frio" else 1))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right" if clave == "horas_frio" else "upper left")
    ejes[1].set_xlabel("Año")
    ejes[0].set_title("Series climáticas de Jocolí (Lavalle, Mendoza), 1990–2024", fontsize=11)
    fuente(fig, "Fuente: ERA5-Land (temperatura) vía Open-Meteo. Tmax con nivel corregido por las normales "
                "SMN 1991–2020 (Mendoza Aero). Mann-Kendall con corrección de Hamed y Rao.")
    fig.tight_layout()
    return guardar(fig, "04_series_climaticas.png")


# ---------------------------------------------------------------- 5. Producción y flujo
def fig_produccion_flujo() -> Path:
    anual = pd.read_csv(RAIZ / "data" / "modelo_v2" / "anual.csv")
    flujo = pd.read_csv(RAIZ / "data" / "modelo_v2" / "referencia_flujo.csv")
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = ejes[0]
    ax.bar(anual["anio"], anual["rendimiento_pct_plena"] * 100, color=VERDE, alpha=0.85)
    ax.set_xlabel("Año del proyecto")
    ax.set_ylabel("Rendimiento (% de plena producción)")
    ax.set_title("Curva de producción (100 % = 3.000 kg verdes/ha)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax = ejes[1]
    f = flujo["flujo_evaluacion_usd"] / 1e6
    ax.bar(flujo["anio"], f, color=np.where(f >= 0, AZUL, ROJO), alpha=0.85, label="Flujo de fondos anual")
    ax.plot(flujo["anio"], f.cumsum(), color="black", marker="o", markersize=3.5, linewidth=1.6,
            label="Flujo acumulado (sin descontar)")
    ax.plot(flujo["anio"], flujo["flujo_descontado_usd"].cumsum() / 1e6, color=GRIS, linestyle="--",
            linewidth=1.4, label="Acumulado descontado (14,23 % real)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Año del proyecto")
    ax.set_ylabel("Millones de USD (constantes 2026)")
    ax.set_title("Flujo de fondos del caso base (120 ha, venta en verde a 4 USD/kg)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(fmt_eje(1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fuente(fig, "Fuente: modelo financiero v2 (Excel), hojas Supuestos y Flujo; año 20 incluye el valor "
                "residual de la tierra. VAN = −2.836.458 USD; TIR real = 1,42 %.")
    fig.tight_layout()
    return guardar(fig, "05_produccion_y_flujo.png")


# ---------------------------------------------------------------- 6. Histogramas VAN
def van_escenario(escenario: str, **kw) -> np.ndarray:
    params = ParametrosMC(n_simulaciones=N_MC, escenario=escenario, semilla=SEMILLA, **kw)
    costos = ParametrosCostos(hectareas=params.hectareas)
    return resumen_financiero(run_monte_carlo(params, costos), costos)["van_neto_usd"].to_numpy()


def fig_histogramas_van() -> Path:
    vans = {e: van_escenario(e) for e in ESCENARIOS}
    lo = min(v.min() for v in vans.values())
    hi = max(v.max() for v in vans.values())
    bins = np.linspace(lo, hi, 80)
    fig, ax = plt.subplots(figsize=(10, 5))
    for e, v in vans.items():
        ax.hist(v / 1e6, bins=bins / 1e6, color=COLOR_ESC[e], alpha=0.45,
                label=(f"{e.capitalize()}: media {coma(v.mean() / 1e6, 2)} M; "
                       f"P(VAN<0) = {coma((v < 0).mean() * 100, 1)} %"))
        ax.axvline(v.mean() / 1e6, color=COLOR_ESC[e], linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("VAN al año 20 (millones de USD constantes 2026)")
    ax.set_ylabel("Frecuencia (simulaciones)")
    ax.set_title("Distribución del VAN por escenario de precio (AR(1) en niveles, 10.000 simulaciones)",
                 fontsize=10)
    ax.xaxis.set_major_formatter(fmt_eje(1))
    ax.yaxis.set_major_formatter(fmt_eje(0))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fuente(fig, "Venta en verde a la procesadora, 120 ha, tasa real 14,23 %; precio, clima, vecería, falla "
                "de plantas y OPEX estocásticos; líneas verticales: VAN medio.")
    return guardar(fig, "06_histogramas_van.png")


# ---------------------------------------------------------------- 7. Descomposición
def fig_descomposicion() -> Path:
    d = pd.read_csv(DIR_OUT / "descomposicion_varianza_base_fracciones.csv")
    nombres = {"precio": "Precio", "clima": "Clima", "veceria": "Vecería",
               "plantas": "Falla de plantas", "opex": "OPEX"}
    d = d.sort_values("fraccion_al_quitar", ascending=False)
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for desplaz, col, etiqueta, color in (
        (-w / 2, "fraccion_al_quitar", "Al quitar la fuente (1 − Var sin / Var total)", AZUL),
        (w / 2, "fraccion_sola", "Con la fuente sola (Var sola / Var total)", "#8FB3D9"),
    ):
        val = d[col].to_numpy() * 100
        err = np.vstack([val - d[f"{col}_ic95_inf"] * 100, d[f"{col}_ic95_sup"] * 100 - val])
        ax.bar(x + desplaz, val, w, yerr=err, color=color, label=etiqueta,
               error_kw=dict(ecolor="#555555", capsize=3, lw=1))
        for xi, v, e in zip(x + desplaz, val, err[1]):
            ax.text(xi, v + e + 1.0, coma(v, 1), ha="center", fontsize=8)
    ax.set_xticks(x, d["fuente"].map(nombres))
    ax.set_ylabel("Fracción de la varianza del VAN (%)")
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(fmt_eje(0))
    ax.set_title("Descomposición de la varianza del VAN por fuente de riesgo (caso base verde)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fuente(fig, "10.000 simulaciones antitéticas por corrida; barras de error: IC 95 % bootstrap "
                "(scripts/descomponer_varianza.py).")
    return guardar(fig, "07_descomposicion_varianza.png")


# ---------------------------------------------------------------- 8. Penalizaciones
def fig_penalizaciones() -> Path:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    casos = (
        (ejes[0], np.linspace(0, 1600, 801), _media_factor_frio, _HORAS_FRIO_MEDIA, _HORAS_FRIO_STD,
         200.0, "Horas de frío acumuladas (< 7 °C, may–sep)", "h", AZUL, 0),
        (ejes[1], np.linspace(15, 36, 841), _media_factor_calor, _CALOR_VERANO_MEDIA, _CALOR_VERANO_STD,
         7.3, "Tmax media de verano, ene–feb (°C, ERA5-Land)", "°C", ROJO, 2),
    )
    for ax, x, f, media, desvio, ancho, xlabel, unidad, color, d in casos:
        ax.axvspan(media - desvio, media + desvio, color=color, alpha=0.10,
                   label=f"Media ± 1 desvío histórico ({coma(media, d)} ± {coma(desvio, d)} {unidad})")
        ax.plot(x, f(x), color=color, linewidth=2.2, label="Factor esperado de rendimiento")
        ax.axvline(media, color=color, linestyle="--", linewidth=1)
        ax.axvline(media - ancho, color=GRIS, linestyle=":", linewidth=1,
                   label=f"Umbral severo ({coma(media - ancho, d)} {unidad})")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.08)
        ax.xaxis.set_major_formatter(fmt_eje(0))
        ax.yaxis.set_major_formatter(fmt_eje(1))
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="lower right")
    ejes[0].set_ylabel("Factor de rendimiento (media)")
    ejes[0].set_title("Penalización por falta de frío invernal", fontsize=10)
    ejes[1].set_title("Penalización por verano más fresco que la media", fontsize=10)
    fuente(fig, "Lineal por tramos (1,00 → 0,70 en la zona moderada; 0,70 → 0,40 en la severa), centrada en "
                "la media del sitio. Sobre esta media se sortea un factor Beta con precisión κ = 15.")
    fig.tight_layout()
    return guardar(fig, "08_penalizaciones_frio_calor.png")


# ---------------------------------------------------------------- 10. Estrés climático
def tabla_estres() -> pd.DataFrame:
    filas = []
    variantes = [("Sin estrés (caso base)", dict())] + [
        (v, dict(estres_climatico=True, estres_variante=v)) for v in PENDIENTES_ESTRES_CLIMATICO
    ]
    for nombre, kw in variantes:
        v = van_escenario("base", **kw)
        n_peores = int(np.ceil(0.10 * len(v)))
        pend = PENDIENTES_ESTRES_CLIMATICO.get(kw.get("estres_variante"), {"horas_frio": 0.0, "calor_verano": 0.0})
        filas.append({
            "variante": nombre,
            "pendiente_frio_h_año": pend["horas_frio"],
            "pendiente_calor_c_año": pend["calor_verano"],
            "van_medio_usd": v.mean(),
            "ic95_usd": 1.96 * v.std(ddof=1) / np.sqrt(len(v)),
            "p_van_negativo": (v < 0).mean(),
            "cvar_10_usd": np.sort(v)[:n_peores].mean(),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    rutas = [fig_precio_fred(), fig_fan_chart(), fig_posterior()]
    series = series_climaticas()
    clima = tabla_clima(series)
    clima.to_csv(DIR_OUT / "clima_estadisticas_tendencias.csv", index=False)
    rutas += [fig_clima(series, clima), fig_produccion_flujo(), fig_histogramas_van(),
              fig_descomposicion(), fig_penalizaciones()]
    estres = tabla_estres()
    estres.to_csv(DIR_OUT / "estres_climatico_verde.csv", index=False)
    sobol = pd.read_parquet(RAIZ / "data" / "processed" / "sobol_indices_bayes.parquet")
    sobol.to_csv(DIR_OUT / "sobol_bayes_verde_s1.csv", index=False)
    print(clima.round(4).to_string(index=False))
    print(estres.round(4).to_string(index=False))
    print(sobol.round(4).to_string(index=False))
    for r in rutas:
        print(r)
