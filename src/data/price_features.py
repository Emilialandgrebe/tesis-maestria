"""Serie anual de precio del pistacho y diagnóstico de autocorrelación.

Convierte el índice mensual de FRED (con huecos reales de reporte) en una
serie anual, y calibra un AR(1) sobre los retornos logarítmicos — ver
notas/plan_precio_historico.md (Paso 2) para la justificación de por qué
un AR(1) sobre retornos (no sobre el nivel) y por qué la sensibilidad de
composición a 20 años importa más que la significancia marginal del
coeficiente.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller


@dataclass
class DiagnosticoAutocorrelacion:
    """Diagnóstico de autocorrelación e independencia de los retornos anuales."""

    n_obs: int
    acf_lag1: float
    ljung_box_pvalor_lag1: float
    adf_estadistico: float
    adf_pvalor: float
    ar1_phi: float
    ar1_pvalor: float
    conclusion: str


@dataclass
class ParametrosAR1:
    """Parámetros calibrados de un AR(1) sobre los retornos log anuales."""

    c: float
    phi: float
    sigma_eps: float
    n_obs: int
    periodo: str


def calcular_serie_anual(df_mensual: pd.DataFrame) -> pd.Series:
    """
    Convierte el índice de precio mensual en una serie anual.

    Toma el último valor reportado de cada año (ignorando meses sin dato),
    reindexa a una grilla anual completa (todos los años entre el mínimo y
    el máximo) y rellena hacia adelante los años sin ningún reporte — la
    serie de FRED es de un mercado angosto, con años enteros sin dato.

    Parámetros
    ----------
    df_mensual : pd.DataFrame
        DataFrame con columnas `fecha` (datetime) e `indice_precio` (float),
        formato de salida de `fetch_price_data()`.

    Retorna
    -------
    pd.Series
        Serie indexada por año (int) con el índice de precio anual.
    """
    mensual = df_mensual.set_index("fecha")["indice_precio"].dropna()
    anual = mensual.groupby(mensual.index.year).last()

    grilla = pd.RangeIndex(anual.index.min(), anual.index.max() + 1)
    anual = anual.reindex(grilla).ffill()

    anual.index.name = "año"
    anual.name = "indice_precio"
    return anual


def calcular_retornos_log(serie_anual: pd.Series) -> pd.Series:
    """
    Calcula los retornos logarítmicos anuales: r_t = log(P_t) - log(P_t-1).

    Parámetros
    ----------
    serie_anual : pd.Series
        Serie anual de precio, salida de `calcular_serie_anual()`.

    Retorna
    -------
    pd.Series
        Retornos logarítmicos, indexados por año (un año menos que la
        serie de precio, ya que el primer año no tiene retorno anterior).
    """
    retornos = np.log(serie_anual).diff().dropna()
    retornos.name = "retorno_log"
    return retornos


def _ajustar_ar1(retornos: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Ajusta r_t = c + phi * r_(t-1) + eps por OLS."""
    y = retornos.iloc[1:]
    x = sm.add_constant(retornos.iloc[:-1].values)
    return sm.OLS(y.values, x).fit()


def diagnosticar_autocorrelacion(serie_anual: pd.Series) -> DiagnosticoAutocorrelacion:
    """
    Diagnostica si los retornos logarítmicos anuales son independientes.

    Corre tres chequeos complementarios sobre `serie_anual`:
    - ADF sobre el NIVEL de precio (no sobre los retornos): testea si el
      precio tiene raíz unitaria, i.e. si un supuesto tipo GBM es plausible.
    - ACF y Ljung-Box en lag 1 sobre los RETORNOS: testea autocorrelación
      serial directamente.
    - Un AR(1) por OLS sobre los RETORNOS: mismo modelo que `calibrar_ar1()`,
      para reportar el coeficiente y su significancia en un solo lugar.

    Parámetros
    ----------
    serie_anual : pd.Series
        Serie anual de precio, salida de `calcular_serie_anual()`.

    Retorna
    -------
    DiagnosticoAutocorrelacion
    """
    retornos = calcular_retornos_log(serie_anual)

    adf_estadistico, adf_pvalor = adfuller(serie_anual.values)[:2]

    acf_lag1 = acf(retornos.values, nlags=1, fft=False)[1]
    ljung_box = acorr_ljungbox(retornos.values, lags=[1], return_df=True)
    ljung_box_pvalor_lag1 = float(ljung_box["lb_pvalue"].iloc[0])

    modelo = _ajustar_ar1(retornos)
    ar1_phi = float(modelo.params[1])
    ar1_pvalor = float(modelo.pvalues[1])

    n_obs = len(retornos)

    if adf_pvalor > 0.05 and ar1_pvalor > 0.05:
        conclusion = (
            "El ADF no rechaza raíz unitaria en el nivel (p="
            f"{adf_pvalor:.3f}) y el AR(1) no es significativo al 5% (p="
            f"{ar1_pvalor:.3f}), pero phi={ar1_phi:.4f} es económicamente "
            "relevante por el efecto de composición a 20 años (ver "
            "notas/plan_precio_historico.md) — no se descarta autocorrelación "
            "solo por falta de significancia estadística con N chico."
        )
    else:
        conclusion = (
            f"ADF p={adf_pvalor:.3f}, AR(1) p={ar1_pvalor:.3f} "
            f"(phi={ar1_phi:.4f})."
        )

    return DiagnosticoAutocorrelacion(
        n_obs=n_obs,
        acf_lag1=float(acf_lag1),
        ljung_box_pvalor_lag1=ljung_box_pvalor_lag1,
        adf_estadistico=float(adf_estadistico),
        adf_pvalor=float(adf_pvalor),
        ar1_phi=ar1_phi,
        ar1_pvalor=ar1_pvalor,
        conclusion=conclusion,
    )


def calibrar_ar1(serie_anual: pd.Series) -> ParametrosAR1:
    """
    Calibra un AR(1) sobre los retornos logarítmicos anuales por OLS.

    r_t = c + phi * r_(t-1) + eps_t

    `sigma_eps` es el desvío estándar de los RESIDUOS del ajuste, no el
    desvío incondicional de los retornos — son distintos salvo que phi=0.

    Parámetros
    ----------
    serie_anual : pd.Series
        Serie anual de precio, salida de `calcular_serie_anual()`.

    Retorna
    -------
    ParametrosAR1
    """
    retornos = calcular_retornos_log(serie_anual)
    modelo = _ajustar_ar1(retornos)

    c = float(modelo.params[0])
    phi = float(modelo.params[1])
    sigma_eps = float(modelo.resid.std(ddof=1))
    n_obs = len(retornos) - 1

    periodo = f"{serie_anual.index.min()}-{serie_anual.index.max()}"

    return ParametrosAR1(
        c=c,
        phi=phi,
        sigma_eps=sigma_eps,
        n_obs=n_obs,
        periodo=periodo,
    )
