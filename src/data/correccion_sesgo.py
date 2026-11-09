"""Corrección de sesgo de niveles de la serie climática contra normales SMN.

La serie homogénea (temperatura ERA5-Land, precipitación y ET0 ERA5; ver
`climate_fetcher.py`) tiene tendencias confiables pero niveles sesgados en
Jocolí: ERA5 sobreestima la lluvia ~2,9 veces y ERA5-Land subestima la tmax
de verano ~2,5 °C. Se corrigen los niveles por escalado de medias contra las
normales climatológicas 1991-2020 del SMN, conservando la variabilidad
interanual y las tendencias:

- precipitación: factor multiplicativo anual único
  (normal anual / media anual de ERA5 en 1991-2020). Conserva el CV y el
  test de Mann-Kendall (la serie anual queda multiplicada por una
  constante).
- tmax diaria: corrimiento aditivo mensual
  (normal de tmax media del mes / media de la tmax diaria de ERA5-Land de
  ese mes en 1991-2020). Conserva el desvío y la pendiente de Sen.
- ET0: sin corregir (las normales no la incluyen; Hargreaves con la tmax y
  la tmin de la normal de Mendoza Aero da ~1.451 mm contra 1.413 de ERA5).
- tmin / horas de frío / heladas / GDD: sin corregir (la tmin de ERA5-Land
  no tiene sesgo contra la normal).

Solo la usa el notebook 00 (capítulo climático). El motor NO la usa: el
factor de calor está centrado en la media de su propia serie, así que un
corrimiento aditivo de la tmax no cambia ninguna penalidad.

Fuente: WMO Climatological Standard Normals for 1991-2020, datos provistos
por el Servicio Meteorológico Nacional (Argentina), NCEI Accession 0253808,
https://www.nodc.noaa.gov/archive/arc0216/0253808/3.3/data/0-data/Region-3-WMO-Normals-9120/Argentina/
(archivos MENDOZA_AERO_87418.csv y SAN_MARTIN_87416.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODO_NORMAL = ("1991", "2020")

# Normales OMM 1991-2020 del SMN. tmax_mensual: Daily_Maximum_Temperature,
# Mean (°C), enero..diciembre. precipitacion_anual: Precipitation_Total, Sum
# (mm), anual.
NORMALES_SMN_1991_2020 = {
    87418: {
        "nombre": "Mendoza Aero",
        "lat": -(32 + 50 / 60 + 37 / 3600),
        "lon": -(68 + 47 / 60 + 46 / 3600),
        "altura_m": 704,
        "precipitacion_anual": 238.1,
        "tmax_mensual": (32.8, 31.0, 28.1, 23.3, 18.7, 15.9, 15.3, 18.6, 21.8, 25.7, 29.2, 32.0),
        "tmin_mensual": (19.3, 17.8, 15.7, 10.8, 6.8, 2.9, 2.0, 4.3, 7.8, 11.9, 15.4, 18.1),
    },
    87416: {
        "nombre": "San Martín (Mendoza)",
        "lat": -(33 + 4 / 60 + 23 / 3600),
        "lon": -(68 + 27 / 60 + 53 / 3600),
        "altura_m": 653,
        "precipitacion_anual": 262.4,
        "tmax_mensual": (32.9, 31.2, 28.5, 23.6, 19.2, 16.2, 15.8, 19.0, 22.2, 25.9, 29.4, 32.2),
        "tmin_mensual": (18.3, 16.9, 15.0, 10.5, 6.6, 3.0, 1.9, 4.1, 7.3, 11.1, 14.4, 17.0),
    },
}
ESTACION_REFERENCIA = 87418  # Mendoza Aero, la más cercana (~31 km)
ESTACION_CONTROL = 87416     # San Martín (~58 km)


def _periodo(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[PERIODO_NORMAL[0]:PERIODO_NORMAL[1]]


def factor_precipitacion(df: pd.DataFrame, estacion: int = ESTACION_REFERENCIA) -> float:
    """Normal anual de la estación / precipitación anual media del reanálisis
    en 1991-2020."""
    p = _periodo(df)["precipitation"]
    media = p.groupby(p.index.year).sum().mean()
    return NORMALES_SMN_1991_2020[estacion]["precipitacion_anual"] / media


def _tmax_diaria(df: pd.DataFrame) -> pd.Series:
    t = df["temperature_2m"]
    tmax = t.groupby(t.index.date).max()
    tmax.index = pd.to_datetime(tmax.index)
    return tmax.rename("tmax_diaria")


def delta_tmax_mensual(df: pd.DataFrame, estacion: int = ESTACION_REFERENCIA) -> pd.Series:
    """Normal de tmax media de cada mes menos la media de la tmax diaria del
    reanálisis en ese mes, 1991-2020. Índice: mes 1..12."""
    tmax = _tmax_diaria(_periodo(df))
    media = tmax.groupby(tmax.index.month).mean()
    normal = pd.Series(NORMALES_SMN_1991_2020[estacion]["tmax_mensual"], index=range(1, 13))
    return (normal - media).rename("delta_tmax")


def tmax_diaria_corregida(df: pd.DataFrame, estacion: int = ESTACION_REFERENCIA) -> pd.Series:
    """Tmax diaria del reanálisis más el corrimiento del mes, para todo el
    período del DataFrame."""
    tmax = _tmax_diaria(df)
    delta = delta_tmax_mensual(df, estacion)
    return (tmax + delta.reindex(tmax.index.month).to_numpy()).rename("tmax_diaria")


def corregir_precipitacion(df: pd.DataFrame, estacion: int = ESTACION_REFERENCIA) -> pd.DataFrame:
    """Copia del DataFrame horario con la precipitación multiplicada por el
    factor anual. Temperatura y ET0 sin cambios."""
    out = df.copy()
    out["precipitation"] = out["precipitation"] * factor_precipitacion(df, estacion)
    return out


def resumen_correccion(df: pd.DataFrame, estacion: int = ESTACION_REFERENCIA) -> pd.DataFrame:
    """Tabla con el factor de lluvia y el corrimiento de tmax por mes."""
    n = NORMALES_SMN_1991_2020[estacion]
    delta = delta_tmax_mensual(df, estacion)
    return pd.DataFrame({
        "tmax_normal_c": n["tmax_mensual"],
        "tmax_reanalisis_c": np.asarray(n["tmax_mensual"]) - delta.to_numpy(),
        "delta_tmax_c": delta.to_numpy(),
    }, index=pd.Index(range(1, 13), name="mes")).assign(
        factor_precipitacion=factor_precipitacion(df, estacion)
    )
