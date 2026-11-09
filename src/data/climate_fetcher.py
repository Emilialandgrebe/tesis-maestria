"""Descarga y cacheo de datos climáticos históricos desde Open-Meteo Archive API.

Serie homogénea (default desde 2026-11-09, rama `clima-era5`):

- temperatura (`temperature_2m`): ERA5-Land (`models=era5_land`, 0,1°).
- precipitación y ET0 (`precipitation`, `et0_fao_evapotranspiration`):
  ERA5 (`models=era5`, 0,25°). La API no devuelve esas dos variables con
  `models=era5_land` (vienen vacías).

El default de la API ("Best Match") NO es homogéneo: hasta 2016 coincide con
ERA5-Land/ERA5, pero desde 2017 usa ECMWF IFS 9 km, con un salto de nivel
(+2 °C en la tmax de verano; ver notas/PLAN_TESIS.md). Open-Meteo recomienda
usar solo ERA5 o ERA5-Land para tendencias de varias décadas. La serie Best
Match se conserva en `clima_jocoli.parquet` solo como registro
(`fuente="best_match"`).
"""

import time
from pathlib import Path

import pandas as pd
import requests

LAT = -32.5833
LON = -68.6833
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"
HOURLY_VARS = ["temperature_2m", "precipitation", "et0_fao_evapotranspiration"]
TIMEZONE = "America/Argentina/Mendoza"
API_URL = "https://archive-api.open-meteo.com/v1/archive"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
# Serie Best Match (default de la API, sin `models`), descargada en 2026-07.
# Solo registro: no la usa el motor ni el notebook 00.
PARQUET_BEST_MATCH = RAW_DIR / "clima_jocoli.parquet"
PARQUET_POR_MODELO = {
    "era5_land": RAW_DIR / "clima_jocoli_era5_land.parquet",
    "era5": RAW_DIR / "clima_jocoli_era5.parquet",
}
# Variable -> modelo del que se toma en la serie homogénea
FUENTE_POR_VARIABLE = {
    "temperature_2m": "era5_land",
    "precipitation": "era5",
    "et0_fao_evapotranspiration": "era5",
}

# Compatibilidad: nombre anterior de la ruta de la serie Best Match.
PARQUET_PATH = PARQUET_BEST_MATCH


def fetch_climate_data(
    lat: float = LAT,
    lon: float = LON,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    timezone: str = TIMEZONE,
    max_reintentos: int = 3,
    fuente: str = "homogenea",
) -> pd.DataFrame:
    """
    Datos climáticos horarios históricos de Jocolí (1990-2024).

    Parámetros
    ----------
    fuente : {"homogenea", "era5_land", "era5", "best_match"}
        "homogenea" (default): temperatura de ERA5-Land y precipitación/ET0
        de ERA5 (ver `FUENTE_POR_VARIABLE`). "era5_land" / "era5": la serie
        completa de ese modelo (ERA5-Land viene sin precipitación ni ET0).
        "best_match": la serie histórica guardada (no homogénea, solo como
        registro).

    Cada modelo se cachea en su propio parquet de `data/raw/`; si existe, se
    lee sin llamar a la API.

    Retorna
    -------
    pd.DataFrame con índice DatetimeIndex horario y columnas
    temperature_2m, precipitation, et0_fao_evapotranspiration.
    """
    if fuente == "best_match":
        print(f"Cargando datos locales desde {PARQUET_BEST_MATCH}")
        return pd.read_parquet(PARQUET_BEST_MATCH)
    if fuente in PARQUET_POR_MODELO:
        return _serie_modelo(fuente, lat, lon, start_date, end_date, timezone, max_reintentos)
    if fuente != "homogenea":
        raise ValueError(f"fuente desconocida: {fuente!r}")

    por_modelo = {
        m: _serie_modelo(m, lat, lon, start_date, end_date, timezone, max_reintentos)
        for m in sorted(set(FUENTE_POR_VARIABLE.values()))
    }
    df = pd.DataFrame({v: por_modelo[m][v] for v, m in FUENTE_POR_VARIABLE.items()})
    if df.isna().any().any():
        raise ValueError(f"Serie homogénea con datos faltantes: {df.isna().sum().to_dict()}")
    return df


def _serie_modelo(
    modelo: str, lat, lon, start_date, end_date, timezone, max_reintentos
) -> pd.DataFrame:
    ruta = PARQUET_POR_MODELO[modelo]
    if ruta.exists():
        print(f"Cargando datos locales desde {ruta}")
        return pd.read_parquet(ruta)

    print(f"Descargando {modelo} desde Open-Meteo ({start_date} a {end_date})...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "models": modelo,
    }
    respuesta = _get_con_reintentos(API_URL, params, max_reintentos)
    df = _parsear_respuesta(respuesta, sufijo=f"_{modelo}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta)
    print(f"Datos guardados en {ruta} ({len(df):,} filas)")
    return df


def _get_con_reintentos(
    url: str, params: dict, max_reintentos: int
) -> dict:
    """Ejecuta un GET con reintentos exponenciales ante errores de conexión."""
    for intento in range(1, max_reintentos + 1):
        try:
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if intento == max_reintentos:
                raise RuntimeError(
                    f"Error al conectar con Open-Meteo tras {max_reintentos} intentos: {e}"
                ) from e
            espera = 2**intento
            print(f"Intento {intento} fallido. Reintentando en {espera}s...")
            time.sleep(espera)


def _parsear_respuesta(datos: dict, sufijo: str = "") -> pd.DataFrame:
    """Convierte la respuesta JSON de Open-Meteo en un DataFrame indexado por
    tiempo. Con `models=...` la API puede agregar el sufijo del modelo a cada
    variable; se quita."""
    hourly = datos.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("La respuesta de Open-Meteo no contiene datos horarios.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "datetime"
    if sufijo:
        df.columns = [c.removesuffix(sufijo) for c in df.columns]
    return df
