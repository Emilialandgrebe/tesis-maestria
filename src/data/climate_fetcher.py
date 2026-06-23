"""Descarga y cacheo de datos climáticos históricos desde Open-Meteo Archive API."""

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
PARQUET_PATH = RAW_DIR / "clima_jocoli.parquet"


def fetch_climate_data(
    lat: float = LAT,
    lon: float = LON,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    variables: list[str] = HOURLY_VARS,
    timezone: str = TIMEZONE,
    max_reintentos: int = 3,
) -> pd.DataFrame:
    """
    Descarga datos climáticos horarios históricos desde Open-Meteo Archive API.

    Si el archivo parquet local ya existe, lo carga directamente sin llamar a la API.
    Si no existe, descarga los datos, los guarda en parquet y los retorna.

    Parámetros
    ----------
    lat : float
        Latitud de la ubicación (por defecto: Jocolí, Mendoza).
    lon : float
        Longitud de la ubicación.
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD'.
    variables : list[str]
        Variables horarias a solicitar a la API.
    timezone : str
        Zona horaria para el índice datetime del resultado.
    max_reintentos : int
        Número máximo de reintentos ante errores de conexión.

    Retorna
    -------
    pd.DataFrame
        DataFrame con índice DatetimeIndex y una columna por variable climática.
    """
    if PARQUET_PATH.exists():
        print(f"Cargando datos locales desde {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    print(f"Descargando datos desde Open-Meteo ({start_date} → {end_date})...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": timezone,
    }

    respuesta = _get_con_reintentos(API_URL, params, max_reintentos)
    df = _parsear_respuesta(respuesta)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH)
    print(f"Datos guardados en {PARQUET_PATH} ({len(df):,} filas)")

    return df


def _get_con_reintentos(
    url: str, params: dict, max_reintentos: int
) -> dict:
    """Ejecuta un GET con reintentos exponenciales ante errores de conexión."""
    for intento in range(1, max_reintentos + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
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


def _parsear_respuesta(datos: dict) -> pd.DataFrame:
    """Convierte la respuesta JSON de Open-Meteo en un DataFrame indexado por tiempo."""
    hourly = datos.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("La respuesta de Open-Meteo no contiene datos horarios.")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "datetime"
    return df
