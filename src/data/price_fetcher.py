"""Descarga y cacheo del índice de precio del pistacho desde FRED."""

import io
import time
from pathlib import Path

import pandas as pd
import requests

SERIE_ID = "WPU01190106"
API_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PARQUET_PATH = RAW_DIR / "precio_pistacho_fred.parquet"


def fetch_price_data(
    serie_id: str = SERIE_ID,
    max_reintentos: int = 3,
) -> pd.DataFrame:
    """
    Descarga el índice de precio del pistacho (PPI, BLS) desde FRED.

    Serie WPU01190106: Producer Price Index, Farm Products: Pistachios.
    Mercado angosto — la serie tiene meses sin dato reportado por BLS.

    Si el archivo parquet local ya existe, lo carga directamente sin llamar a la API.
    Si no existe, descarga los datos, los guarda en parquet y los retorna.

    Parámetros
    ----------
    serie_id : str
        Identificador de la serie en FRED.
    max_reintentos : int
        Número máximo de reintentos ante errores de conexión.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas `fecha` (datetime, mensual) e `indice_precio`
        (float, NaN en meses sin dato reportado).
    """
    if PARQUET_PATH.exists():
        print(f"Cargando datos locales desde {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    print(f"Descargando serie {serie_id} desde FRED...")
    params = {"id": serie_id}

    respuesta = _get_con_reintentos(API_URL, params, max_reintentos)
    df = _parsear_respuesta(respuesta, serie_id)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH)
    print(f"Datos guardados en {PARQUET_PATH} ({len(df):,} filas)")

    return df


def _get_con_reintentos(url: str, params: dict, max_reintentos: int) -> str:
    """Ejecuta un GET con reintentos exponenciales ante errores de conexión."""
    for intento in range(1, max_reintentos + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            if intento == max_reintentos:
                raise RuntimeError(
                    f"Error al conectar con FRED tras {max_reintentos} intentos: {e}"
                ) from e
            espera = 2**intento
            print(f"Intento {intento} fallido. Reintentando en {espera}s...")
            time.sleep(espera)


def _parsear_respuesta(texto: str, serie_id: str) -> pd.DataFrame:
    """Convierte el CSV de FRED en un DataFrame con `fecha` e `indice_precio`."""
    df = pd.read_csv(io.StringIO(texto), na_values=["", "."])
    if "observation_date" not in df.columns or serie_id not in df.columns:
        raise ValueError("La respuesta de FRED no tiene el formato esperado.")

    df = df.rename(columns={"observation_date": "fecha", serie_id: "indice_precio"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["indice_precio"] = df["indice_precio"].astype(float)
    return df[["fecha", "indice_precio"]]
