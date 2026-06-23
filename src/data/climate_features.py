"""Cálculo de indicadores agro-climáticos anuales a partir de datos horarios."""

import pandas as pd


def calcular_horas_frio(
    df: pd.DataFrame,
    umbral: float = 7.0,
) -> pd.Series:
    """
    Cuenta las horas con temperatura por debajo del umbral entre mayo y septiembre.

    El pistacho requiere acumulación de frío invernal para romper la dormición.
    Se consideran únicamente los meses de mayo a septiembre de cada año calendario.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame horario con columna 'temperature_2m' e índice DatetimeIndex.
    umbral : float
        Temperatura límite en °C. Se cuentan las horas estrictamente por debajo.
        Valor estándar para pistacho: 7.0 °C.

    Retorna
    -------
    pd.Series
        Serie indexada por año (int) con las horas de frío acumuladas.
    """
    mascara_meses = df.index.month.isin([5, 6, 7, 8, 9])
    frio = df.loc[mascara_meses, "temperature_2m"] < umbral
    return frio.groupby(frio.index.year).sum().rename("horas_frio").astype(int)


def calcular_heladas_tardias(df: pd.DataFrame) -> pd.Series:
    """
    Cuenta las horas con temperatura bajo cero durante el período de brotación.

    El intervalo crítico es del 15 de septiembre al 15 de octubre, cuando el
    pistacho inicia la brotación y es vulnerable a heladas tardías.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame horario con columna 'temperature_2m' e índice DatetimeIndex.

    Retorna
    -------
    pd.Series
        Serie indexada por año (int) con la cantidad de horas bajo 0 °C
        en el período de brotación. El año corresponde al del 15 de septiembre.
    """
    mes = df.index.month
    dia = df.index.day

    mascara = (
        ((mes == 9) & (dia >= 15)) |
        ((mes == 10) & (dia <= 15))
    )

    heladas = df.loc[mascara, "temperature_2m"] < 0.0

    # Agrupa por el año de septiembre: para registros de octubre se retrocede un mes
    año_referencia = df.index[mascara].to_series().apply(
        lambda ts: ts.year if ts.month == 9 else ts.year
    )

    return (
        heladas.groupby(año_referencia.values).sum()
        .rename("heladas_tardias")
        .astype(int)
    )


def calcular_calor_verano(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la temperatura máxima diaria promedio de enero y febrero.

    El calor estival es necesario para la apertura de cáscara del pistacho.
    El rango óptimo es 35–38 °C; valores por encima de 42 °C generan estrés.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame horario con columna 'temperature_2m' e índice DatetimeIndex.

    Retorna
    -------
    pd.Series
        Serie indexada por año (int) con la temperatura máxima diaria promedio
        de enero-febrero de ese año.
    """
    mascara_meses = df.index.month.isin([1, 2])
    verano = df.loc[mascara_meses, "temperature_2m"]

    tmax_diaria = verano.groupby(verano.index.date).max()
    tmax_diaria.index = pd.to_datetime(tmax_diaria.index)

    return (
        tmax_diaria.groupby(tmax_diaria.index.year).mean()
        .rename("tmax_verano_media")
        .round(2)
    )


def calcular_deficit_hidrico(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el déficit hídrico anual acumulado como ET₀ menos precipitación.

    Indica cuánta agua (en mm) debe aportar el riego para cubrir la demanda
    evapotranspirativa del cultivo. Ambas variables son horarias en mm.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame horario con columnas 'et0_fao_evapotranspiration' y
        'precipitation' e índice DatetimeIndex.

    Retorna
    -------
    pd.Series
        Serie indexada por año (int) con el déficit hídrico anual en mm.
        Valores positivos indican necesidad de riego.
    """
    deficit_horario = df["et0_fao_evapotranspiration"] - df["precipitation"]
    return (
        deficit_horario.groupby(deficit_horario.index.year).sum()
        .rename("deficit_hidrico_mm")
        .round(1)
    )
