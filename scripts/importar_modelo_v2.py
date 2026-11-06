"""Exporta los parámetros del modelo financiero v2 (Excel) a CSV.

El motor (`src/modelo_v2.py`, `src/costos.py`) NUNCA lee el .xlsx: lee los
CSV que genera este script en `data/modelo_v2/`. Así el Excel queda como
fuente de referencia versionada y el motor no depende de openpyxl ni de la
estructura de celdas de la planilla.

Archivos que genera:

- supuestos.csv   parámetros escalares (hoja Supuestos + hoja Tasa)
- capex.csv       ítems de CAPEX con vida útil y regla de escala con hectáreas
- opex.csv        ítems de OPEX (parte A de la hoja OPEX) con etapa, rampa y tipo
- anual.csv       series por año 0-20: curva de rendimiento, precios por
                  escenario y rampa de fertilización (Supuestos, filas 54-58)
- referencia_flujo.csv        flujo de fondos año a año de la hoja Flujo
                              (incluye la fila 32, quebrantos utilizados)
- referencia_indicadores.csv  VAN, TIR, tasa y CAPEX total del Excel

Los dos `referencia_*` no los usa el motor: son el blanco contra el que
`tests/test_consistencia_excel.py` verifica la corrida determinística.

Uso:
    python scripts/importar_modelo_v2.py [ruta_xlsx] [directorio_salida]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
XLSX_DEFAULT = RAIZ / "data" / "referencia" / "Modelo financiero - Pistacho Mendoza v2.xlsx"
SALIDA_DEFAULT = RAIZ / "data" / "modelo_v2"

# Columnas D..X de las hojas con proyección anual = años 0..20
COLUMNAS_AÑOS = [openpyxl.utils.get_column_letter(c) for c in range(4, 25)]

# Celdas escalares de la hoja Supuestos (y Tasa) que usa el motor. Se mapean
# explícitamente por dirección para que un cambio de layout en el Excel
# falle ruidosamente en vez de exportar el valor de otra fila.
SUPUESTOS_ESCALARES: dict[str, tuple[str, str, str]] = {
    # parametro: (hoja!celda, unidad, etiqueta esperada en la columna A)
    "ultimo_anio_preproductivo": ("Supuestos!B8", "año", "Último año pre-productivo"),
    "hectareas_referencia": ("Supuestos!B11", "ha", "Superficie a plantar"),
    "plantas_totales": ("Supuestos!B16", "plantas", "Plantas totales"),
    "proporcion_machos": ("Supuestos!B17", "fracción", "Proporción de machos"),
    "rendimiento_plena_kg_ha": ("Supuestos!B20", "kg/ha", "Rendimiento a plena producción"),
    "ha_por_sistema_solar": ("Supuestos!B21", "ha", "Hectáreas cubiertas por cada sistema solar"),
    "salario_operario_usd_mes": ("Supuestos!B25", "USD/mes", "Salario base operario"),
    "salario_encargado_usd_mes": ("Supuestos!B26", "USD/mes", "Salario base encargado"),
    "cargas_patronales": ("Supuestos!B27", "fracción", "Cargas patronales"),
    "aumento_real_salario": ("Supuestos!B28", "fracción/año", "Aumento real de salario anual"),
    "antiguedad_anual": ("Supuestos!B29", "fracción/año", "Antigüedad anual"),
    "operarios_anio1": ("Supuestos!B30", "personas", "Dotación de operarios – año 1"),
    "ha_por_operario": ("Supuestos!B31", "ha", "Hectáreas por operario"),
    "anio_dotacion_plena": ("Supuestos!B33", "año", "Año en que se alcanza la dotación plena"),
    "encargados": ("Supuestos!B34", "personas", "Encargados de finca"),
    "tipo_cambio_ars_usd": ("Supuestos!B37", "ARS/USD", "Tipo de cambio de referencia"),
    "alicuota_ganancias": ("Supuestos!B46", "fracción", "Alícuota de Impuesto a las Ganancias"),
    "alicuota_iibb": ("Supuestos!B47", "fracción de ventas", "Alícuota de Ingresos Brutos"),
    "inflacion_eeuu": ("Supuestos!B48", "fracción/año", "Inflación esperada EE.UU."),
    "tasa_descuento_real": ("Supuestos!B49", "fracción/año", "Tasa de descuento real"),
    "quebrantos_aplicar": ("Supuestos!B64", "0/1", "Aplicar traslado de quebrantos"),
    "quebrantos_anios": ("Supuestos!B65", "años", "Años de traslado de quebrantos"),
    "tasa_rf": ("Tasa!B6", "fracción/año", "Tasa libre de riesgo"),
    "prima_mercado": ("Tasa!B7", "fracción/año", "Prima de riesgo de mercado"),
    "beta_desapalancada": ("Tasa!B8", "-", "Beta desapalancada"),
    "prima_riesgo_pais": ("Tasa!B9", "fracción/año", "Prima de riesgo país"),
    "prima_especifica": ("Tasa!B10", "fracción/año", "Prima específica"),
    "wacc_nominal_usd": ("Tasa!B20", "fracción/año", "WACC nominal en USD"),
}

# Filas de la hoja Supuestos con series por año (columnas D..X = años 0..20)
SERIES_ANUALES: dict[str, tuple[int, str]] = {
    "rendimiento_pct_plena": (54, "Rendimiento (% de plena producción)"),
    "precio_pesimista_usd_kg": (55, "Precio pesimista"),
    "precio_base_usd_kg": (56, "Precio base"),
    "precio_optimista_usd_kg": (57, "Precio optimista"),
    "rampa_fertilizacion": (58, "Fertilización y fitosanidad"),
}

# Celdas de Supuestos que dependen de la superficie: si la fórmula de
# cantidad de un ítem las referencia, el ítem escala con las hectáreas.
_REFS_ESCALA_HA = ("$B$11", "$B$16", "$B$18", "$B$19")
_REF_SISTEMA_SOLAR = "$B$21"


def _celda(libro: openpyxl.Workbook, ref: str):
    hoja, celda = ref.split("!")
    return libro[hoja][celda]


def _escala_desde_formula(formula) -> str:
    """'ha', 'sistema_solar' o 'fijo' según qué celdas de Supuestos referencie."""
    if not isinstance(formula, str) or not formula.startswith("="):
        return "fijo"
    if _REF_SISTEMA_SOLAR in formula:
        return "sistema_solar"
    if any(ref in formula for ref in _REFS_ESCALA_HA):
        return "ha"
    return "fijo"


def _exportar_supuestos(valores, formulas) -> pd.DataFrame:
    filas = []
    for parametro, (ref, unidad, etiqueta) in SUPUESTOS_ESCALARES.items():
        hoja, celda = ref.split("!")
        fila_excel = int(re.sub(r"\D", "", celda))
        texto_a = str(valores[hoja][f"A{fila_excel}"].value or "")
        if not texto_a.startswith(etiqueta):
            raise ValueError(
                f"{ref}: se esperaba la etiqueta {etiqueta!r} en la columna A, "
                f"se encontró {texto_a!r}. ¿Cambió el layout del Excel?"
            )
        filas.append({
            "parametro": parametro,
            "valor": float(_celda(valores, ref).value),
            "unidad": unidad,
            "fuente": ref,
        })

    # Valor residual: la fórmula de Supuestos!B50 apunta a una fila de CAPEX
    # (hoy la tierra). Se exporta como marca en capex.csv, no como monto,
    # para que escale con las hectáreas igual que el ítem referenciado.
    return pd.DataFrame(filas)


def _fila_valor_residual(formulas) -> int:
    formula = formulas["Supuestos"]["B50"].value
    m = re.fullmatch(r"=CAPEX!\$F\$(\d+)", str(formula))
    if not m:
        raise ValueError(f"Supuestos!B50 no tiene la forma =CAPEX!$F$n: {formula!r}")
    return int(m.group(1))


def _exportar_capex(valores, formulas) -> pd.DataFrame:
    hv, hf = valores["CAPEX"], formulas["CAPEX"]
    fila_residual = _fila_valor_residual(formulas)
    filas = []
    r = 5
    while hv[f"B{r}"].value != "TOTAL CAPEX":
        if hv[f"B{r}"].value is None:
            raise ValueError(f"CAPEX!B{r} vacío antes de 'TOTAL CAPEX'")
        filas.append({
            "categoria": hv[f"A{r}"].value,
            "item": hv[f"B{r}"].value,
            "unidad": hv[f"C{r}"].value,
            "cantidad": float(hv[f"D{r}"].value or 0),
            "precio_unit_usd": float(hv[f"E{r}"].value or 0),
            "total_usd": float(hv[f"F{r}"].value or 0),
            "moneda_pago": hv[f"G{r}"].value,
            "vida_util_anios": int(hv[f"H{r}"].value or 0),
            "etapa": hv[f"J{r}"].value,
            "estado": hv[f"K{r}"].value,
            "fuente": hv[f"L{r}"].value,
            "escala": _escala_desde_formula(hf[f"D{r}"].value),
            "valor_residual": r == fila_residual,
        })
        r += 1

    df = pd.DataFrame(filas)
    total_excel = float(hv[f"F{r}"].value)
    if abs(df["total_usd"].sum() - total_excel) > 0.01:
        raise ValueError(
            f"Suma de CAPEX exportado ({df['total_usd'].sum():.2f}) != "
            f"TOTAL CAPEX del Excel ({total_excel:.2f})"
        )
    return df


def _tipo_opex(unidad, costo_anual, aplica_en) -> str:
    if isinstance(costo_anual, str) and "Personal" in costo_anual:
        return "personal"
    if unidad == "USD/kg":
        return "por_kg"
    if unidad == "% ventas":
        return "pct_ventas"
    return "fijo_anual"


def _exportar_opex(valores, formulas) -> pd.DataFrame:
    hv, hf = valores["OPEX"], formulas["OPEX"]
    filas = []
    r = 6
    while hv[f"B{r}"].value is not None:
        unidad = hv[f"C{r}"].value
        costo_anual = hv[f"F{r}"].value
        tipo = _tipo_opex(unidad, costo_anual, hv[f"H{r}"].value)
        cantidad = hv[f"D{r}"].value
        precio = hv[f"E{r}"].value
        filas.append({
            "categoria": hv[f"A{r}"].value,
            "item": hv[f"B{r}"].value,
            "unidad": unidad,
            # pct_ventas: la alícuota está en la columna D (Cant./año)
            "cantidad": float(cantidad) if isinstance(cantidad, (int, float)) else 0.0,
            "precio_unit_usd": float(precio) if isinstance(precio, (int, float)) else 0.0,
            "costo_anual_usd": float(costo_anual) if isinstance(costo_anual, (int, float)) else 0.0,
            "moneda_pago": hv[f"G{r}"].value,
            "aplica_en": hv[f"H{r}"].value,
            "rampa": hv[f"I{r}"].value == "Rampa",
            "estado": hv[f"J{r}"].value,
            "fuente": hv[f"K{r}"].value,
            "tipo": tipo,
            "escala": _escala_desde_formula(hf[f"D{r}"].value),
        })
        r += 1

    df = pd.DataFrame(filas)
    etapas_validas = {"Todos", "Pre", "Prod", "Año 1"}
    desconocidas = set(df["aplica_en"]) - etapas_validas
    if desconocidas:
        raise ValueError(f"OPEX: valores de 'Aplica en' no soportados: {desconocidas}")
    return df


def _exportar_anual(valores) -> pd.DataFrame:
    hv = valores["Supuestos"]
    datos: dict[str, list] = {"anio": list(range(len(COLUMNAS_AÑOS)))}
    for nombre, (fila, etiqueta) in SERIES_ANUALES.items():
        if not str(hv[f"A{fila}"].value).startswith(etiqueta):
            raise ValueError(f"Supuestos!A{fila}: se esperaba {etiqueta!r}")
        datos[nombre] = [float(hv[f"{c}{fila}"].value or 0) for c in COLUMNAS_AÑOS]
    return pd.DataFrame(datos)


# Filas de la hoja Flujo exportadas como referencia (año 0..20)
_FILAS_FLUJO = {
    "ingresos_usd": 16,
    "opex_usd": 17,
    "ebitda_usd": 18,
    "amortizaciones_usd": 19,
    "impuesto_ganancias_usd": 21,
    "capex_usd": 24,
    "flujo_libre_usd": 25,
    "valor_residual_usd": 26,
    "flujo_evaluacion_usd": 27,
    "flujo_descontado_usd": 30,
    # Fila agregada al final de la hoja (después del flujo descontado
    # acumulado): Flujo!21 resta estos quebrantos del EBIT antes de Ganancias.
    "quebrantos_utilizados_usd": 32,
}

# Etiqueta esperada en la columna A de las filas de Flujo que se leen por
# número: si alguien inserta una fila en el medio, falla en vez de exportar
# la fila equivocada.
_ETIQUETAS_FLUJO = {
    21: "(−) Impuesto a las Ganancias",
    30: "Flujo descontado",
    32: "Quebrantos utilizados",
}


def _exportar_referencia(valores) -> tuple[pd.DataFrame, pd.DataFrame]:
    hv = valores["Flujo"]
    for fila, etiqueta in _ETIQUETAS_FLUJO.items():
        texto_a = str(hv[f"A{fila}"].value or "")
        if not texto_a.startswith(etiqueta):
            raise ValueError(
                f"Flujo!A{fila}: se esperaba {etiqueta!r}, se encontró {texto_a!r}. "
                "¿Cambió el layout del Excel?"
            )
    flujo = pd.DataFrame({"anio": list(range(len(COLUMNAS_AÑOS)))})
    for nombre, fila in _FILAS_FLUJO.items():
        flujo[nombre] = [float(hv[f"{c}{fila}"].value or 0) for c in COLUMNAS_AÑOS]

    indicadores = pd.DataFrame([
        {"indicador": "van_usd", "valor": float(hv["B5"].value), "fuente": "Flujo!B5"},
        {"indicador": "tir_real", "valor": float(hv["B6"].value), "fuente": "Flujo!B6"},
        {"indicador": "tasa_descuento_real", "valor": float(hv["G5"].value), "fuente": "Flujo!G5"},
        {"indicador": "capex_total_usd", "valor": float(valores["CAPEX"]["F37"].value), "fuente": "CAPEX!F37"},
        {"indicador": "escenario_precio", "valor": float(valores["Supuestos"]["B22"].value), "fuente": "Supuestos!B22 (1/2/3)"},
    ])
    return flujo, indicadores


def importar(xlsx: Path = XLSX_DEFAULT, salida: Path = SALIDA_DEFAULT) -> dict[str, pd.DataFrame]:
    valores = openpyxl.load_workbook(xlsx, data_only=True)
    formulas = openpyxl.load_workbook(xlsx, data_only=False)

    flujo, indicadores = _exportar_referencia(valores)
    tablas = {
        "supuestos": _exportar_supuestos(valores, formulas),
        "capex": _exportar_capex(valores, formulas),
        "opex": _exportar_opex(valores, formulas),
        "anual": _exportar_anual(valores),
        "referencia_flujo": flujo,
        "referencia_indicadores": indicadores,
    }

    salida.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.to_csv(salida / f"{nombre}.csv", index=False, encoding="utf-8", lineterminator="\n")
    return tablas


if __name__ == "__main__":
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else XLSX_DEFAULT
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else SALIDA_DEFAULT
    tablas = importar(xlsx, salida)
    for nombre, df in tablas.items():
        print(f"{salida / (nombre + '.csv')}: {len(df)} filas")
