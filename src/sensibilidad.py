"""Análisis de sensibilidad global (Sobol) sobre el simulador de Monte Carlo real.

No usa un dataset sintético ni Hipercubo Latino: cada punto de la muestra de
Sobol corre una simulación completa (`run_monte_carlo_antitetico` +
`resumen_financiero`) con esos parámetros. El precio usa `modo_precio="ar1"`
explícito (ver `_evaluar_metrica`) -- el modelo AR(1) calibrado con datos
reales de FRED (en niveles, la especificación principal), no el triangular naive.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.costos import ParametrosCostos
from src.monte_carlo import ParametrosMC, resumen_financiero, run_monte_carlo_antitetico
from src.precio_estocastico import ParametrosPrecioAR1

# Rangos (sincronización con el Excel v2, 2026-11):
# - tasa_descuento [0.10, 0.18]: tasa REAL en USD, alrededor del 14,23% del
#   Excel (hoja Tasa); el rango anterior [0.05, 0.12] dejaba esa tasa afuera.
# - hectareas [60, 180]: la mitad y 1,5 veces las 120 ha del plan v2.
# - precio_drift_anual [-0.0125, 0.0125]: drift real ADICIONAL del AR(1)
#   sobre la trayectoria del Excel (ParametrosPrecioAR1.c), mismo rango que
#   src/dataset_ml.py. Centrado en 0: desde 2026-11-16 la trayectoria del
#   Excel es plana (precio real constante), así que el drift total del
#   precio también queda centrado en 0, coherente con la serie histórica
#   deflactada (drift estimado -0,15% anual, indistinguible de cero). Es un
#   supuesto incierto, no ruido dentro de la simulación, así que entra a
#   Sobol como el resto.
# - precio_nivel_mult [0.625, 1.5]: multiplica TODA la trayectoria base de
#   precio (ParametrosPrecioAR1.nivel_mult): incertidumbre sobre el nivel
#   del precio. Fijado para el caso en verde (caso base desde 2026-11-20;
#   4 USD/kg verde -> 2,5-6, algo más ancho que el rango pesimista-optimista
#   del bloque 9, 2,5-5,5). El Sobol principal es el bayesiano
#   (PROBLEMA_SOBOL_BAYES, más abajo).
PROBLEMA_SOBOL: dict = {
    "num_vars": 8,
    "names": [
        "tasa_descuento",
        "precision_factor_frio",
        "precision_factor_calor",
        "p_bajo_si_alto",
        "capex_extra_pct",
        "hectareas",
        "precio_drift_anual",
        "precio_nivel_mult",
    ],
    "bounds": [
        [0.10, 0.18],
        [5.0, 30.0],
        [5.0, 30.0],
        [0.50, 0.80],
        [0.0, 0.30],
        [60.0, 180.0],
        [-0.0125, 0.0125],
        [0.625, 1.5],
    ],
}


_METRICAS_VALIDAS = ("van_neto_medio_usd", "prob_van_negativo")

# Superficie del plan v2 (ParametrosMC.hectareas). Para la tesis la superficie
# es una DECISIÓN, no una incertidumbre: con VAN por ha negativo, barrer
# hectáreas en [60, 180] solo reescala la pérdida (efecto escala) y le roba
# varianza al resto. Ver `comparar_sensibilidad_ha_fija()`.
HECTAREAS_PLAN = 120.0


def _problema(fijos: dict[str, float] | None) -> dict:
    """PROBLEMA_SOBOL sin las variables de `fijos` (que quedan constantes)."""
    if not fijos:
        return PROBLEMA_SOBOL
    desconocidas = set(fijos) - set(PROBLEMA_SOBOL["names"])
    if desconocidas:
        raise ValueError(f"variables fijas desconocidas: {sorted(desconocidas)}")
    pares = [
        (n, b) for n, b in zip(PROBLEMA_SOBOL["names"], PROBLEMA_SOBOL["bounds"])
        if n not in fijos
    ]
    return {
        "num_vars": len(pares),
        "names": [n for n, _ in pares],
        "bounds": [b for _, b in pares],
    }


def _completar(fila: np.ndarray, problema: dict, fijos: dict[str, float] | None) -> np.ndarray:
    """Vector de 8 parámetros (orden de PROBLEMA_SOBOL) a partir de una fila
    del problema reducido más los valores fijos."""
    valores = dict(zip(problema["names"], fila))
    valores.update(fijos or {})
    return np.array([valores[n] for n in PROBLEMA_SOBOL["names"]], dtype=float)


def _evaluar_metrica(
    fila: np.ndarray,
    escenario: str,
    n_simulaciones: int,
    semilla: int,
    metrica: str = "van_neto_medio_usd",
) -> float:
    """
    Corre una simulación completa con los 8 parámetros de `fila` (mismo
    orden que PROBLEMA_SOBOL["names"]) y devuelve el output escalar que
    analiza Sobol.

    `metrica` controla qué estadístico de `resumen_financiero()` se devuelve:
    - "van_neto_medio_usd" (default): resumen["van_neto_usd"].mean() — el
      VAN esperado del proyecto. Es el output histórico, para no romper el
      análisis ya guardado en data/processed/sobol_indices.parquet.
    - "prob_van_negativo": (resumen["van_neto_usd"] < 0).mean() — fracción
      de simulaciones con VAN negativo, una pregunta de RIESGO distinta de
      "cuánto VAN se espera en promedio".

    `capex_extra_pct` se pasa como `ParametrosCostos.capex_factor`
    (1 + pct), un recargo sobre todo el CAPEX que representa la
    incertidumbre de los ítems "A confirmar" del Excel v2 (riego por goteo,
    subsolado, tutores, reservorio -- ver data/modelo_v2/capex.csv).

    La semilla se mantiene FIJA en todas las evaluaciones (no varía con
    `fila`): así la única fuente de variación en el output es el barrido de
    parámetros, no ruido de muestreo adicional. Sobol asume una función
    determinística de las variables analizadas; si la semilla variara, el
    ruido de Monte Carlo (más notorio con `n_simulaciones` chico) se
    mezclaría con la sensibilidad real de cada parámetro.
    """
    if metrica not in _METRICAS_VALIDAS:
        raise ValueError(f"metrica debe ser una de {_METRICAS_VALIDAS}, recibido {metrica!r}")

    (
        tasa_descuento,
        precision_factor_frio,
        precision_factor_calor,
        p_bajo_si_alto,
        capex_extra_pct,
        hectareas,
        precio_drift_anual,
        precio_nivel_mult,
    ) = fila

    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=hectareas,
        tasa_descuento=tasa_descuento,
        precision_factor_frio=precision_factor_frio,
        precision_factor_calor=precision_factor_calor,
        p_bajo_si_alto=p_bajo_si_alto,
        escenario=escenario,
        semilla=semilla,
        # Explícitas en False: Sobol descompone la varianza atribuible a las 8
        # variables de PROBLEMA_SOBOL, no al ruido de CAPEX/OPEX estocástico ni
        # a la correlación frío/calor (Fases 1-2 del motor), que no son
        # variables analizadas acá. Sin esto se heredaría el default del motor
        # (capex_opex_estocastico=True) y los índices S1/ST absorberían esa
        # fuente de ruido no barrida, fuera del problema de Sobol.
        capex_opex_estocastico=False,
        correlacionar_frio_calor=False,
        # Explícito en "ar1" (el default del motor desde Fase A/paso 2, ver
        # PLAN_TESIS.md): antes de que existiera este flag, esta función
        # heredaba en silencio el triangular de run_monte_carlo_antitetico()
        # sin que nadie lo hubiera decidido a propósito para Sobol -- mismo
        # tipo de gap que ya se cerró acá para capex_opex_estocastico/
        # correlacionar_frio_calor. Se fija ar1 (no se deja el default
        # heredado) porque es el modelo de precio calibrado con datos reales
        # y Sobol debe analizar el simulador "oficial", no el modelo naive.
        modo_precio="ar1",
    )
    costos = ParametrosCostos(hectareas=hectareas, capex_factor=1.0 + capex_extra_pct)

    precio_params = ParametrosPrecioAR1(
        escenario=escenario, c=precio_drift_anual, nivel_mult=precio_nivel_mult
    )

    df = run_monte_carlo_antitetico(params, costos, precio_params)
    resumen = resumen_financiero(df, costos)

    if metrica == "van_neto_medio_usd":
        return float(resumen["van_neto_usd"].mean())
    return float((resumen["van_neto_usd"] < 0).mean())


def analizar_sensibilidad_sobol(
    escenario: str,
    n_base: int = 128,
    n_simulaciones: int = 1_000,
    semilla: int = 42,
    metrica: str = "van_neto_medio_usd",
    fijos: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad global de Sobol sobre el simulador real
    (`run_monte_carlo_antitetico`), para un escenario de precio fijo.

    El parámetro `escenario` de ParametrosMC es categórico y Sobol requiere
    variables continuas, así que este análisis se corre por separado para
    cada uno de los 3 valores de escenario (ver `comparar_sensibilidad_escenarios`).

    Se usa `calc_second_order=False` porque solo hacen falta S1 y ST: el
    muestreo de Saltelli requiere entonces N * (D + 2) evaluaciones del
    modelo (D=8 variables), en vez de N * (2D + 2) si también se pidieran
    los índices de segundo orden.

    Parámetros
    ----------
    escenario : str
        "pesimista", "base" u "optimista".
    n_base : int
        Tamaño base de la muestra de Sobol (N). Total de evaluaciones del
        modelo = n_base * (8 + 2) = n_base * 10. Calibrado empíricamente:
        cada evaluación con n_simulaciones=1.000 tarda ~0.2s, así que
        n_base=128 (1.280 evaluaciones/escenario) da ~13-21 minutos en
        total para los 3 escenarios (medido en corridas reales); n_base=256
        ya sube a ~21 minutos.
    n_simulaciones : int
        n_simulaciones de cada corrida de Monte Carlo dentro de cada
        evaluación de Sobol (deliberadamente más chico que el N de
        producción, para que el total de evaluaciones sea manejable).
    semilla : int
        Semilla FIJA para todas las evaluaciones (ver `_evaluar_metrica`) y
        para el muestreador de Sobol (`sobol_sample.sample(..., seed=semilla)`)
        y para el bootstrap de `sobol_analyze.analyze`. Sin esto, cada
        corrida usa una secuencia de Sobol distinta y S1/ST y sus bandas de
        confianza no son reproducibles de una corrida a otra.
    metrica : str
        Qué estadístico de `resumen_financiero()` analiza Sobol — ver
        `_evaluar_metrica`. Default "van_neto_medio_usd" para no romper el
        análisis ya guardado en data/processed/sobol_indices.parquet.
    fijos : dict, opcional
        Variables de PROBLEMA_SOBOL que se sacan del análisis y quedan
        constantes en el valor dado (p. ej. {"hectareas": 120.0}). Sobol se
        corre sobre las restantes: N * (D' + 2) evaluaciones.

    Retorna
    -------
    pd.DataFrame con columnas: parametro, S1, S1_conf, ST, ST_conf.
    """
    problema = _problema(fijos)
    muestra = sobol_sample.sample(
        problema, n_base, calc_second_order=False, seed=semilla
    )
    total = muestra.shape[0]
    print(f"[{escenario}/{metrica}] {total} evaluaciones del modelo (N={n_base} x {problema['num_vars'] + 2})...")

    outputs = np.empty(total)
    inicio = time.time()
    paso_reporte = max(1, total // 10)
    for i, fila in enumerate(muestra):
        fila_completa = _completar(fila, problema, fijos)
        outputs[i] = _evaluar_metrica(fila_completa, escenario, n_simulaciones, semilla, metrica)
        if (i + 1) % paso_reporte == 0 or (i + 1) == total:
            transcurrido = time.time() - inicio
            print(f"[{escenario}/{metrica}] {i + 1}/{total} ({transcurrido:.0f}s transcurridos)")

    indices = sobol_analyze.analyze(
        problema,
        outputs,
        calc_second_order=False,
        print_to_console=False,
        seed=semilla,
    )

    return pd.DataFrame({
        "parametro": problema["names"],
        "S1":        indices["S1"],
        "S1_conf":   indices["S1_conf"],
        "ST":        indices["ST"],
        "ST_conf":   indices["ST_conf"],
    })


def comparar_sensibilidad_escenarios(
    n_base: int = 128,
    n_simulaciones: int = 1_000,
    semilla: int = 42,
    metrica: str = "van_neto_medio_usd",
    ruta_salida: Path | str | None = None,
    fijos: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Corre `analizar_sensibilidad_sobol()` para los 3 escenarios de precio y
    devuelve un único DataFrame combinado (columna `escenario` adicional).

    Guarda el resultado en `ruta_salida`, o en
    data/processed/sobol_indices.parquet si no se especifica (comportamiento
    histórico, default `metrica="van_neto_medio_usd"`). Ver
    `comparar_sensibilidad_riesgo()` para el análisis con `metrica="prob_van_negativo"`,
    que guarda en un archivo separado para no pisar este.
    """
    resultados = []
    for escenario in ["pesimista", "base", "optimista"]:
        df_escenario = analizar_sensibilidad_sobol(
            escenario, n_base, n_simulaciones, semilla, metrica, fijos
        )
        df_escenario["escenario"] = escenario
        resultados.append(df_escenario)

    combinado = pd.concat(resultados, ignore_index=True)

    if ruta_salida is None:
        ruta_salida = Path(__file__).resolve().parents[1] / "data" / "processed" / "sobol_indices.parquet"
    else:
        ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(ruta_salida)
    print(f"Guardado en {ruta_salida}")

    return combinado


def comparar_sensibilidad_riesgo(
    n_base: int = 128, n_simulaciones: int = 1_000, semilla: int = 42
) -> pd.DataFrame:
    """
    Igual que `comparar_sensibilidad_escenarios()`, pero con
    `metrica="prob_van_negativo"` en vez del VAN medio: mide qué variable
    explica más la varianza de la PROBABILIDAD de que el proyecto termine
    con VAN negativo, en vez de la varianza del VAN esperado.

    Motivación: `precision_factor_frio`/`precision_factor_calor` dan
    S1≈ST≈0 en el análisis de VAN medio, porque la precisión de una Beta
    controla su varianza, no su media — no puede mover el promedio del VAN
    por construcción. Pero sí debería importar para el riesgo (con qué
    frecuencia un año malo de frío/calor arrastra el proyecto a VAN
    negativo), una pregunta de negocio distinta.

    Guarda en data/processed/sobol_indices_riesgo.parquet (archivo separado
    del análisis de VAN medio, para no pisarlo).
    """
    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "sobol_indices_riesgo.parquet"
    return comparar_sensibilidad_escenarios(
        n_base=n_base,
        n_simulaciones=n_simulaciones,
        semilla=semilla,
        metrica="prob_van_negativo",
        ruta_salida=ruta,
    )


def comparar_sensibilidad_ha_fija(
    n_base: int = 128, n_simulaciones: int = 1_000, semilla: int = 42
) -> pd.DataFrame:
    """
    Igual que `comparar_sensibilidad_escenarios()` (VAN medio), pero con
    `hectareas` fija en HECTAREAS_PLAN (120 ha) y Sobol sobre las otras 7
    variables. Es la versión para la tesis: la superficie es una decisión
    de inversión, no una incertidumbre. Con VAN por ha negativo, barrer
    hectáreas en [60, 180] reescala la pérdida (VAN determinístico: -1,91 M
    a 60 ha, -2,84 M a 120 ha, -3,84 M a 180 ha) y se lleva una parte
    grande de la varianza (ST 0,48 en el base, 0,76 en el pesimista) por
    puro efecto escala.

    Guarda en data/processed/sobol_indices_ha_fija.parquet (no pisa
    sobol_indices.parquet, que conserva la versión con 8 variables).
    """
    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "sobol_indices_ha_fija.parquet"
    return comparar_sensibilidad_escenarios(
        n_base=n_base,
        n_simulaciones=n_simulaciones,
        semilla=semilla,
        ruta_salida=ruta,
        fijos={"hectareas": HECTAREAS_PLAN},
    )


# ---------------------------------------------------------------------------
# Sobol con el precio bayesiano (Paso 7)
# ---------------------------------------------------------------------------

# 7 variables, superficie fija en 120 ha. Respecto de PROBLEMA_SOBOL, el
# nivel y el drift del precio (precio_nivel_mult, precio_drift_anual) se
# reemplazan por el peso de la paridad de exportación w de la mezcla del
# nivel mu (src/precio_bayesiano.py): en verde (caso base) w = 1 -> E[mu] =
# 2,5, w = 0 -> 5,5 USD/kg verde. Como el escenario solo fija w, se corre UNA vez con w en [0, 1] (no
# por escenario). factor_ars: tipo de cambio real, shock permanente sobre
# los costos en pesos (ParametrosCostos.factor_ars), en [0,8; 1,2].
PROBLEMA_SOBOL_BAYES: dict = {
    "num_vars": 7,
    "names": [
        "tasa_descuento",
        "precision_factor_frio",
        "precision_factor_calor",
        "p_bajo_si_alto",
        "capex_extra_pct",
        "peso_exportacion",
        "factor_ars",
    ],
    "bounds": [
        [0.10, 0.18],
        [5.0, 30.0],
        [5.0, 30.0],
        [0.50, 0.80],
        [0.0, 0.30],
        [0.0, 1.0],
        [0.8, 1.2],
    ],
}
METRICAS_BAYES = ("van_neto_medio_usd", "cvar_10_usd")


def _evaluar_bayes(fila: np.ndarray, n_simulaciones: int, semilla: int) -> tuple[float, float]:
    """Una simulación completa con el precio bayesiano; devuelve (VAN medio,
    CVaR 10 %) de la MISMA corrida. Semilla fija, como `_evaluar_metrica`."""
    from src.precio_bayesiano import ParametrosPrecioBayes

    (tasa, frio, calor, p_bajo, capex_extra, w, factor_ars) = fila
    params = ParametrosMC(
        n_simulaciones=n_simulaciones,
        hectareas=HECTAREAS_PLAN,
        tasa_descuento=tasa,
        precision_factor_frio=frio,
        precision_factor_calor=calor,
        p_bajo_si_alto=p_bajo,
        semilla=semilla,
        capex_opex_estocastico=False,
        correlacionar_frio_calor=False,
        modo_precio="bayesiano",
        # factor_ars es una variable barrida por Sobol: no se sortea además.
        factor_ars_estocastico=False,
    )
    costos = ParametrosCostos(
        hectareas=HECTAREAS_PLAN, capex_factor=1.0 + capex_extra, factor_ars=factor_ars
    )
    df = run_monte_carlo_antitetico(params, costos, ParametrosPrecioBayes(peso_exportacion=w))
    van = resumen_financiero(df, costos)["van_neto_usd"].to_numpy()
    n_peores = max(1, int(np.ceil(0.10 * len(van))))
    return float(van.mean()), float(np.sort(van)[:n_peores].mean())


def analizar_sobol_bayes(
    n_base: int = 256,
    n_simulaciones: int = 1_000,
    semilla: int = 42,
    ruta_salida: Path | str | None = None,
) -> pd.DataFrame:
    """
    Sobol (S1 y ST) sobre PROBLEMA_SOBOL_BAYES para el VAN medio y el CVaR
    10 % (las dos métricas salen de las mismas N * (7 + 2) evaluaciones).
    Guarda en data/processed/sobol_indices_bayes.parquet (columna `metrica`).
    """
    problema = PROBLEMA_SOBOL_BAYES
    muestra = sobol_sample.sample(problema, n_base, calc_second_order=False, seed=semilla)
    total = muestra.shape[0]
    print(f"[bayes] {total} evaluaciones (N={n_base} x {problema['num_vars'] + 2})...")
    salidas = np.empty((total, 2))
    inicio = time.time()
    paso = max(1, total // 20)
    for i, fila in enumerate(muestra):
        salidas[i] = _evaluar_bayes(fila, n_simulaciones, semilla)
        if (i + 1) % paso == 0 or (i + 1) == total:
            print(f"[bayes] {i + 1}/{total} ({time.time() - inicio:.0f}s)")

    resultados = []
    for j, metrica in enumerate(METRICAS_BAYES):
        ind = sobol_analyze.analyze(
            problema, salidas[:, j], calc_second_order=False, print_to_console=False, seed=semilla
        )
        resultados.append(pd.DataFrame({
            "parametro": problema["names"],
            "S1": ind["S1"], "S1_conf": ind["S1_conf"],
            "ST": ind["ST"], "ST_conf": ind["ST_conf"],
            "metrica": metrica,
        }))
    combinado = pd.concat(resultados, ignore_index=True)
    if ruta_salida is None:
        ruta_salida = Path(__file__).resolve().parents[1] / "data" / "processed" / "sobol_indices_bayes.parquet"
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(ruta_salida)
    # Salidas crudas, para poder re-analizar sin volver a simular.
    pd.DataFrame(np.column_stack([muestra, salidas]),
                 columns=problema["names"] + list(METRICAS_BAYES)).to_parquet(
        ruta_salida.with_name(ruta_salida.stem + "_evaluaciones.parquet"))
    print(f"Guardado en {ruta_salida}")
    return combinado
