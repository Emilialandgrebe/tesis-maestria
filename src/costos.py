"""Estructura de costos e impuestos del proyecto de pistacho (modelo v2).

Fuente: `data/referencia/Modelo financiero - Pistacho Mendoza v2.xlsx`,
exportado a `data/modelo_v2/` por `scripts/importar_modelo_v2.py` y leído
vía `src/modelo_v2.py`. Este módulo reproduce las fórmulas de las hojas
CAPEX, Personal, OPEX y Flujo del Excel; no tiene montos propios.

Todo en USD constantes de 2026 (flujos reales, ver Supuestos!B7), proyecto
sin financiamiento: el servicio de deuda que cargaba el plan de negocio
viejo como OPEX ya no existe (la hoja Financiamiento del Excel queda fuera
del flujo del proyecto, igual que en el propio Excel).

Escala con hectáreas: cada ítem de CAPEX y OPEX trae en su CSV una regla
`escala` que el importador derivó de la fórmula de cantidad del Excel:
- "ha": la cantidad referencia la superficie (o las plantas, que salen de
  la superficie) y escala linealmente con `hectareas`;
- "sistema_solar": ROUNDUP(hectareas / ha_por_sistema_solar) unidades;
- "fijo": no depende de la superficie.
La dotación de operarios escala por la regla de la hoja Personal
(1 operario cada `ha_por_operario` ha en plena producción, redondeado).

El valor de los CSV a 120 ha reproduce el Excel exactamente; a otra
superficie, la única aproximación es que las plantas escalan linealmente en
vez de redondearse a planta entera como en Supuestos!B16.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.modelo_v2 import ModeloV2, cargar_modelo_v2

_MODELO = cargar_modelo_v2()

HECTAREAS_REFERENCIA = _MODELO.supuestos["hectareas_referencia"]

# Variación estructural del OPEX respecto del valor del Excel: triangular
# simétrica ±15%, UN draw por simulación (ver simulate_opex_multiplicador()
# en src/monte_carlo.py). No hay rango de mercado investigado para esto: es
# un supuesto declarado, no una afirmación empírica. Se aplica a todo el
# OPEX operativo (personal, insumos, maquila) pero NO a Ingresos Brutos,
# que es una alícuota sobre ventas, no un costo que pueda salir más caro de
# lo presupuestado.
OPEX_VARIACION_PCT = 0.15


def _redondear_excel(x: np.ndarray | float) -> np.ndarray:
    """ROUND() de Excel para valores >= 0 (mitades hacia arriba, no al par)."""
    return np.floor(np.asarray(x, dtype=float) + 0.5)


@dataclass
class ParametrosCostos:
    """Parámetros de costos del proyecto, escalados a `hectareas`.

    `capex_factor` multiplica TODO el CAPEX (y por lo tanto también las
    amortizaciones): es el recargo que barren `src/sensibilidad.py` y
    `src/dataset_ml.py` como `capex_extra_pct` (factor = 1 + pct),
    representando ítems "A confirmar" que terminen saliendo más caros. No
    afecta el valor residual: la tierra se recupera a su valor de compra.

    `quebrantos` activa el traslado de quebrantos impositivos (Supuestos
    bloque 7 / hoja Quebrantos del Excel): la pérdida fiscal (base imponible
    < 0) de un año se descuenta de la base de Ganancias de los
    `quebrantos_anios` años siguientes, usando primero las más antiguas
    (FIFO). Default ENCENDIDO: es el caso base del Excel (Supuestos!B64 = 1).
    Ver `quebrantos_utilizados()`.

    `activar_opex_pre` es una sensibilidad (Supuestos!B66, default apagado):
    el OPEX de los años pre-productivos (1..Supuestos!B8) no se deduce en el
    año sino que se activa como costo de implantación y se amortiza en línea
    recta en `vida_util_implantacion` años desde la primera cosecha. Solo
    cambia la base imponible de Ganancias, no la caja. Ver
    `opex_pre_activado()`. Tratamiento fiscal a confirmar con el contador.

    Sincronizar `hectareas` con `ParametrosMC.hectareas` es responsabilidad
    de quien arma la simulación (`run_monte_carlo()` lo hace automático).
    """

    hectareas: float = HECTAREAS_REFERENCIA
    capex_factor: float = 1.0
    opex_variacion_pct: float = OPEX_VARIACION_PCT
    alicuota_ganancias: float = _MODELO.supuestos["alicuota_ganancias"]
    quebrantos: bool = True
    quebrantos_anios: int = int(_MODELO.supuestos["quebrantos_anios"])
    activar_opex_pre: bool = False
    vida_util_implantacion: int = int(_MODELO.supuestos["vida_util_implantacion"])
    modelo: ModeloV2 = field(default=_MODELO, repr=False)

    def __post_init__(self) -> None:
        """
        Valida los rangos al construir el objeto: fuera de rango no fallan
        acá sino más tarde y en silencio (p. ej. un multiplicador de OPEX
        <= 0 haría que el flujo SUME el OPEX en vez de restarlo).
        """
        if not (0.0 <= self.opex_variacion_pct < 1.0):
            raise ValueError(
                "opex_variacion_pct debe estar en [0, 1); recibido "
                f"{self.opex_variacion_pct!r}. Fuera de ese rango, la "
                "triangular(1-d, 1, 1+d) de simulate_opex_multiplicador() "
                "puede sortear un multiplicador <= 0."
            )
        if self.capex_factor <= 0:
            raise ValueError(f"capex_factor debe ser > 0; recibido {self.capex_factor!r}")
        if self.hectareas <= 0:
            raise ValueError(f"hectareas debe ser > 0; recibido {self.hectareas!r}")
        if int(self.quebrantos_anios) != self.quebrantos_anios or self.quebrantos_anios < 1:
            raise ValueError(
                f"quebrantos_anios debe ser un entero >= 1; recibido {self.quebrantos_anios!r}"
            )
        if (int(self.vida_util_implantacion) != self.vida_util_implantacion
                or self.vida_util_implantacion < 1):
            raise ValueError(
                "vida_util_implantacion debe ser un entero >= 1; recibido "
                f"{self.vida_util_implantacion!r}"
            )

    # ------------------------------------------------------------------
    # Escala con hectáreas
    # ------------------------------------------------------------------

    def _cantidad_escalada(self, items: pd.DataFrame) -> np.ndarray:
        s = self.modelo.supuestos
        factor_ha = self.hectareas / s["hectareas_referencia"]
        n_sistemas = np.ceil(self.hectareas / s["ha_por_sistema_solar"])
        return np.select(
            [items["escala"] == "ha", items["escala"] == "sistema_solar"],
            [items["cantidad"] * factor_ha, n_sistemas],
            default=items["cantidad"],
        )

    # ------------------------------------------------------------------
    # CAPEX, amortizaciones y valor residual
    # ------------------------------------------------------------------

    @property
    def capex_items(self) -> pd.DataFrame:
        """Hoja CAPEX escalada: agrega `total_escalado_usd` (con capex_factor)."""
        items = self.modelo.capex.copy()
        items["total_escalado_usd"] = (
            self._cantidad_escalada(items) * items["precio_unit_usd"] * self.capex_factor
        )
        return items

    @property
    def capex_inicial(self) -> float:
        """CAPEX total del año 0 (CAPEX!F37 a 120 ha con capex_factor=1)."""
        return float(self.capex_items["total_escalado_usd"].sum())

    def amortizaciones(self, n_años: int) -> np.ndarray:
        """
        Amortización de cada año 1..n_años, forma (n_años,). Lineal por vida
        útil; un ítem con vida útil H amortiza en los años 1..H (fórmula de
        Flujo!E19: SUMPRODUCT(amortización * (vida_útil >= año))). Vida útil
        0 = no se amortiza (tierra, análisis, acta).
        """
        items = self.capex_items
        vida = items["vida_util_anios"].to_numpy()
        anual = np.divide(
            items["total_escalado_usd"].to_numpy(), vida,
            out=np.zeros(len(items)), where=vida > 0,
        )
        años = np.arange(1, n_años + 1).reshape(-1, 1)
        return (anual.reshape(1, -1) * (vida.reshape(1, -1) >= años)).sum(axis=1)

    @property
    def valor_residual(self) -> float:
        """Valor residual al final del horizonte (Supuestos!B50: la tierra).

        Sin `capex_factor`: el recargo representa sobrecostos de ítems por
        confirmar, no un mayor valor de reventa de la tierra.
        """
        items = self.modelo.capex
        marca = items["valor_residual"].astype(bool)
        return float(
            (self._cantidad_escalada(items) * items["precio_unit_usd"])[marca].sum()
        )

    # ------------------------------------------------------------------
    # OPEX
    # ------------------------------------------------------------------

    def costo_personal(self, n_años: int) -> np.ndarray:
        """Costo laboral total por año 1..n_años (hoja Personal, fila 14)."""
        s = self.modelo.supuestos
        t = np.arange(1, n_años + 1)

        op1 = s["operarios_anio1"]
        op_plena = _redondear_excel(self.hectareas / s["ha_por_operario"])
        años_rampa = s["anio_dotacion_plena"] - 1
        dotacion = _redondear_excel(
            op1 + (op_plena - op1) * np.minimum(t - 1, años_rampa) / años_rampa
        )

        indice_salarial = (
            (1 + s["aumento_real_salario"]) ** (t - 1)
            * (1 + s["antiguedad_anual"] * (t - 1))
            * (1 + s["cargas_patronales"])
        )
        operarios = s["salario_operario_usd_mes"] * indice_salarial * 12 * dotacion
        encargados = s["salario_encargado_usd_mes"] * indice_salarial * 12 * s["encargados"]
        return operarios + encargados

    def _aplica_por_año(self, aplica_en: pd.Series, n_años: int) -> np.ndarray:
        """Máscara (n_items, n_años): si cada ítem corre en cada año (OPEX!H)."""
        ultimo_pre = self.modelo.supuestos["ultimo_anio_preproductivo"]
        t = np.arange(1, n_años + 1).reshape(1, -1)
        a = aplica_en.to_numpy().reshape(-1, 1)
        return np.select(
            [a == "Todos", a == "Pre", a == "Prod", a == "Año 1"],
            [np.ones_like(t), t <= ultimo_pre, t > ultimo_pre, t == 1],
            default=0,
        ).astype(float)

    def opex_fijo(self, n_años: int) -> np.ndarray:
        """
        OPEX que no depende de la producción ni de las ventas, por año
        1..n_años, forma (n_años,): personal + ítems de costo anual, con su
        etapa (Todos / Pre / Prod / Año 1) y la rampa de fertilización de
        los años 1-6 en los ítems marcados "Rampa" (OPEX parte B).
        """
        opex = self.modelo.opex
        fijos = opex[opex["tipo"] == "fijo_anual"]
        costo = self._cantidad_escalada(fijos) * fijos["precio_unit_usd"].to_numpy()

        rampa = self.modelo.rampa_fertilizacion(n_años).reshape(1, -1)
        factor_rampa = np.where(fijos["rampa"].to_numpy().reshape(-1, 1), rampa, 1.0)
        mascara = self._aplica_por_año(fijos["aplica_en"], n_años)

        por_item = costo.reshape(-1, 1) * mascara * factor_rampa
        return por_item.sum(axis=0) + self.costo_personal(n_años)

    @property
    def costo_por_kg(self) -> float:
        """USD por kg producido (maquila de descascarado y secado)."""
        opex = self.modelo.opex
        return float(opex.loc[opex["tipo"] == "por_kg", "precio_unit_usd"].sum())

    @property
    def alicuota_iibb(self) -> float:
        """Ingresos Brutos como fracción de las ventas."""
        opex = self.modelo.opex
        return float(opex.loc[opex["tipo"] == "pct_ventas", "cantidad"].sum())


def costo_operativo_anual(año: int, params: ParametrosCostos) -> float:
    """
    OPEX FIJO del año dado (personal + ítems anuales), sin maquila ni
    Ingresos Brutos -- esos dos dependen de la producción y las ventas de
    cada simulación y se calculan en `opex_total()`.
    """
    return float(params.opex_fijo(año)[año - 1])


def opex_total(
    produccion_kg: np.ndarray,
    ingresos_usd: np.ndarray,
    params: ParametrosCostos,
    opex_multiplicador: np.ndarray,
) -> np.ndarray:
    """
    OPEX total por simulación y año (OPEX!D68):
    (fijo + maquila * kg) * multiplicador + alícuota IIBB * ventas.

    Parámetros
    ----------
    produccion_kg, ingresos_usd : np.ndarray
        Forma (n_simulaciones, n_años).
    opex_multiplicador : np.ndarray
        Forma (n_simulaciones,). UN valor por simulación, aplicado a todos
        los años (ver `simulate_opex_multiplicador()` en src/monte_carlo.py).
    """
    n, T = produccion_kg.shape
    operativo = params.opex_fijo(T).reshape(1, T) + params.costo_por_kg * produccion_kg
    multiplicador = np.asarray(opex_multiplicador).reshape(n, 1)
    return operativo * multiplicador + params.alicuota_iibb * ingresos_usd


def opex_pre_activado(
    opex_usd: np.ndarray, params: ParametrosCostos
) -> tuple[np.ndarray, np.ndarray]:
    """
    OPEX pre-productivo activado y su amortización, por simulación y año
    1..T (hoja Flujo, filas 33 y 34). Ceros si `params.activar_opex_pre`
    está apagado.

        activado(t)   = OPEX(t)                       si 1 <= t <= B8
        amortización  = sum(activado) / vida útil     si B8 < t <= B8 + vida útil

    con B8 = último año pre-productivo. El total activado es el de cada
    simulación (el OPEX es estocástico). Con vida útil 20 y horizonte 20,
    solo se amortizan los años 7..20: el saldo no amortizado no se deduce
    dentro del horizonte (igual que el Excel).
    """
    opex_usd = np.asarray(opex_usd, dtype=float)
    activado = np.zeros_like(opex_usd)
    amortizacion = np.zeros_like(opex_usd)
    if not params.activar_opex_pre:
        return activado, amortizacion
    T = opex_usd.shape[1]
    años = np.arange(1, T + 1)
    ultimo_pre = params.modelo.supuestos["ultimo_anio_preproductivo"]
    vida = int(params.vida_util_implantacion)
    pre = años <= ultimo_pre
    activado[:, pre] = opex_usd[:, pre]
    cuota = activado.sum(axis=1, keepdims=True) / vida
    en_amortizacion = (años > ultimo_pre) & (años <= ultimo_pre + vida)
    amortizacion[:, en_amortizacion] = cuota
    return activado, amortizacion


def quebrantos_utilizados(ebit: np.ndarray, anios: int) -> np.ndarray:
    """
    Quebrantos impositivos usados en cada año, por simulación (hoja
    Quebrantos del Excel, fila 55).

    `ebit` es la base imponible antes de quebrantos (Flujo!35; igual al
    EBIT si no se activa el OPEX pre-productivo). Cada año con base < 0
    genera un quebranto igual a la pérdida. Un
    quebranto originado en el año y puede usarse en los años t con
    y < t <= y + `anios` (vence después); en cada año se consume hasta la
    base positiva, empezando por los de origen más antiguo (FIFO). El
    quebranto mantiene su valor nominal en USD constantes (Supuestos!D65:
    se asume actualizado).

    Se calcula por fila: en Monte Carlo cada trayectoria tiene su propia
    secuencia de pérdidas y ganancias, así que el saldo de quebrantos es
    distinto en cada simulación.

    Parámetros
    ----------
    ebit : np.ndarray
        Forma (n_simulaciones, T), años 1..T. El año 0 del Excel tiene
        EBIT = 0 (solo CAPEX), así que no genera quebranto.
    anios : int
        Plazo de traslado (Supuestos!B65).

    Retorna
    -------
    np.ndarray de la misma forma que `ebit`, >= 0, con el quebranto usado
    en cada año (nunca mayor que el EBIT positivo del año).
    """
    ebit = np.asarray(ebit, dtype=float)
    n, T = ebit.shape
    saldo = np.maximum(0.0, -ebit)  # saldo por año de origen, se va consumiendo
    usado = np.zeros_like(ebit)
    for t in range(T):
        restante = np.maximum(0.0, ebit[:, t])
        for y in range(max(0, t - anios), t):  # de más antiguo a más nuevo
            uso = np.minimum(saldo[:, y], restante)
            saldo[:, y] -= uso
            restante -= uso
            usado[:, t] += uso
    return usado


def flujo_caja_neto(
    ingresos_usd: np.ndarray,
    opex_usd: np.ndarray,
    params: ParametrosCostos,
) -> dict[str, np.ndarray]:
    """
    Flujo libre del proyecto por simulación y año 1..T, sin el CAPEX del
    año 0 (hoja Flujo, filas 18-27 y 32-35):

        EBITDA      = ingresos - OPEX
        EBIT        = EBITDA - amortizaciones
        base        = EBIT + OPEX activado - amortización implantación  (Flujo!35)
        quebrantos  = quebrantos_utilizados(base)  si params.quebrantos, si no 0
        Ganancias   = max(0, (base - quebrantos) * alícuota)   (Flujo!21)
        flujo       = EBITDA - Ganancias (+ valor residual en el año T)

    La activación del OPEX pre-productivo (`opex_pre_activado()`) solo
    mueve la base imponible: la caja cambia únicamente vía Ganancias.

    Retorna
    -------
    dict con "ebitda", "opex_pre_activado", "amortizacion_implantacion",
    "base_imponible", "quebrantos_utilizados", "impuesto_ganancias" y
    "flujo_neto", cada uno de la misma forma que `ingresos_usd`.
    """
    T = ingresos_usd.shape[1]
    ebitda = ingresos_usd - opex_usd
    ebit = ebitda - params.amortizaciones(T).reshape(1, T)
    activado, amort_implantacion = opex_pre_activado(opex_usd, params)
    base = ebit + activado - amort_implantacion
    if params.quebrantos:
        usados = quebrantos_utilizados(base, int(params.quebrantos_anios))
    else:
        usados = np.zeros_like(base)
    impuesto = np.maximum(0.0, (base - usados) * params.alicuota_ganancias)

    flujo = ebitda - impuesto
    flujo[:, -1] += params.valor_residual
    return {
        "ebitda": ebitda,
        "opex_pre_activado": activado,
        "amortizacion_implantacion": amort_implantacion,
        "base_imponible": base,
        "quebrantos_utilizados": usados,
        "impuesto_ganancias": impuesto,
        "flujo_neto": flujo,
    }
