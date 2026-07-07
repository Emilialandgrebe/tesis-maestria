# Notas de calidad — dataset financiero (capa 2)

Fuente: `Plan_Negocio_Pistacho_Mendoza` (Google Sheets, 13 hojas), extraído el 2026-07-03.
Estos CSVs son una transcripción fiel de los valores de la planilla — no se corrigió
ni se completó ningún dato faltante. Los totales de cada archivo fueron verificados
contra los totales que la propia planilla declara (coinciden exactamente).

## Decisiones ya tomadas (no volver a discutir)

- Superficie base para el dataset financiero: **25 ha** (Fase I, la que tiene riego
  y plantación costeados en el CAPEX itemizado).
- Total de CAPEX de referencia: **USD 1.129.013,12** (suma itemizada en `capex.csv`),
  no los USD 630.000 de la hoja Financiamiento/Flujo de Caja ni los USD 460.000 de
  la tabla "movimientos de cash" dentro de la misma hoja CAPEX.

## Conflictos sin resolver — pendientes de decisión

1. **CAPEX itemizado está incompleto.** De los ~29 ítems, 11 tienen `costeado=NO`
   (precio unitario vacío en la fuente → total cargado como 0): todo el sistema de
   riego (pozo, bombas, filtros, tuberías, represa), desmonte, subsolado, marcación,
   paneles solares, red antigranizo, estación meteorológica, sitio web. El total de
   USD 1.129.013 **no incluye el costo real del riego**, que es la partida más
   incierta del proyecto. El total real va a ser mayor una vez que Facundo/Patricio
   o los proveedores contactados devuelvan cotizaciones.

2. **Tres superficies conviven en la misma planilla:** 100 ha (tierra comprada),
   50 ha (desmonte/nivelación), 25 ha (riego y plantación). No parece un error sino
   fases de desarrollo no explicitadas (Fase I ~25 ha, Fase II "ampliación a 50 ha"
   mencionada en la hoja Financiamiento). El resto de la tierra (hasta 100 ha) no
   tiene fecha de desarrollo definida.

3. **`produccion_ingresos_plan.csv` mezcla al menos dos bases de hectáreas.**
   `kg_total_fuente` = `kg_ha × 100` de forma exacta y consistente para todos los
   años verificados. Pero los ingresos (`ingreso_*_usd`) no son consistentes con
   ninguna base de hectáreas única: para el año 8 coinciden exactamente con
   `kg_ha × 25 ha × precio`, pero para el año 9 no coinciden ni con 25, 50 ni 100 ha
   bajo ese mismo cálculo. Es decir, **los propios ingresos de la planilla no son
   reproducibles con una fórmula simple y consistente** — probablemente dependen de
   celdas/fórmulas no visibles en la exportación de valores que hice. No usar esta
   columna de ingresos como fuente de verdad sin antes revisar las fórmulas
   originales en Google Sheets.

4. **Costos de asesores agronómicos NO están en ningún OPEX.** Ver
   `asesores_equipo.csv`: "Asesor Agronómico Pistacho" (el rol que cumplen Facundo
   y Patricio en la práctica) tiene un costo estimado en la hoja "Equipo y Org."
   (USD 300-500/mes) pero no aparece como línea en `opex_preproductivo.csv` ni en
   `opex_productivo.csv`. Tampoco están: Asesor Legal, Desarrollador Web,
   Asociación Frutos Secos. El Contador y el Seguro sí están parcialmente cubiertos
   en ambos OPEX.

5. **Posible error de carga en OPEX Pre-productivo, año 1.** La fila "Encargado de
   finca (full-time)" tiene `cantidad_por_anio = 72` (interpretado como 72 meses)
   con precio unitario USD 1.200/mes → USD 86.400 solo en el año 1, lo que dispara
   el total de ese año a USD 126.790 contra ~USD 57-63k en los años siguientes. Si
   fuera un error de tipeo (72 en vez de 12), el total real del año 1 sería
   ~USD 54.790, mucho más consistente con la progresión de los años 2-6.

6. **Break-even inconsistente entre hojas.** "Factibilidad Técnica" dice
   "punto de equilibrio: 470 kg"; "Flujo de Caja y KPIs" dice "~1.200 kg/ha". No se
   sabe si son la misma métrica en unidades distintas o cálculos distintos.

7. **TIR/VAN/Payback en la hoja "Flujo de Caja y KPIs" son rangos estimados a
   mano** ("~14%-16%", "USD 280.000-450.000", "Año 12-14"), no recalculados a
   partir del flujo de caja fila por fila que está en la misma hoja. No asumir que
   son el resultado de la fórmula real del flujo de caja mostrado arriba.

## Estimaciones web para los ítems sin costear (`capex_estimaciones_web.csv`)

El 2026-07-03 busqué en la web valores de mercado para los ítems de CAPEX marcados
`costeado=NO` en `capex.csv`. Son **estimaciones de orden de magnitud, no
cotizaciones reales** — no reemplazan la cotización de Facundo/Patricio o de los
proveedores ya contactados. Quedaron en un archivo separado para no mezclar datos
reales con estimados dentro de `capex.csv`.

| Ítem | Estimado (USD) | Confianza | Fuente |
|---|---|---|---|
| Sistema de riego por goteo enterrado (25 ha) | 81.250 (rango 70.000-87.500) | media | Los Andes, Argentina.gob.ar, Novagric — USD 2.800-3.500/ha para leñosos, 2025 |
| Pozo de agua x2 (perforación+entubado+bomba+estudios) | 33.000 (rango 20.000-40.000) | baja | granperforista.ar — solo la perforación (USD 100/m) tiene fuente directa |
| Desmonte + nivelación + subsolado (25 ha) | 50.000 | muy baja | Estimado por resta contra un comparable de granado en Mendoza/San Juan; la finca ya está "semidesbrozada", así que el costo real probablemente sea menor |
| Represa/cisterna (5.000 m³) | sin estimar | — | No encontré ninguna fuente confiable; requiere cotización real |
| Paneles solares | sin estimar (a propósito) | — | La familia ya está cotizando esto activamente (to-do de la hoja Resumen) |

**Total CAPEX itemizado + estimado web (sin represa ni paneles): ≈ USD 1.293.263**
(USD 1.129.013 real + USD 164.250 estimado). Sigue siendo un piso, no un total
final: faltan represa y paneles, y las estimaciones de desmonte/subsolado tienen
confianza muy baja.

## Hallazgos que confirman decisiones ya tomadas en el proyecto

- El techo de 3.000 kg/ha (plena producción) es consistente con la curva de
  `produccion_ingresos_plan.csv` (satura en 3.000 desde el año 11 en adelante).
- Un asesor agronómico del proyecto (probablemente Facundo, según el contexto de
  "Factibilidad Técnica") afirma textualmente que *"la falta de calor en verano es
  hasta peor que la falta de horas frío"* — respalda la decisión de modelar déficit
  de calor (no exceso) en `_factor_calor()` de `src/monte_carlo.py`.

## Hojas de la planilla no incluidas en esta extracción

Financiamiento (estructura de capital, cronograma de desembolsos), Estrategia
Comercial, Análisis de Riesgos, Cronograma, Análisis de Mercado — son más
cualitativas/estratégicas y no se transcribieron a CSV. Están disponibles en la
planilla si hacen falta más adelante.
