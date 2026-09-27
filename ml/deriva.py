# ==========================================
# DETECCION DE DERIVA DE DATOS
#
# Que es la deriva
# ----------------
# El modelo aprendio de los datos de 2024. Si en 2026 los emprendimientos que
# llegan son distintos, por ejemplo levantan mas dinero o el clima de inversion
# cambio, el modelo sigue respondiendo pero sus respuestas valen menos. A eso se
# le llama deriva de datos, y es la principal razon por la que un modelo en
# produccion se va echando a perder solo, sin que nadie toque una linea.
#
# Como se detecta
# ---------------
# Se comparan los datos con los que se entreno contra los datos nuevos, columna
# por columna, con Evidently. Para las columnas numericas usa la distancia de
# Wasserstein y para las de texto la de Jensen-Shannon. Las dos funcionan igual
# para lo que nos importa: mientras mas alto el numero, mas se movio la columna.
# Si pasa del umbral, esa columna derivo.
# ==========================================

from evidently import Report
from evidently.presets import DataDriftPreset

from ml import config
from ml.data import completar_columnas

# A partir de que proporcion de columnas derivadas se levanta la alerta.
# Con 0.30, si mas de 5 de las 17 variables cambiaron, el pipeline avisa.
# Evidently por su cuenta usa 0.50, que para este proyecto es muy permisivo.
PROPORCION_PARA_ALERTAR = 0.30

# Las cinco variables que mas pesan en el modelo, segun la importancia por
# permutacion medida en la Unidad I. Si cualquiera de estas se mueve, se avisa
# aunque las demas esten quietas.
#
# Por que hace falta esta segunda regla: la simulacion de dos anios mostro que
# una deriva fuerte pero concentrada en pocas columnas se quedaba en 11 por
# ciento y nunca llegaba al 30. Y eran justamente el clima macroeconomico y el
# gasto mensual, dos de las variables que mas influyen en la prediccion.
VARIABLES_CRITICAS = [
    "product_market_fit_score",
    "runway_months",
    "monthly_burn_rate",
    "macro_climate",
    "cofounder_conflict",
]


def detectar_deriva(datos_referencia, datos_nuevos, ruta_html=None):
    """Compara los datos nuevos contra los del entrenamiento.

    "datos_referencia" son los datos con los que se entreno el modelo que esta
    en produccion, y "datos_nuevos" los que llegaron despues.

    Si se le pasa "ruta_html", guarda ahi el reporte visual de Evidently, que es
    el que se adjunta al informe.

    Devuelve un diccionario con el veredicto y el detalle por columna.
    """
    # Solo se miran las 17 variables que usa el modelo. Las demas columnas del
    # archivo pueden cambiar todo lo que quieran, no lo afectan.
    columnas = config.CARACTERISTICAS
    referencia = completar_columnas(datos_referencia)[columnas]
    nuevos = completar_columnas(datos_nuevos)[columnas]

    reporte = Report([DataDriftPreset()])
    resultado = reporte.run(current_data=nuevos, reference_data=referencia)

    if ruta_html is not None:
        ruta_html.parent.mkdir(parents=True, exist_ok=True)
        # as_iframe=False deja un HTML normal, que se abre solo en el navegador
        ruta_html.write_text(resultado.get_html_str(as_iframe=False), encoding="utf-8")

    # ---------- Leer el resultado ----------
    # Evidently devuelve una lista de metricas. Hay una por cada columna, que se
    # llama ValueDrift, y una de resumen, que se llama DriftedColumnsCount.
    detalle = []
    columnas_con_deriva = 0

    for metrica in resultado.dict()["metrics"]:
        tipo = metrica["config"]["type"]

        if tipo.endswith("ValueDrift"):
            umbral = metrica["config"]["threshold"]
            valor = float(metrica["value"])
            derivo = valor > umbral

            if derivo:
                columnas_con_deriva += 1

            detalle.append({
                "columna": metrica["config"]["column"],
                "metodo": metrica["config"]["method"],
                "distancia": round(valor, 4),
                "umbral": umbral,
                "derivo": derivo,
            })

    # Las columnas derivadas se muestran primero, que es lo que uno quiere ver
    detalle.sort(key=lambda c: c["distancia"], reverse=True)

    proporcion = columnas_con_deriva / len(columnas)

    # Las criticas que se movieron, que alcanzan por si solas para avisar
    criticas_derivadas = []
    for columna in detalle:
        if columna["derivo"] and columna["columna"] in VARIABLES_CRITICAS:
            criticas_derivadas.append(columna["columna"])

    hay_deriva = proporcion >= PROPORCION_PARA_ALERTAR or len(criticas_derivadas) > 0

    return {
        "hay_deriva": hay_deriva,
        "columnas_con_deriva": columnas_con_deriva,
        "total_de_columnas": len(columnas),
        "proporcion": round(proporcion, 4),
        "umbral_de_alerta": PROPORCION_PARA_ALERTAR,
        "criticas_derivadas": criticas_derivadas,
        "detalle": detalle,
    }


def resumir(resultado):
    """Arma el texto de una linea que se escribe en el registro del pipeline."""
    if resultado["hay_deriva"]:
        estado = "HAY DERIVA"
    else:
        estado = "sin deriva"

    linea = (f"{estado}: {resultado['columnas_con_deriva']} de "
             f"{resultado['total_de_columnas']} columnas cambiaron "
             f"({100 * resultado['proporcion']:.1f} por ciento)")

    if resultado["criticas_derivadas"]:
        linea += ", entre ellas " + ", ".join(resultado["criticas_derivadas"])

    return linea
