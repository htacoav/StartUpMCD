# ==========================================
# PIPELINE DE MANTENIMIENTO
# Ejecutar:  python -m ml.mantenimiento
#
# Lo corre GitHub Actions todos los lunes, y tambien se puede disparar a mano.
# Es el flujo que mantiene vivo al modelo sin que nadie tenga que acordarse.
#
# Son cinco pasos:
#   1. buscar los datos nuevos
#   2. revisar si los datos cambiaron (deriva)
#   3. entrenar un modelo candidato, SIN tocar el que esta en produccion
#   4. compararlo contra produccion y decidir si lo reemplaza
#   5. registrar todo en MLflow
#
# La regla de oro del paso 4: el modelo nuevo solo entra si supera al que esta
# y ademas pasa los minimos absolutos. Un pipeline que promueve cualquier cosa
# es peor que no tener pipeline, porque degrada el servicio automaticamente y
# nadie se entera.
# ==========================================

import json
import os
import urllib.request
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# app/almacen.py es el que sabe leer la base de consultas. El mantenimiento la
# lee directo cuando corre en la misma maquina que la aplicacion, y por la API
# cuando corre en GitHub Actions, que no ve el disco del servidor.
from app import almacen
from ml import config
from ml.data import cargar_datos
from ml.deriva import detectar_deriva, resumir
from ml.train import entrenar

# Donde se dejan el reporte de deriva y el resumen de la corrida
CARPETA_REPORTES = config.RAIZ / "reportes"

# Si alguien deja este archivo, se usa como "los datos nuevos". Es la ultima
# opcion, cuando no hay consultas reales disponibles.
RUTA_DATOS_NUEVOS = config.RAIZ / "data" / "datos_nuevos.csv"

# Direccion de la aplicacion en produccion. Si esta definida, el pipeline le
# pide las consultas por la API en vez de leer un archivo local.
API_PRODUCCION = os.environ.get("API_PRODUCCION", "")
TOKEN_ADMIN = os.environ.get("TOKEN_ADMIN", "")

# Cuantas consultas con desenlace conocido hacen falta para que valga la pena
# sumarlas al entrenamiento. Con menos que esto, agregar unas pocas filas sobre
# 48 000 no cambia el modelo y solo ensucia la comparacion.
MINIMO_DE_FILAS_ETIQUETADAS = 500

# Cuantas consultas hacen falta para que medir deriva signifique algo. Con
# cuatro filas las pruebas estadisticas dan cualquier cosa: al probarlo con 4
# consultas reales, Evidently reporto deriva en las 17 columnas, que era falso.
# Por debajo de este numero se prefiere seguir con datos simulados y decirlo.
MINIMO_PARA_MEDIR_DERIVA = 100

# Minimos que tiene que cumplir cualquier modelo para llegar a produccion, sin
# importar que tan bueno sea el que esta. Son los mismos numeros que vigila la
# prueba de puerta de calidad en tests/test_modelo.py.
MINIMOS = {
    "roc_auc": 0.70,
    "balanced_accuracy": 0.65,
}

# Si el modelo marca mas de este porcentaje de startups como fracaso, se
# considera degenerado aunque sus otras notas se vean bien. Es la leccion del
# umbral que quedo documentada en el informe de la Unidad I.
MAXIMO_SENALADOS = 0.80

# Cuanto tiene que mejorar el AUC para que valga la pena reemplazar el modelo.
# Con un numero mas chico se estaria redesplegando por puro ruido estadistico.
MEJORA_MINIMA = 0.002


def consultas_desde_la_api():
    """Baja las consultas guardadas desde la aplicacion en produccion.

    Devuelve None si no hay direccion configurada o si el servidor no responde.
    Que falle no es motivo para cortar el mantenimiento: se sigue con lo que
    haya disponible localmente.
    """
    if not API_PRODUCCION:
        return None

    direccion = f"{API_PRODUCCION.rstrip('/')}/consultations?limit=10000"
    if TOKEN_ADMIN:
        direccion += f"&token={TOKEN_ADMIN}"

    try:
        with urllib.request.urlopen(direccion, timeout=30) as respuesta:
            datos = json.loads(respuesta.read())
    except Exception as error:
        print(f"AVISO: no se pudieron bajar las consultas de produccion ({error})")
        return None

    filas = [c["variables"] for c in datos["consultations"]]
    print(f"Bajadas {len(filas)} consultas de {API_PRODUCCION}")
    return pd.DataFrame(filas) if filas else None


def hay_suficientes(cuantas, de_donde):
    """Dice si alcanzan las consultas para que la medicion de deriva valga.

    Medir deriva con un punado de filas no da un resultado malo, da un
    resultado inventado: las pruebas estadisticas necesitan volumen para
    distinguir un cambio real del azar.
    """
    if cuantas >= MINIMO_PARA_MEDIR_DERIVA:
        return True

    print(f"Hay {cuantas} consultas en {de_donde}, hacen falta "
          f"{MINIMO_PARA_MEDIR_DERIVA} para medir deriva con sentido")
    return False


def obtener_datos_nuevos(datos_completos):
    """Devuelve los datos que llegaron desde el ultimo entrenamiento.

    Busca en cuatro lugares, en este orden:
      1. la API de produccion, que es la fuente real
      2. la base de consultas local, si el pipeline corre en el mismo servidor
      3. el archivo data/datos_nuevos.csv, si alguien lo dejo a mano
      4. una simulacion, que es lo que pasa mientras todavia no hay suficientes

    Las opciones 1 y 2 se descartan si hay menos consultas que el minimo, para
    no medir deriva sobre un punado de filas.

    Devuelve la tabla y una palabra que dice de donde salio, que queda anotada
    en el resumen de la corrida para que nadie confunda datos reales con
    simulados.
    """
    de_la_api = consultas_desde_la_api()
    if de_la_api is not None and hay_suficientes(len(de_la_api), "produccion"):
        return de_la_api, "produccion"

    locales = almacen.listar(limite=10000)
    if locales:
        print(f"Leidas {len(locales)} consultas de la base local")
        if hay_suficientes(len(locales), "base local"):
            return pd.DataFrame([c["variables"] for c in locales]), "base local"

    if RUTA_DATOS_NUEVOS.exists():
        print(f"Datos nuevos leidos de {RUTA_DATOS_NUEVOS.name}")
        return pd.read_csv(RUTA_DATOS_NUEVOS), "archivo"

    muestra = datos_completos.sample(frac=0.25, random_state=2026)
    print(f"No hay consultas todavia, se simulan {len(muestra)} filas nuevas")
    return muestra, "simulados"


def separar_consultas_reales():
    """Parte las consultas con desenlace conocido en entrenamiento y examen.

    El examen se aparta y no se usa para entrenar. Sirve para responder la
    unica pregunta que de verdad importa al decidir si se reemplaza el modelo:
    cual de los dos funciona mejor con los emprendimientos que efectivamente
    usan la aplicacion.

    Devuelve (None, None) si todavia no hay suficientes.
    """
    etiquetadas = almacen.filas_para_entrenar()

    if len(etiquetadas) < MINIMO_DE_FILAS_ETIQUETADAS:
        print(f"Hay {len(etiquetadas)} consultas con desenlace conocido, "
              f"hacen falta {MINIMO_DE_FILAS_ETIQUETADAS} para usarlas")
        return None, None

    tabla = pd.DataFrame(etiquetadas)
    if tabla[config.OBJETIVO].nunique() < 2:
        print("Todas las consultas terminaron igual, no se pueden usar para comparar")
        return None, None

    para_entrenar, para_examen = train_test_split(
        tabla, test_size=0.3, random_state=config.SEMILLA, stratify=tabla[config.OBJETIVO])

    print(f"Consultas reales: {len(para_entrenar)} para entrenar, "
          f"{len(para_examen)} apartadas para comparar")
    return para_entrenar, para_examen


def datos_para_entrenar(datos_completos, consultas_reales=None):
    """El dataset original mas las consultas reales que ya tienen desenlace."""
    if consultas_reales is None or len(consultas_reales) == 0:
        return datos_completos, 0

    juntos = pd.concat([datos_completos, consultas_reales], ignore_index=True)
    return juntos, len(consultas_reales)


def auc_sobre(modelo, para_examen):
    """Que tan bien distingue un modelo sobre las consultas apartadas."""
    from ml.data import preparar

    X, y = preparar(para_examen)
    return float(roc_auc_score(y, modelo.predict_proba(X)[:, 1]))


def comparar_sobre_consultas_reales(candidato, para_examen):
    """Mide al candidato y al modelo en produccion sobre las mismas consultas.

    Por que hace falta esta medicion aparte
    ---------------------------------------
    Las metricas normales del entrenamiento se calculan sobre una particion del
    dataset, que tiene 48 000 filas historicas. Las consultas reales son unos
    miles. Aunque el modelo nuevo sea claramente mejor con los emprendimientos
    que usan la aplicacion, esa mejora queda diluida y el AUC general no se
    mueve, o hasta baja.

    Se comprobo con la simulacion de dos anios: el candidato daba 0.7431 contra
    0.7469 del modelo en produccion y por eso se rechazaba, cuando sobre las
    consultas reales daba 0.6650 contra 0.6484, o sea 0.0166 mejor. Sin esta
    medicion el sistema nunca habria aprendido nada de sus propios datos.
    """
    if para_examen is None or not config.RUTA_MODELO.exists():
        return None

    en_produccion = joblib.load(config.RUTA_MODELO)

    return {
        "auc_produccion": auc_sobre(en_produccion["pipeline"], para_examen),
        "auc_candidato": auc_sobre(candidato["pipeline"], para_examen),
        "filas_del_examen": int(len(para_examen)),
    }


def leer_modelo_en_produccion():
    """Las notas del modelo que esta atendiendo ahora mismo."""
    if not config.RUTA_METRICAS.exists():
        return None
    return json.loads(config.RUTA_METRICAS.read_text(encoding="utf-8"))


def pasa_los_minimos(notas):
    """Revisa que el modelo cumpla los minimos absolutos.

    Devuelve la lista de motivos por los que NO pasa. Si la lista sale vacia,
    el modelo esta bien.
    """
    motivos = []

    for metrica in MINIMOS:
        if notas[metrica] < MINIMOS[metrica]:
            motivos.append(f"{metrica} = {notas[metrica]:.4f}, por debajo de {MINIMOS[metrica]}")

    if notas["flagged_rate"] >= MAXIMO_SENALADOS:
        motivos.append(f"marca el {100 * notas['flagged_rate']:.1f} por ciento de los casos, "
                       f"el limite es {100 * MAXIMO_SENALADOS:.0f}")

    return motivos


def decidir(notas_candidato, notas_produccion, comparacion=None):
    """Decide si el modelo candidato reemplaza al que esta en produccion.

    Son dos condiciones distintas y las dos tienen que cumplirse. Primero, el
    candidato tiene que pasar los minimos absolutos, que se miden sobre el
    dataset completo. Segundo, tiene que ganarle al modelo actual: sobre las
    consultas reales si ya hay suficientes, y sobre el dataset si todavia no.

    Devuelve un diccionario con la decision y el motivo escrito, para que quede
    en el registro del pipeline y cualquiera entienda por que se hizo lo que se
    hizo.
    """
    motivos_de_rechazo = pasa_los_minimos(notas_candidato)

    if motivos_de_rechazo:
        return {
            "promover": False,
            "motivo": "El candidato no pasa los minimos: " + "; ".join(motivos_de_rechazo),
        }

    # ---------- La comparacion buena, cuando se puede hacer ----------
    if comparacion is not None:
        diferencia = comparacion["auc_candidato"] - comparacion["auc_produccion"]
        detalle = (f"sobre {comparacion['filas_del_examen']} consultas reales apartadas, "
                   f"AUC {comparacion['auc_candidato']:.4f} contra "
                   f"{comparacion['auc_produccion']:.4f}, diferencia {diferencia:+.4f}")

        if diferencia < MEJORA_MINIMA:
            return {"promover": False, "motivo": f"El candidato no mejora lo suficiente: {detalle}"}

        return {"promover": True, "motivo": f"El candidato mejora {detalle}"}

    # Si no hay modelo en produccion, cualquiera que pase los minimos sirve
    if notas_produccion is None:
        return {"promover": True, "motivo": "No habia modelo en produccion"}

    auc_candidato = notas_candidato["roc_auc"]
    auc_produccion = notas_produccion["metrics"]["roc_auc"]
    diferencia = auc_candidato - auc_produccion

    if diferencia < MEJORA_MINIMA:
        return {
            "promover": False,
            "motivo": (f"El candidato no mejora lo suficiente: AUC {auc_candidato:.4f} "
                       f"contra {auc_produccion:.4f} en produccion, diferencia "
                       f"{diferencia:+.4f} y se exige {MEJORA_MINIMA:+.4f}"),
        }

    return {
        "promover": True,
        "motivo": (f"El candidato mejora el AUC de {auc_produccion:.4f} a "
                   f"{auc_candidato:.4f}, diferencia {diferencia:+.4f}"),
    }


def registrar_en_mlflow(artefacto, resultado_deriva, decision, ruta_reporte_deriva,
                        comparacion=None):
    """Deja constancia de la corrida en MLflow.

    Si la variable MLFLOW_TRACKING_URI no esta definida, MLflow guarda todo en
    una carpeta local. Asi el pipeline corre igual en una maquina sin servidor.

    Si el servidor esta caido, se avisa y se sigue: perder el registro es
    molesto, pero no es razon para tumbar todo el mantenimiento.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow no esta instalado, se omite el registro")
        return None

    direccion = os.environ.get("MLFLOW_TRACKING_URI")
    if direccion:
        mlflow.set_tracking_uri(direccion)

    try:
        mlflow.set_experiment("startup-survival")

        # ETIQUETA_CORRIDA permite ponerle nombre propio a la corrida. La usa la
        # simulacion para que las cuatro ejecuciones se distingan en MLflow, en
        # vez de quedar todas con la misma hora.
        etiqueta = os.environ.get("ETIQUETA_CORRIDA", "")
        nombre = etiqueta or f"mantenimiento-{datetime.now(timezone.utc):%Y%m%d-%H%M}"

        with mlflow.start_run(run_name=nombre):
            if etiqueta:
                mlflow.set_tag("origen", "simulacion")
                mlflow.log_param("corrida", etiqueta)

            mlflow.log_params(artefacto["hyperparameters"])
            mlflow.log_param("modelo", artefacto["model_name"])
            mlflow.log_param("umbral", artefacto["threshold"])
            mlflow.log_param("filas_de_entrenamiento", artefacto["rows"])

            for nombre in artefacto["metrics"]:
                valor = artefacto["metrics"][nombre]
                if isinstance(valor, (int, float)):
                    mlflow.log_metric(nombre, valor)

            mlflow.log_metric("columnas_con_deriva", resultado_deriva["columnas_con_deriva"])
            mlflow.log_metric("proporcion_de_deriva", resultado_deriva["proporcion"])

            # El AUC sobre consultas reales es el numero con el que se decide,
            # asi que tiene que quedar registrado junto a los demas.
            if comparacion is not None:
                mlflow.log_metric("auc_consultas_reales", comparacion["auc_candidato"])
                mlflow.log_metric("auc_consultas_reales_produccion", comparacion["auc_produccion"])
                mlflow.log_metric("filas_del_examen_real", comparacion["filas_del_examen"])

            mlflow.set_tag("promovido", str(decision["promover"]))
            mlflow.set_tag("motivo", decision["motivo"])

            if ruta_reporte_deriva.exists():
                mlflow.log_artifact(str(ruta_reporte_deriva))

            # El modelo se registra siempre, se promueva o no. Guardar tambien
            # los que no pasaron sirve para ver la historia completa despues.
            info = mlflow.sklearn.log_model(
                artefacto["pipeline"],
                artifact_path="modelo",
                serialization_format="cloudpickle",
                registered_model_name="startup-survival",
            )

            # El alias "produccion" apunta a la version que esta atendiendo
            if decision["promover"]:
                from mlflow import MlflowClient
                cliente = MlflowClient()
                versiones = cliente.search_model_versions("name='startup-survival'")
                ultima = max(versiones, key=lambda v: int(v.version))
                cliente.set_registered_model_alias("startup-survival", "produccion", ultima.version)
                print(f"MLflow: version {ultima.version} marcada como produccion")

            print(f"MLflow: corrida registrada en {mlflow.get_tracking_uri()}")
            return info.model_uri

    except Exception as error:
        print(f"AVISO: no se pudo registrar en MLflow ({error})")
        return None


def ejecutar():
    """El pipeline completo. Devuelve el resumen de lo que paso."""
    print("=" * 60)
    print("PIPELINE DE MANTENIMIENTO")
    print("=" * 60)

    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)

    # ---------- PASO 1: los datos ----------
    print("\n--- 1. Datos ---")
    datos = cargar_datos()
    datos_nuevos, origen = obtener_datos_nuevos(datos)

    # ---------- PASO 2: deriva ----------
    print("\n--- 2. Deriva de datos ---")
    ruta_reporte_deriva = CARPETA_REPORTES / "deriva.html"
    resultado_deriva = detectar_deriva(datos, datos_nuevos, ruta_html=ruta_reporte_deriva)
    print(resumir(resultado_deriva))
    print(f"Reporte visual en {ruta_reporte_deriva}")

    for columna in resultado_deriva["detalle"][:5]:
        marca = "  <-- derivo" if columna["derivo"] else ""
        print(f"   {columna['columna']:<28} {columna['distancia']:.4f}{marca}")

    # ---------- PASO 3: candidato ----------
    print("\n--- 3. Entrenando el candidato ---")
    para_entrenar_real, para_examen_real = separar_consultas_reales()
    datos_de_entrenamiento, filas_reales = datos_para_entrenar(datos, para_entrenar_real)
    # guardar=False es lo importante: no se toca models/ hasta haber decidido
    candidato = entrenar(guardar=False, datos=datos_de_entrenamiento)

    # ---------- PASO 4: decision ----------
    print("\n--- 4. Decision ---")
    produccion = leer_modelo_en_produccion()
    comparacion = comparar_sobre_consultas_reales(candidato, para_examen_real)
    if comparacion is not None:
        print(f"Sobre las consultas reales apartadas: produccion "
              f"{comparacion['auc_produccion']:.4f}, candidato "
              f"{comparacion['auc_candidato']:.4f}")
    decision = decidir(candidato["metrics"], produccion, comparacion)
    print(decision["motivo"])
    print("PROMOVER" if decision["promover"] else "SE QUEDA EL MODELO ACTUAL")

    # ---------- PASO 5: MLflow ----------
    print("\n--- 5. Registro en MLflow ---")
    uri_modelo = registrar_en_mlflow(candidato, resultado_deriva, decision,
                                     ruta_reporte_deriva, comparacion)

    # ---------- Guardar el modelo si gano ----------
    if decision["promover"]:
        import joblib
        joblib.dump(candidato, config.RUTA_MODELO)
        reporte = {}
        for clave in candidato:
            if clave not in ("pipeline", "categorical_options", "numeric_ranges", "shap_background"):
                reporte[clave] = candidato[clave]
        config.RUTA_METRICAS.write_text(json.dumps(reporte, indent=2), encoding="utf-8")
        print(f"Modelo reemplazado en {config.RUTA_MODELO}")

    # ---------- Resumen ----------
    resumen = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen_de_los_datos": origen,
        "filas_nuevas": int(len(datos_nuevos)),
        "consultas_reales_en_el_entrenamiento": filas_reales,
        "deriva": {
            "hay_deriva": resultado_deriva["hay_deriva"],
            "columnas_con_deriva": resultado_deriva["columnas_con_deriva"],
            "proporcion": resultado_deriva["proporcion"],
        },
        "candidato": {
            "modelo": candidato["model_name"],
            "roc_auc": candidato["metrics"]["roc_auc"],
            "balanced_accuracy": candidato["metrics"]["balanced_accuracy"],
            "flagged_rate": candidato["metrics"]["flagged_rate"],
        },
        "produccion_anterior": None if produccion is None else produccion["metrics"]["roc_auc"],
        "comparacion_sobre_consultas_reales": comparacion,
        "promovido": decision["promover"],
        "motivo": decision["motivo"],
        "modelo_en_mlflow": uri_modelo,
    }

    ruta_resumen = CARPETA_REPORTES / "mantenimiento.json"
    ruta_resumen.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"\nResumen guardado en {ruta_resumen}")

    return resumen


if __name__ == "__main__":
    ejecutar()
