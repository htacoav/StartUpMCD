# ==========================================================================
# SIMULACION DE DOS ANIOS DE OPERACION
# Ejecutar:  python -m simulacion.simular_dos_anios
#
# Que simula
# ----------
# Mes 0:   3500 emprendedores responden el cuestionario.
# Mes 10:  la incubadora completa el desenlace de esos 3500, porque recien
#          despues de diez meses se sabe quien sobrevivio.
# Meses 1 al 24: siguen llegando unos 200 emprendedores por mes, y cada mes se
#          completan los desenlaces de la camada de diez meses atras.
# Cada seis meses corre el pipeline de mantenimiento con esos datos reales.
#
# Para que sirve
# --------------
# Es la unica forma de ver funcionar el mantenimiento sin esperar dos anios.
# Muestra tres cosas que en la exposicion no se pueden mostrar de otro modo:
# como se acumulan los datos etiquetados, cuando salta la alerta de deriva, y
# si el modelo reentrenado con datos propios le gana al original.
#
# Que es inventado y que no
# -------------------------
# Las caracteristicas de cada emprendimiento se toman de filas reales del
# dataset, asi que las distribuciones y las correlaciones son las de verdad.
# Lo inventado es el departamento, el desenlace y la deriva del tiempo, que se
# generan con reglas explicitas escritas mas abajo. Todo esto es una
# simulacion y asi se declara: no son emprendimientos peruanos reales.
# ==========================================================================

import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ml import config  # noqa: E402

# ---------------------------------------------------------------
# Lo primero de todo: aislar la simulacion del proyecto de verdad.
# El pipeline reemplaza models/model.joblib cuando promueve un modelo, y una
# simulacion no tiene por que tocar el modelo que esta en produccion.
# ---------------------------------------------------------------
CARPETA = RAIZ / "simulacion" / "resultados"
CARPETA.mkdir(parents=True, exist_ok=True)

config.RUTA_CONSULTAS = CARPETA / "consultas_simuladas.db"
config.RUTA_MODELO = CARPETA / "model_simulado.joblib"
config.RUTA_METRICAS = CARPETA / "metrics_simuladas.json"

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app import almacen  # noqa: E402
from ml import mantenimiento  # noqa: E402
from ml.data import cargar_datos  # noqa: E402

SEMILLA = 2026
random.seed(SEMILLA)
np.random.seed(SEMILLA)

# ---------------------------------------------------------------
# PARAMETROS DE LA SIMULACION
# ---------------------------------------------------------------
MESES = 24
CONSULTAS_DEL_PRIMER_MES = 3500
CONSULTAS_POR_MES = 200
MESES_HASTA_SABER_EL_DESENLACE = 10
MESES_EN_QUE_SE_REENTRENA = [10, 12, 18, 24]

# Cuanta gente de cada departamento usa la aplicacion. Lima concentra la mayor
# parte, como pasa con casi cualquier servicio en el Peru.
PESO_DEL_DEPARTAMENTO = {
    "Lima": 34, "Arequipa": 8, "La Libertad": 6, "Piura": 6, "Cusco": 5,
    "Lambayeque": 5, "Junin": 4, "Callao": 4, "Ancash": 3, "Ica": 3,
    "Cajamarca": 3, "Puno": 3, "San Martin": 2, "Loreto": 2, "Ucayali": 2,
    "Tacna": 2, "Ayacucho": 2, "Huanuco": 2, "Apurimac": 1, "Amazonas": 1,
    "Moquegua": 1, "Tumbes": 1, "Pasco": 1, "Madre de Dios": 1, "Huancavelica": 1,
}

# EL EFECTO QUE SE INYECTA. Esto es lo que el modelo tendria que descubrir
# cuando se reentrene con los datos reales. Son log-odds: un numero positivo
# significa que ahi es mas dificil sobrevivir.
#
# La idea detras es que donde hay mas ecosistema emprendedor (capital,
# proveedores, clientes) sobrevivir cuesta menos. El dataset original no sabe
# nada de esto, porque ni siquiera tiene la columna.
EFECTO_DEL_DEPARTAMENTO = {
    "Lima": -0.45, "Callao": -0.30, "Arequipa": -0.25, "La Libertad": -0.15,
    "Lambayeque": -0.05, "Piura": 0.00, "Cusco": 0.05, "Ica": 0.05,
    "Junin": 0.10, "Ancash": 0.15, "Tacna": 0.15, "Cajamarca": 0.25,
    "Puno": 0.30, "San Martin": 0.30, "Ayacucho": 0.35, "Huanuco": 0.40,
    "Moquegua": 0.20, "Tumbes": 0.35, "Apurimac": 0.45, "Amazonas": 0.50,
    "Loreto": 0.55, "Ucayali": 0.55, "Pasco": 0.55, "Madre de Dios": 0.60,
    "Huancavelica": 0.65,
}


def elegir_departamento():
    """Sortea un departamento segun cuanta gente usa la aplicacion en cada uno."""
    nombres = list(PESO_DEL_DEPARTAMENTO.keys())
    pesos = list(PESO_DEL_DEPARTAMENTO.values())
    return random.choices(nombres, weights=pesos, k=1)[0]


def probabilidad_de_fracasar(fila, departamento):
    """La regla con la que se decide como termina cada emprendimiento.

    No usa el modelo: es una regla aparte, escrita a mano, con las relaciones
    que el analisis de la Unidad I encontro en los datos. Si usara el modelo,
    la simulacion seria circular y el modelo "acertaria" por construccion.

    Los pesos estan en log-odds y las variables van estandarizadas para que
    ninguna pese de mas solo por tener numeros mas grandes.
    """
    def z(valor, promedio, desviacion):
        return (valor - promedio) / desviacion

    logit = 0.75                                                    # arranque
    logit -= 0.34 * z(fila["product_market_fit_score"], 4.61, 2.24)
    logit -= 0.28 * z(fila["runway_months"], 18.88, 8.14)
    logit -= 0.20 * z(fila["unit_economics_score"], 4.60, 2.20)
    logit -= 0.18 * z(fila["team_completeness"], 5.33, 2.16)
    logit -= 0.18 * z(fila["macro_climate"], 5.01, 1.98)
    logit += 0.20 * z(fila["cofounder_conflict"], 4.08, 2.22)
    logit += 0.22 * z(math.log10(max(fila["monthly_burn_rate"], 1)), 3.9, 0.7)

    # Y aca entra lo que el dataset original no tiene
    logit += EFECTO_DEL_DEPARTAMENTO[departamento]

    return 1 / (1 + math.exp(-logit))


def aplicar_deriva(fila, mes):
    """Cambia poco a poco el perfil de los emprendimientos que llegan.

    Simula dos cosas que pasan de verdad: el clima de inversion se enfria con
    el tiempo, y los costos suben. Es lo que el detector de deriva tendria que
    notar en algun momento.
    """
    fila = dict(fila)
    fila["macro_climate"] = max(0.0, fila["macro_climate"] - 0.09 * mes)
    fila["monthly_burn_rate"] = fila["monthly_burn_rate"] * (1.022 ** mes)
    return fila


def generar_consultas(datos, cuantas, mes, modelo):
    """Crea consultas nuevas y las guarda como si hubieran llegado por la web."""
    muestra = datos.sample(cuantas, replace=True, random_state=SEMILLA + mes)

    filas = []
    for _, original in muestra.iterrows():
        fila = aplicar_deriva(original, mes)
        departamento = elegir_departamento()

        variables = {}
        for columna in config.CARACTERISTICAS:
            if columna == "departamento":
                variables[columna] = departamento
            elif columna in config.CATEGORICAS:
                variables[columna] = str(fila[columna])
            else:
                variables[columna] = float(fila[columna])

        # El desenlace se decide ahora, pero recien se revela a los 10 meses
        fracasa = random.random() < probabilidad_de_fracasar(fila, departamento)
        filas.append({"variables": variables, "fracasa": fracasa})

    # Se predice en lote, que es mucho mas rapido que de a una
    tabla = pd.DataFrame([f["variables"] for f in filas])[config.CARACTERISTICAS]
    probabilidades = modelo["pipeline"].predict_proba(tabla)[:, 1]

    numeros = []
    for i, f in enumerate(filas):
        resultado = {
            "failure_probability": round(float(probabilidades[i]), 4),
            "at_risk": bool(probabilidades[i] >= modelo["threshold"]),
            "model_version": modelo["model_version"],
        }
        numero = almacen.guardar_consulta(f["variables"], resultado, "questionnaire")
        numeros.append({"id": numero, "fracasa": f["fracasa"]})

    return numeros


def main():
    print("=" * 66)
    print("SIMULACION DE DOS ANIOS DE OPERACION")
    print("=" * 66)

    # Base limpia: cada corrida empieza de cero
    if config.RUTA_CONSULTAS.exists():
        config.RUTA_CONSULTAS.unlink()

    # Se parte del modelo que esta hoy en produccion
    shutil.copy(RAIZ / "models" / "model.joblib", config.RUTA_MODELO)
    shutil.copy(RAIZ / "models" / "metrics.json", config.RUTA_METRICAS)

    datos = cargar_datos()
    modelo_actual = joblib.load(config.RUTA_MODELO)
    auc_original = json.loads(config.RUTA_METRICAS.read_text())["metrics"]["roc_auc"]
    print(f"\nModelo de partida: AUC {auc_original:.4f}, sin la variable departamento\n")

    pendientes_por_mes = {}
    bitacora = []

    for mes in range(MESES + 1):
        cuantas = CONSULTAS_DEL_PRIMER_MES if mes == 0 else CONSULTAS_POR_MES
        pendientes_por_mes[mes] = generar_consultas(datos, cuantas, mes, modelo_actual)

        # La incubadora completa los desenlaces de la camada de 10 meses atras
        mes_a_revelar = mes - MESES_HASTA_SABER_EL_DESENLACE
        revelados = 0
        if mes_a_revelar in pendientes_por_mes:
            for consulta in pendientes_por_mes[mes_a_revelar]:
                almacen.marcar_desenlace(consulta["id"], consulta["fracasa"])
                revelados += 1

        conteo = almacen.contar()
        linea = {
            "mes": mes,
            "consultas_del_mes": cuantas,
            "acumuladas": conteo["total"],
            "con_desenlace": conteo["con_desenlace"],
            "desenlaces_revelados_este_mes": revelados,
            "reentreno": False,
        }

        # ---------- Cada tanto corre el mantenimiento ----------
        if mes in MESES_EN_QUE_SE_REENTRENA:
            print(f"--- Mes {mes}: corriendo el mantenimiento "
                  f"({conteo['con_desenlace']} consultas con desenlace) ---")

            # Para que las cuatro corridas se distingan en MLflow
            os.environ["ETIQUETA_CORRIDA"] = f"simulacion-mes-{mes:02d}"
            resumen = mantenimiento.ejecutar()

            modelo_actual = joblib.load(config.RUTA_MODELO)

            # Las dos cifras salen del propio pipeline, que las mide sobre el
            # subconjunto que aparto y NO uso para entrenar. Antes la simulacion
            # las calculaba por su cuenta sobre todas las consultas etiquetadas,
            # incluidas las que el candidato acababa de ver, y por eso el numero
            # de "despues" salia inflado.
            comparacion = resumen["comparacion_sobre_consultas_reales"]
            auc_antes = None if comparacion is None else comparacion["auc_produccion"]
            auc_despues = None if comparacion is None else comparacion["auc_candidato"]

            linea.update({
                "reentreno": True,
                "deriva": resumen["deriva"]["proporcion"],
                "hay_deriva": resumen["deriva"]["hay_deriva"],
                "filas_reales_usadas": resumen["consultas_reales_en_el_entrenamiento"],
                "auc_del_candidato": resumen["candidato"]["roc_auc"],
                "promovido": resumen["promovido"],
                "auc_real_antes": auc_antes,
                "auc_real_despues": auc_despues,
            })

            print(f"    deriva: {100 * resumen['deriva']['proporcion']:.0f}% de las columnas")
            print(f"    promovido: {'si' if resumen['promovido'] else 'no'}")
            if auc_antes and auc_despues:
                print(f"    AUC sobre consultas reales: {auc_antes:.4f} -> {auc_despues:.4f}")
            print()

        bitacora.append(linea)

    guardar_resultados(bitacora, auc_original)


def guardar_resultados(bitacora, auc_original):
    """Escribe la bitacora y dibuja los graficos para el informe."""
    tabla = pd.DataFrame(bitacora)
    tabla.to_csv(CARPETA / "bitacora.csv", index=False)

    reentrenos = tabla[tabla["reentreno"]]

    # ---------- Grafico 1: como se acumulan los datos ----------
    plt.figure(figsize=(7, 3.8))
    plt.plot(tabla["mes"], tabla["acumuladas"], label="Assessments received", linewidth=2)
    plt.plot(tabla["mes"], tabla["con_desenlace"], label="With known outcome", linewidth=2)
    plt.axhline(500, color="#d1495b", linestyle="--", linewidth=1,
                label="Minimum to retrain (500)")
    plt.xlabel("Month")
    plt.ylabel("Assessments")
    plt.title("Data accumulated over two years")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CARPETA / "01_acumulacion.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---------- Grafico 2: el AUC sobre las consultas reales ----------
    if len(reentrenos) > 0 and reentrenos["auc_real_antes"].notna().any():
        plt.figure(figsize=(7, 3.8))
        ancho = 0.35
        x = np.arange(len(reentrenos))
        plt.bar(x - ancho / 2, reentrenos["auc_real_antes"], ancho,
                label="Model before retraining", color="#9aa5b1")
        plt.bar(x + ancho / 2, reentrenos["auc_real_despues"], ancho,
                label="Model after retraining", color="#2a9d8f")
        plt.xticks(x, [f"Month {m}" for m in reentrenos["mes"]])
        plt.ylabel("ROC-AUC on held-out real assessments")
        plt.ylim(0.5, 0.85)
        plt.title("Does retraining with real data help?")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CARPETA / "02_auc_datos_reales.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ---------- Grafico 3: la deriva ----------
    if len(reentrenos) > 0:
        plt.figure(figsize=(7, 3.5))
        plt.plot(reentrenos["mes"], 100 * reentrenos["deriva"], marker="o", linewidth=2,
                 color="#4a7ba7")
        plt.axhline(100 * 0.30, color="#d1495b", linestyle="--", linewidth=1,
                    label="Alert threshold (30%)")
        plt.xlabel("Month")
        plt.ylabel("Drifted columns (%)")
        plt.title("Data drift detected over time")
        plt.ylim(0, 105)
        plt.legend()
        plt.tight_layout()
        plt.savefig(CARPETA / "03_deriva.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ---------- Grafico 4: el efecto del departamento ----------
    filas = almacen.filas_para_entrenar()
    if filas:
        reales = pd.DataFrame(filas)
        por_departamento = (reales.groupby("departamento")[config.OBJETIVO]
                            .agg(["mean", "count"]))
        por_departamento = por_departamento[por_departamento["count"] >= 40]
        por_departamento = por_departamento.sort_values("mean")

        plt.figure(figsize=(7, 5))
        plt.barh(por_departamento.index, 100 * por_departamento["mean"], color="#4a7ba7")
        plt.axvline(100 * reales[config.OBJETIVO].mean(), color="#d1495b",
                    linestyle="--", label="Overall average")
        plt.xlabel("Failure rate (%)")
        plt.title("Failure rate by region, in the simulated data")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CARPETA / "04_por_departamento.png", dpi=120, bbox_inches="tight")
        plt.close()

    resumen = {
        "meses_simulados": MESES,
        "consultas_totales": int(tabla["acumuladas"].iloc[-1]),
        "con_desenlace": int(tabla["con_desenlace"].iloc[-1]),
        "auc_del_modelo_original": auc_original,
        "reentrenamientos": reentrenos.to_dict(orient="records"),
    }
    (CARPETA / "resumen.json").write_text(json.dumps(resumen, indent=2, default=str),
                                          encoding="utf-8")

    print("=" * 66)
    print(f"Consultas simuladas:   {resumen['consultas_totales']}")
    print(f"Con desenlace conocido: {resumen['con_desenlace']}")
    print(f"Resultados en:          {CARPETA}")
    print("=" * 66)


if __name__ == "__main__":
    main()
