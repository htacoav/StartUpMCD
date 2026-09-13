# ==========================================
# ENTRENAR EL MODELO
# Ejecutar:  python -m ml.train
#
# El programa va en 7 pasos, en orden:
#   1. cargar los datos
#   2. partirlos en tres
#   3. preparar las columnas para el modelo
#   4. probar tres modelos y quedarse con el mejor
#   5. elegir el punto de corte
#   6. medir cuanto aporta cada grupo de variables
#   7. guardar todo
# ==========================================

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config
from ml.data import cargar_datos, opciones_de_formulario, preparar, rangos_numericos

VERSION_MODELO = "1.0.0"


# Los tres modelos que se van a probar, cada uno con los valores que se le
# probaran. Si "opciones" esta vacio, el modelo se entrena tal cual.
MODELOS = [
    {
        "nombre": "LogisticRegression",
        "modelo": LogisticRegression(max_iter=3000, random_state=config.SEMILLA),
        "opciones": {"modelo__C": [0.01, 0.1, 1.0, 10.0]},
    },
    {
        "nombre": "RandomForest",
        "modelo": RandomForestClassifier(random_state=config.SEMILLA, n_jobs=-1),
        "opciones": {
            "modelo__n_estimators": [200, 400],
            "modelo__max_depth": [6, 10, 16, None],
            "modelo__min_samples_leaf": [5, 20, 50],
        },
    },
    {
        "nombre": "HistGradientBoosting",
        "modelo": HistGradientBoostingClassifier(random_state=config.SEMILLA),
        "opciones": {
            "modelo__max_depth": [3, 6, None],
            "modelo__learning_rate": [0.05, 0.1],
            "modelo__max_iter": [200, 400],
            "modelo__min_samples_leaf": [20, 50],
        },
    },
]


def preparar_columnas(categoricas, numericas):
    """Deja las columnas listas para que el modelo las entienda.

    Las categoricas, como industry, se convierten en una columna de ceros y
    unos por cada valor posible. Un modelo no sabe que hacer con la palabra
    "fintech", pero si con un 1 en la columna "es fintech".

    El handle_unknown="ignore" es importante: si manana aparece una industria
    nueva que no estaba al entrenar, la API responde igual en vez de romperse.

    Las numericas se estandarizan, o sea que se les resta su promedio y se
    dividen entre su desviacion. Hace falta porque total_raised_usd llega a 52
    millones y product_market_fit_score va de 0 a 10. Sin estandarizar, la
    regresion logistica le daria mucha mas importancia a la primera solo porque
    sus numeros son mas grandes.
    """
    convertir_categoricas = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    convertir_numericas = StandardScaler()
    return ColumnTransformer([
        ("categoricas", convertir_categoricas, categoricas),
        ("numericas", convertir_numericas, numericas),
    ])


def elegir_umbral(y_real, probabilidades):
    """Elige a partir de que probabilidad se considera que la startup fracasara.

    Por que no se usa 0.5, que seria lo obvio
    -----------------------------------------
    En este dataset el 65.8 por ciento de las startups fracasa. Con tantos casos
    de un lado, un modelo puede sacar buenas notas simplemente diciendo que
    todas fracasan. De hecho se probo elegir el umbral maximizando F1 y salio
    0.37: con ese corte el modelo marcaba el 90.7 por ciento de los casos y daba
    F1 de 0.8084. Suena bien, pero decir "todas fracasan" sin ningun modelo ya
    da 0.7937. O sea que el modelo no estaba aportando casi nada.

    Por eso se usa el estadistico J de Youden, que es la sensibilidad mas la
    especificidad menos uno. Ese numero obliga a acertar en las dos clases a la
    vez, asi que no se puede hacer trampa prediciendo siempre lo mismo. Con este
    criterio el umbral queda en 0.64 y el modelo marca el 57.5 por ciento.

    El umbral se busca con los datos de validacion, nunca con los de prueba,
    porque si no la nota final saldria inflada.
    """
    falsos_positivos, verdaderos_positivos, umbrales = roc_curve(y_real, probabilidades)
    puntaje_j = verdaderos_positivos - falsos_positivos
    mejor = np.argmax(puntaje_j)
    return round(float(umbrales[mejor]), 2)


def medir(y_real, probabilidades, umbral):
    """Calcula todas las notas del modelo con un punto de corte dado."""
    y_predicho = (probabilidades >= umbral).astype(int)
    matriz = confusion_matrix(y_real, y_predicho)
    verdaderos_negativos = int(matriz[0][0])
    falsos_positivos = int(matriz[0][1])
    falsos_negativos = int(matriz[1][0])
    verdaderos_positivos = int(matriz[1][1])

    return {
        "roc_auc": float(roc_auc_score(y_real, probabilidades)),
        "pr_auc": float(average_precision_score(y_real, probabilidades)),
        "accuracy": float(accuracy_score(y_real, y_predicho)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_predicho)),
        "precision": float(precision_score(y_real, y_predicho, zero_division=0)),
        "recall": float(recall_score(y_real, y_predicho, zero_division=0)),
        "f1": float(f1_score(y_real, y_predicho, zero_division=0)),
        "flagged_rate": float(y_predicho.mean()),
        "true_negatives": verdaderos_negativos,
        "false_positives": falsos_positivos,
        "false_negatives": falsos_negativos,
        "true_positives": verdaderos_positivos,
    }


def auc_usando(columnas, X, y):
    """Entrena un modelo simple con las columnas que se le pasen y devuelve su AUC.

    Se usa en el paso 6 para medir grupos de variables, entrenando una vez con
    cada grupo. Siempre usa regresion logistica para que la comparacion entre
    grupos sea justa.
    """
    categoricas = []
    numericas = []
    for columna in columnas:
        if columna in config.CATEGORICAS:
            categoricas.append(columna)
        else:
            numericas.append(columna)

    # Si un grupo no tiene columnas de algun tipo, ese paso no se agrega
    pasos = []
    if categoricas:
        pasos.append(("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas))
    if numericas:
        pasos.append(("numericas", StandardScaler(), numericas))

    modelo = Pipeline([
        ("preparacion", ColumnTransformer(pasos)),
        ("modelo", LogisticRegression(max_iter=3000, random_state=config.SEMILLA)),
    ])

    X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(
        X[columnas], y, test_size=0.2, random_state=config.SEMILLA, stratify=y)
    modelo.fit(X_entrenar, y_entrenar)
    probabilidades = modelo.predict_proba(X_probar)[:, 1]
    return float(roc_auc_score(y_probar, probabilidades))


def entrenar():
    reloj = time.time()

    # ---------- PASO 1: cargar los datos ----------
    tabla = cargar_datos()
    X, y = preparar(tabla)
    print(f"Datos: {len(X)} startups, {len(config.CARACTERISTICAS)} variables")
    print(f"Fracasaron: {y.sum()} ({100 * y.mean():.1f} por ciento)\n")

    # ---------- PASO 2: partir los datos en tres ----------
    # Se parte en tres y no en dos porque hay dos decisiones distintas que tomar
    # y no pueden usar los mismos datos: los hiperparametros se buscan con el
    # conjunto de entrenamiento y el umbral con el de validacion. El de prueba
    # no se toca hasta el final, para que la nota sea honesta.
    X_entrenar, X_sobrante, y_entrenar, y_sobrante = train_test_split(
        X, y, test_size=0.4, random_state=config.SEMILLA, stratify=y)
    X_validar, X_probar, y_validar, y_probar = train_test_split(
        X_sobrante, y_sobrante, test_size=0.5, random_state=config.SEMILLA, stratify=y_sobrante)
    print(f"Particion: {len(X_entrenar)} entrenamiento, {len(X_validar)} validacion, "
          f"{len(X_probar)} prueba\n")

    # ---------- PASO 3 y 4: probar los tres modelos ----------
    validacion_cruzada = StratifiedKFold(5, shuffle=True, random_state=config.SEMILLA)
    resultados = {}

    # Aca se va guardando el mejor que se haya visto hasta el momento
    mejor_nombre = ""
    mejor_modelo = None
    mejor_auc = 0.0
    mejor_umbral = 0.5
    mejores_opciones = {}

    for candidato in MODELOS:
        nombre = candidato["nombre"]
        inicio = time.time()

        modelo = Pipeline([
            ("preparacion", preparar_columnas(config.CATEGORICAS, config.NUMERICAS)),
            ("modelo", candidato["modelo"]),
        ])

        # Se prueban varias combinaciones de hiperparametros y se queda la mejor.
        # Es busqueda aleatoria y no exhaustiva porque probar todas las
        # combinaciones tardaria muchisimo y casi siempre son pocos los
        # hiperparametros que de verdad cambian el resultado.
        busqueda = RandomizedSearchCV(
            modelo,
            candidato["opciones"],
            n_iter=6,
            cv=validacion_cruzada,
            scoring="roc_auc",
            random_state=config.SEMILLA,
            n_jobs=-1,
        )
        busqueda.fit(X_entrenar, y_entrenar)
        modelo = busqueda.best_estimator_

        # Los nombres vienen como "modelo__C" y se les quita el prefijo
        opciones_ganadoras = {}
        for clave in busqueda.best_params_:
            nombre_corto = clave.split("__")[-1]
            opciones_ganadoras[nombre_corto] = busqueda.best_params_[clave]

        # ---------- PASO 5: elegir el umbral ----------
        probabilidades_validacion = modelo.predict_proba(X_validar)[:, 1]
        umbral = elegir_umbral(y_validar, probabilidades_validacion)

        probabilidades_prueba = modelo.predict_proba(X_probar)[:, 1]
        notas = medir(y_probar, probabilidades_prueba, umbral)

        notas["cv_roc_auc"] = float(busqueda.best_score_)
        notas["hyperparameters"] = opciones_ganadoras
        notas["threshold"] = umbral
        notas["seconds"] = round(time.time() - inicio, 1)
        resultados[nombre] = notas

        print(f"{nombre:<22} AUC={notas['roc_auc']:.4f}  "
              f"balanceada={notas['balanced_accuracy']:.4f}  "
              f"precision={notas['precision']:.4f}  recall={notas['recall']:.4f}  "
              f"umbral={umbral}  marca {100 * notas['flagged_rate']:.0f}%  "
              f"({notas['seconds']}s)")

        if notas["roc_auc"] > mejor_auc:
            mejor_nombre = nombre
            mejor_modelo = modelo
            mejor_auc = notas["roc_auc"]
            mejor_umbral = umbral
            mejores_opciones = opciones_ganadoras

    print(f"\nGana {mejor_nombre} con AUC {mejor_auc:.4f}")

    # ---------- PASO 6: medir cada grupo de variables ----------
    # Esta es la pregunta del proyecto: que caracteristicas anticipan la
    # supervivencia. Se mide de dos formas que se complementan:
    #   solo:  se entrena usando nada mas ese grupo, para ver cuanto predice solo
    #   sin:   se entrena con todo menos ese grupo, y lo que se pierde dice
    #          cuanta informacion aportaba que no estuviera ya en los demas
    print("\nMidiendo cuanto aporta cada grupo de variables...")
    auc_con_todo = auc_usando(config.CARACTERISTICAS, X, y)
    estudio = {"auc_con_todo": auc_con_todo, "grupos": {}}

    for grupo in config.GRUPOS:
        columnas_del_grupo = config.GRUPOS[grupo]

        # Las columnas que quedan si se saca este grupo
        columnas_restantes = []
        for columna in config.CARACTERISTICAS:
            if columna not in columnas_del_grupo:
                columnas_restantes.append(columna)

        auc_solo = auc_usando(columnas_del_grupo, X, y)
        auc_sin = auc_usando(columnas_restantes, X, y)

        estudio["grupos"][grupo] = {
            "nombre": config.NOMBRES_DE_GRUPO[grupo],
            "variables": columnas_del_grupo,
            "auc_solo_este_grupo": auc_solo,
            "auc_sin_este_grupo": auc_sin,
            "cuanto_se_pierde_al_quitarlo": auc_con_todo - auc_sin,
        }
        print(f"  {config.NOMBRES_DE_GRUPO[grupo]:<24} solo={auc_solo:.4f}   "
              f"al quitarlo se pierde {auc_con_todo - auc_sin:+.4f}")

    # ---------- PASO 7: guardar ----------
    # El artefacto lleva el modelo y ademas todo lo que la aplicacion necesita
    # para armar sus formularios sola, sin escribir nada a mano.
    artefacto = {
        "pipeline": mejor_modelo,
        "model_version": VERSION_MODELO,
        "model_name": mejor_nombre,
        "hyperparameters": mejores_opciones,
        "features": config.CARACTERISTICAS,
        "groups": config.GRUPOS,
        "group_names": config.NOMBRES_DE_GRUPO,
        "categorical_options": opciones_de_formulario(X),
        "numeric_ranges": rangos_numericos(X),
        "threshold": mejor_umbral,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(X)),
        "failure_rate": float(y.mean()),
        "metrics": resultados[mejor_nombre],
        "group_study": estudio,
    }

    config.RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, config.RUTA_MODELO)

    # El JSON lleva solo los numeros, sin el modelo, para que el informe y el
    # pipeline de mantenimiento puedan leerlo sin necesitar joblib.
    reporte = {}
    for clave in artefacto:
        if clave not in ("pipeline", "categorical_options", "numeric_ranges"):
            reporte[clave] = artefacto[clave]
    reporte["all_models"] = resultados
    reporte["training_seconds"] = round(time.time() - reloj, 1)

    config.RUTA_METRICAS.write_text(json.dumps(reporte, indent=2), encoding="utf-8")

    print(f"\nModelo guardado en    {config.RUTA_MODELO}")
    print(f"Metricas guardadas en {config.RUTA_METRICAS}")
    print(f"Tiempo total: {reporte['training_seconds']}s")
    return artefacto


if __name__ == "__main__":
    entrenar()
