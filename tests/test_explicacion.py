"""Pruebas de la explicacion SHAP y de la pagina en espanol."""

import math

import joblib
import pandas as pd
import pytest

from app import textos_es
from app.explicacion import explicar
from app.questionnaire import PREGUNTAS
from ml import config


@pytest.fixture(scope="module")
def artefacto():
    return joblib.load(config.RUTA_MODELO)


def test_los_shap_suman_exactamente_la_prediccion(artefacto, startup):
    """La propiedad que hace confiable a SHAP: no es una aproximacion.

    El punto de partida mas todos los empujones tiene que dar exactamente lo que
    predijo el modelo. Si no suma, la explicacion le estaria mintiendo al usuario.
    """
    fila = pd.DataFrame([startup])[artefacto["features"]]
    explicacion = explicar(artefacto, fila)

    probabilidad_de_fracaso = artefacto["pipeline"].predict_proba(fila)[0][1]
    log_odds_del_modelo = math.log(probabilidad_de_fracaso / (1 - probabilidad_de_fracaso))

    # Los impact vienen con el signo cambiado, porque hablan de supervivencia
    suma = explicacion["base_failure_log_odds"]
    for factor in explicacion["factors"]:
        suma -= factor["impact"]

    # Se tolera el redondeo a 4 decimales de cada uno de los 17 valores
    assert abs(suma - log_odds_del_modelo) < 0.002


def test_coincide_con_la_libreria_shap(artefacto, startup):
    """Compara la formula propia contra shap.LinearExplainer.

    La libreria no esta en requirements.txt para no engordar la imagen de Docker,
    asi que si no esta instalada la prueba se salta en vez de fallar.
    """
    shap = pytest.importorskip("shap")

    datos = pd.read_csv(config.RUTA_DATOS)
    preparacion = artefacto["pipeline"].named_steps["preparacion"]
    modelo = artefacto["pipeline"].named_steps["modelo"]

    # Se reconstruye el mismo conjunto de entrenamiento que uso train.py
    from sklearn.model_selection import train_test_split
    from ml.data import preparar
    X, y = preparar(datos)
    X_entrenar, _, _, _ = train_test_split(X, y, test_size=0.4, random_state=config.SEMILLA, stratify=y)
    fondo = preparacion.transform(X_entrenar)

    # Ojo: LinearExplainer, si se le pasa la tabla sin mas, toma solo 100 filas al
    # azar como referencia y sus numeros salen corridos hasta en 0.07. Hay que
    # pedirle explicitamente que use las 28 800 filas, igual que train.py.
    fondo_completo = shap.maskers.Independent(fondo, max_samples=len(fondo))

    fila = pd.DataFrame([startup])[artefacto["features"]]
    oficial = shap.LinearExplainer(modelo, fondo_completo).shap_values(preparacion.transform(fila))[0]

    # La libreria da un valor por columna transformada: se juntan por variable
    from app.explicacion import variable_original
    oficial_por_variable = {}
    for nombre, valor in zip(preparacion.get_feature_names_out(), oficial):
        variable = variable_original(nombre)
        oficial_por_variable[variable] = oficial_por_variable.get(variable, 0.0) + valor

    propia = explicar(artefacto, fila)
    for factor in propia["factors"]:
        assert abs(-factor["impact"] - oficial_por_variable[factor["feature"]]) < 0.0001


def test_la_api_explica_todas_las_variables_en_orden(cliente, startup):
    """La respuesta trae una explicacion por variable, de la que mas pesa a la que menos.

    La cantidad se toma de config y no se escribe a mano, asi la prueba sigue
    valiendo cuando el modelo incorpora una variable nueva.
    """
    explicacion = cliente.post("/predict", json=startup).json()["explanation"]

    assert len(explicacion["factors"]) == len(config.CARACTERISTICAS)
    pesos = [abs(f["impact"]) for f in explicacion["factors"]]
    assert pesos == sorted(pesos, reverse=True)


def test_la_explicacion_apunta_hacia_donde_debe(cliente, startup):
    """Si el encaje producto-mercado es muy bueno, tiene que aparecer ayudando.

    Y si es muy malo, perjudicando. Es la misma coherencia de negocio que se le
    exige al modelo, pero ahora a la explicacion.
    """
    startup["product_market_fit_score"] = 9.0
    bueno = cliente.post("/predict", json=startup).json()["explanation"]["factors"]
    startup["product_market_fit_score"] = 1.0
    malo = cliente.post("/predict", json=startup).json()["explanation"]["factors"]

    def impacto(factores):
        for f in factores:
            if f["feature"] == "product_market_fit_score":
                return f["impact"]

    assert impacto(bueno) > 0
    assert impacto(malo) < 0


def test_los_textos_en_espanol_cubren_todas_las_preguntas():
    """Cada pregunta tiene su traduccion y la misma cantidad de opciones.

    Esto importa mas de lo que parece: la pagina manda el numero de la opcion
    elegida. Si en espanol faltara una opcion o estuvieran en otro orden, el
    usuario elegiria una cosa y el modelo recibiria otra, sin ningun error.
    """
    for pregunta in PREGUNTAS:
        assert pregunta["id"] in textos_es.PREGUNTAS
        assert len(textos_es.PREGUNTAS[pregunta["id"]]["options"]) == len(pregunta["options"])
        assert pregunta["section"] in textos_es.SECCIONES

    for variable in config.CARACTERISTICAS:
        assert variable in textos_es.VARIABLES


def test_la_pagina_en_espanol_carga(cliente, artefacto):
    """La version en espanol responde y trae todos los sectores traducidos."""
    pagina = cliente.get("/es")
    assert pagina.status_code == 200
    assert "¿Tu emprendimiento va a sobrevivir?" in pagina.text
    assert "Ver mi resultado" in pagina.text
    assert "¿Por qué este resultado?" in pagina.text

    for sector in artefacto["categorical_options"]["industry"]:
        assert sector in textos_es.SECTORES
