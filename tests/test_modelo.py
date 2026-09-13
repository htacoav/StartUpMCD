"""Pruebas del artefacto del modelo, sin pasar por la API."""

import joblib
import pandas as pd
import pytest

from ml import config


@pytest.fixture(scope="module")
def artefacto():
    if not config.RUTA_MODELO.exists():
        pytest.skip("no hay modelo entrenado, corre python -m ml.train")
    return joblib.load(config.RUTA_MODELO)


def test_el_artefacto_trae_todo_lo_necesario(artefacto):
    """Sin estos campos la API no puede armar el formulario ni versionar."""
    for clave in ("pipeline", "model_version", "features", "groups",
                  "categorical_options", "numeric_ranges", "threshold",
                  "metrics", "group_study", "trained_at"):
        assert clave in artefacto


def test_la_calidad_no_bajo_del_minimo(artefacto):
    """Puerta de calidad: si un reentrenamiento degrada el modelo, esto falla.

    Este mismo criterio lo usara el pipeline de mantenimiento de la Unidad II
    para decidir si el modelo nuevo se promueve a produccion o se descarta.

    Se vigilan el AUC y la exactitud balanceada, no el F1. El motivo es que el
    65.8 por ciento de las startups fracasa, y con ese desbalance el F1 se puede
    inflar sin que el modelo aporte nada: declarar que todas fracasan, sin
    modelo alguno, ya da F1 de 0.7937. La exactitud balanceada no se deja
    enganar asi, porque un modelo que siempre responde lo mismo saca 0.50.
    """
    m = artefacto["metrics"]
    assert m["roc_auc"] >= 0.70, f"el AUC cayo a {m['roc_auc']}"
    assert m["balanced_accuracy"] >= 0.65, \
        f"la exactitud balanceada cayo a {m['balanced_accuracy']}"


def test_el_modelo_le_gana_a_no_tener_modelo(artefacto):
    """El modelo tiene que aportar algo sobre la respuesta trivial.

    Si un modelo marcara casi todas las startups como fracaso, sus metricas se
    pareceran a las de no tener modelo. Esta prueba lo detecta por dos vias: la
    exactitud balanceada tiene que superar el 0.50 de un clasificador que
    siempre responde lo mismo, y la proporcion de startups marcadas no puede
    dispararse al 90 por ciento.
    """
    m = artefacto["metrics"]
    assert m["balanced_accuracy"] > 0.55, "el modelo no distingue mejor que el azar"
    assert m["flagged_rate"] < 0.80, \
        f"marca el {100*m['flagged_rate']:.0f} por ciento de los casos, esta prediciendo siempre lo mismo"


def test_no_se_colo_ninguna_columna_con_fuga(artefacto):
    """La prueba mas importante de todas.

    Columnas como 'outcome' o 'failure_reason' solo se conocen despues de que
    la startup ya fracaso. Si alguna se colara entre las variables del modelo,
    las metricas se dispararian y el modelo seria inutil en la practica, porque
    en produccion esos datos no existen todavia.
    """
    for columna in config.COLUMNAS_CON_FUGA:
        assert columna not in artefacto["features"], f"{columna} no puede ser una variable"


def test_las_variables_coinciden_con_los_grupos(artefacto):
    """La suma de los grupos tiene que dar exactamente la lista de variables."""
    del_grupo = []
    for columnas in artefacto["groups"].values():
        del_grupo.extend(columnas)
    assert sorted(del_grupo) == sorted(artefacto["features"])


def test_el_pipeline_predice_sobre_una_fila(artefacto):
    """El modelo acepta una sola fila, que es como le llegan los datos en produccion."""
    fila = {}
    for columna in config.CATEGORICAS:
        fila[columna] = artefacto["categorical_options"][columna][0]
    for columna in config.NUMERICAS:
        fila[columna] = artefacto["numeric_ranges"][columna]["median"]

    probabilidades = artefacto["pipeline"].predict_proba(
        pd.DataFrame([fila])[artefacto["features"]])
    assert probabilidades.shape == (1, 2)
    assert 0.0 <= probabilidades[0, 1] <= 1.0
