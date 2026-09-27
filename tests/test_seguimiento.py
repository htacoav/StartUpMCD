"""Pruebas del guardado de consultas y del seguimiento de desenlaces.

Esta es la parte que hace posible reentrenar con datos reales. Sin ella, el
pipeline de mantenimiento solo puede trabajar con datos simulados.
"""

import pytest

from app import almacen
from ml import config
from ml.mantenimiento import (MINIMO_DE_FILAS_ETIQUETADAS, datos_para_entrenar,
                             separar_consultas_reales)


def test_cada_prediccion_queda_guardada(cliente, startup):
    """Al evaluar una startup se devuelve el id con el que quedo guardada."""
    antes = almacen.contar()["total"]

    respuesta = cliente.post("/predict", json=startup).json()

    assert respuesta["consultation_id"] > 0
    assert almacen.contar()["total"] == antes + 1


def test_el_cuestionario_tambien_guarda(cliente, respuestas):
    """La via del cuestionario guarda igual que la tecnica, con otro origen."""
    numero = cliente.post("/assess", json=respuestas).json()["consultation_id"]

    guardadas = almacen.listar(limite=5)
    esta = [c for c in guardadas if c["id"] == numero][0]

    assert esta["origen"] == "questionnaire"
    # Se guardan las 17 variables que recibio el modelo, no las respuestas crudas
    assert len(esta["variables"]) == len(config.CARACTERISTICAS)


def test_marcar_el_desenlace_la_saca_de_pendientes(cliente, startup):
    """Cuando se sabe como termino, esa consulta deja de estar pendiente."""
    numero = cliente.post("/predict", json=startup).json()["consultation_id"]

    pendientes = [c["id"] for c in almacen.listar(solo_sin_desenlace=True, limite=500)]
    assert numero in pendientes

    respuesta = cliente.post("/outcome", json={"consultation_id": numero, "failed": True})
    assert respuesta.status_code == 200

    pendientes = [c["id"] for c in almacen.listar(solo_sin_desenlace=True, limite=500)]
    assert numero not in pendientes


def test_si_no_se_puede_guardar_igual_se_responde(cliente, startup, monkeypatch):
    """Un fallo al guardar no puede tumbar la prediccion.

    Esto salio de produccion: el volumen de consultas quedo con permisos que el
    contenedor no podia escribir, y como guardar era obligatorio, todas las
    predicciones devolvian error 500. La aplicacion entera dejo de funcionar
    por algo secundario.
    """
    def revienta(*args, **kwargs):
        raise OSError("attempt to write a readonly database")

    monkeypatch.setattr(almacen, "guardar_consulta", revienta)

    respuesta = cliente.post("/predict", json=startup)

    assert respuesta.status_code == 200
    d = respuesta.json()
    assert 0.0 <= d["survival_probability"] <= 1.0
    # Se informa que no quedo guardada, en vez de mentir con un numero
    assert d["consultation_id"] is None


def test_una_consulta_que_no_existe_da_404(cliente):
    """Marcar el desenlace de algo inexistente no puede pasar en silencio."""
    respuesta = cliente.post("/outcome", json={"consultation_id": 999999, "failed": False})
    assert respuesta.status_code == 404


def test_la_consulta_etiquetada_sirve_para_entrenar(cliente, startup):
    """Una consulta con desenlace se convierte en una fila con su respuesta correcta.

    Es el punto de todo el seguimiento: sin el desenlace la fila no sirve,
    porque el modelo no tendria de que aprender.
    """
    numero = cliente.post("/predict", json=startup).json()["consultation_id"]
    cliente.post("/outcome", json={"consultation_id": numero, "failed": True})

    filas = almacen.filas_para_entrenar()
    assert len(filas) >= 1

    fila = filas[-1]
    assert fila[config.OBJETIVO] in (0, 1)
    for variable in config.CARACTERISTICAS:
        assert variable in fila


def test_no_se_usan_pocas_consultas_para_entrenar(cliente, startup):
    """Con pocas filas reales, el entrenamiento sigue usando solo el dataset.

    Agregar veinte filas sobre 48 000 no cambiaria el modelo, pero si haria
    imposible explicar de donde salio cada numero del informe.
    """
    import pandas as pd

    assert almacen.contar()["con_desenlace"] < MINIMO_DE_FILAS_ETIQUETADAS

    para_entrenar, para_examen = separar_consultas_reales()
    assert para_entrenar is None
    assert para_examen is None

    datos = pd.DataFrame([{**startup, config.OBJETIVO: 1}] * 10)
    resultado, cuantas = datos_para_entrenar(datos, para_entrenar)
    assert cuantas == 0
    assert len(resultado) == len(datos)


def test_la_vista_de_seguimiento_carga(cliente, startup):
    """La pagina donde la incubadora marca los desenlaces responde."""
    cliente.post("/predict", json=startup)

    pagina = cliente.get("/admin")
    assert pagina.status_code == 200
    assert "Follow-up" in pagina.text
    assert "Shut down" in pagina.text
    assert "Still running" in pagina.text


def test_sin_token_no_se_ven_las_consultas_ajenas(cliente, monkeypatch):
    """Con un token configurado, las rutas de seguimiento quedan cerradas.

    Estas rutas exponen los datos que la gente ingreso y permiten modificar
    registros, asi que en produccion no pueden quedar abiertas.
    """
    monkeypatch.setattr(config, "TOKEN_ADMIN", "una-clave-secreta")

    assert cliente.get("/consultations").status_code == 401
    assert cliente.get("/admin").status_code == 401
    assert cliente.post("/outcome", json={"consultation_id": 1, "failed": True}).status_code == 401

    assert cliente.get("/consultations?token=una-clave-secreta").status_code == 200


@pytest.fixture
def respuestas():
    """Un juego completo de respuestas del cuestionario."""
    return {
        "paying_customers": 2,
        "talked_to_customers": 1,
        "margin_per_sale": 2,
        "how_customers_find_you": 1,
        "scaled_early": 0,
        "founder_disagreements": 1,
        "team_skills": ["Product or engineering", "Sales"],
        "founded_before": 0,
        "years_in_industry": 4.0,
        "how_many_customers": 2,
        "how_many_competitors": 1,
        "fundraising_climate": 2,
        "industry": "tech_saas",
        "total_raised_usd": 100000.0,
        "cash_available_usd": 50000.0,
        "monthly_spend_usd": 5000.0,
        "funding_path": "angel",
    }
