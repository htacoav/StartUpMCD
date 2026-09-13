"""Pruebas de la API."""


def test_health_responde_ok(cliente):
    """El servicio esta vivo y con el modelo cargado."""
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True


def test_prediccion_valida(cliente, startup):
    """Caso feliz: una startup valida recibe una prediccion coherente."""
    respuesta = cliente.post("/predict", json=startup)
    assert respuesta.status_code == 200
    d = respuesta.json()

    assert 0.0 <= d["survival_probability"] <= 1.0
    # Las dos probabilidades son caras de la misma moneda y tienen que sumar 1
    assert abs(d["survival_probability"] + d["failure_probability"] - 1.0) < 0.01
    # La bandera tiene que ser consistente con la probabilidad y el umbral
    assert d["at_risk"] == (d["failure_probability"] >= d["threshold"])


def test_campo_faltante_es_rechazado(cliente, startup):
    """Si falta un dato obligatorio la API responde 422 en vez de adivinar."""
    del startup["product_market_fit_score"]
    assert cliente.post("/predict", json=startup).status_code == 422


def test_valor_fuera_de_escala_es_rechazado(cliente, startup):
    """Los puntajes van de 0 a 10, un 50 no puede llegar al modelo."""
    startup["product_market_fit_score"] = 50
    assert cliente.post("/predict", json=startup).status_code == 422

    startup["product_market_fit_score"] = -3
    assert cliente.post("/predict", json=startup).status_code == 422


def test_industria_nunca_vista_no_rompe(cliente, startup):
    """Una industria que no estaba en el entrenamiento no debe tumbar el servicio.

    En produccion van a aparecer sectores nuevos. El OneHotEncoder se configuro
    con handle_unknown='ignore' justamente para que la API siga respondiendo.
    """
    startup["industry"] = "quantum_computing"
    respuesta = cliente.post("/predict", json=startup)
    assert respuesta.status_code == 200
    assert 0.0 <= respuesta.json()["survival_probability"] <= 1.0


def test_prediccion_es_determinista(cliente, startup):
    """La misma startup tiene que dar siempre el mismo resultado."""
    primera = cliente.post("/predict", json=startup).json()
    segunda = cliente.post("/predict", json=startup).json()
    assert primera == segunda


def test_mejor_encaje_producto_mercado_sube_la_supervivencia(cliente, startup):
    """Prueba de coherencia del negocio, no solo del codigo.

    El encaje producto-mercado es la variable mas importante del modelo y la
    falta de encaje es la causa numero uno de fracaso segun el diccionario del
    dataset. En los datos, el fracaso cae de 78 a 52 por ciento al pasar del
    cuartil mas bajo al mas alto. Si un reentrenamiento invirtiera esta relacion,
    algo se rompio y esta prueba lo detecta.
    """
    startup["product_market_fit_score"] = 1.0
    malo = cliente.post("/predict", json=startup).json()["survival_probability"]

    startup["product_market_fit_score"] = 9.0
    bueno = cliente.post("/predict", json=startup).json()["survival_probability"]

    assert bueno > malo


def test_el_conflicto_entre_socios_baja_la_supervivencia(cliente, startup):
    """Otra relacion que el modelo tiene que respetar.

    En los datos el fracaso sube de 59 a 73 por ciento conforme crece el
    conflicto entre cofundadores.
    """
    startup["cofounder_conflict"] = 0.0
    sin_conflicto = cliente.post("/predict", json=startup).json()["survival_probability"]

    startup["cofounder_conflict"] = 10.0
    con_conflicto = cliente.post("/predict", json=startup).json()["survival_probability"]

    assert sin_conflicto > con_conflicto


def test_mas_caja_sube_la_supervivencia(cliente, startup):
    """Quedarse sin efectivo es la segunda causa de fracaso del dataset."""
    startup["runway_months"] = 3.0
    poca_caja = cliente.post("/predict", json=startup).json()["survival_probability"]

    startup["runway_months"] = 40.0
    mucha_caja = cliente.post("/predict", json=startup).json()["survival_probability"]

    assert mucha_caja > poca_caja


def test_las_dos_paginas_cargan_en_ingles(cliente):
    """Las dos vistas responden y estan en ingles, como pide la rubrica.

    La raiz es el cuestionario para emprendedores y /expert es la vista tecnica
    con los 17 valores numericos, pensada para un analista.
    """
    cuestionario = cliente.get("/")
    assert cuestionario.status_code == 200
    assert "Will your startup survive?" in cuestionario.text
    assert "See my result" in cuestionario.text

    tecnica = cliente.get("/expert")
    assert tecnica.status_code == 200
    assert "Startup Survival Prediction" in tecnica.text
    assert "Predict survival" in tecnica.text


def test_ficha_del_modelo_expone_metricas(cliente):
    """El endpoint de monitoreo publica version y metricas reales."""
    d = cliente.get("/model-info").json()
    assert d["model_version"]
    assert d["metrics"]["roc_auc"] > 0.70
    assert d["training_rows"] == 48000


def test_el_estudio_de_grupos_esta_publicado(cliente):
    """El analisis que responde la pregunta del proyecto se expone por la API."""
    d = cliente.get("/feature-groups").json()
    assert "grupos" in d
    assert len(d["grupos"]) == 6
    # La ejecucion tiene que predecir mejor que el perfil del fundador
    assert (d["grupos"]["execution"]["auc_solo_este_grupo"]
            > d["grupos"]["founder"]["auc_solo_este_grupo"])
