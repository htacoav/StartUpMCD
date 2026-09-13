"""Pruebas del cuestionario para usuarios finales."""

import pytest

from app.questionnaire import PREGUNTAS, traducir
from ml import config


@pytest.fixture
def respuestas():
    """Un emprendedor promedio respondiendo el cuestionario."""
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


def test_la_traduccion_genera_las_17_variables(respuestas):
    """Si faltara una variable el modelo fallaria de forma rara."""
    variables = traducir(respuestas)
    assert sorted(variables.keys()) == sorted(config.CARACTERISTICAS)


def test_el_runway_se_calcula_no_se_pregunta(respuestas):
    """Nadie sabe de memoria su runway, pero si cuanto tiene y cuanto gasta.

    50 000 de caja entre 5 000 de gasto mensual son 10 meses.
    """
    assert traducir(respuestas)["runway_months"] == 10.0


def test_sin_gastos_el_runway_es_el_maximo(respuestas):
    """Una startup que no gasta nada no se queda sin caja. Ademas hay que evitar
    la division entre cero."""
    respuestas["monthly_spend_usd"] = 0
    assert traducir(respuestas)["runway_months"] == 50.0


def test_mas_areas_cubiertas_mas_equipo_completo(respuestas):
    """El puntaje de equipo sube con cada area que el equipo cubre."""
    respuestas["team_skills"] = []
    sin_nadie = traducir(respuestas)["team_completeness"]
    respuestas["team_skills"] = ["Product or engineering", "Sales", "Operations", "Finance"]
    completo = traducir(respuestas)["team_completeness"]
    assert completo > sin_nadie


def test_cada_pregunta_alimenta_una_variable_del_modelo():
    """Ninguna pregunta puede apuntar a una variable que el modelo no usa."""
    for pregunta in PREGUNTAS:
        assert pregunta["variable"] in config.CARACTERISTICAS


def test_las_opciones_van_de_peor_a_mejor():
    """El orden importa para que la pagina se lea bien.

    Se revisa que los valores de cada pregunta esten ordenados, en un sentido o
    en el otro, y no mezclados al azar.
    """
    for pregunta in PREGUNTAS:
        valores = [o["value"] for o in pregunta["options"]]
        assert valores == sorted(valores) or valores == sorted(valores, reverse=True), \
            f"las opciones de {pregunta['id']} no estan ordenadas"


def test_el_cuestionario_se_publica_por_la_api(cliente):
    """La pagina se arma con lo que devuelve este endpoint."""
    d = cliente.get("/questionnaire").json()
    assert len(d["sections"]) == 4
    assert len(d["team_areas"]) == 4


def test_evaluacion_valida(cliente, respuestas):
    """Caso feliz del cuestionario."""
    r = cliente.post("/assess", json=respuestas)
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["survival_probability"] <= 1.0
    # Se devuelven las variables usadas para que el resultado sea auditable
    assert len(d["derived_features"]) == 17


def test_opcion_inexistente_es_rechazada(cliente, respuestas):
    """Si la opcion elegida no existe, la API responde 422 en vez de reventar."""
    respuestas["paying_customers"] = 9
    assert cliente.post("/assess", json=respuestas).status_code == 422


def test_las_dos_rutas_dan_el_mismo_resultado(cliente, respuestas):
    """La prueba mas importante de esta vista.

    /assess traduce respuestas a numeros y luego usa el mismo modelo que
    /predict. Si las dos rutas dieran resultados distintos para la misma
    startup, la aplicacion seria incoherente y no se podria confiar en ninguna.
    """
    por_cuestionario = cliente.post("/assess", json=respuestas).json()

    # Se toman las variables que salieron de la traduccion y se mandan a /predict
    directo = cliente.post("/predict", json=por_cuestionario["derived_features"]).json()

    assert por_cuestionario["survival_probability"] == directo["survival_probability"]
    assert por_cuestionario["risk_level"] == directo["risk_level"]


def test_una_startup_sana_supera_a_una_en_problemas(cliente, respuestas):
    """Coherencia de negocio de punta a punta, respondiendo el cuestionario."""
    mala = dict(respuestas)
    mala.update({"paying_customers": 0, "margin_per_sale": 0, "founder_disagreements": 3,
                 "scaled_early": 1, "team_skills": [], "cash_available_usd": 10000.0})
    buena = dict(respuestas)
    buena.update({"paying_customers": 3, "margin_per_sale": 3, "founder_disagreements": 0,
                  "scaled_early": 0, "team_skills": ["Product or engineering", "Sales",
                                                     "Operations", "Finance"],
                  "cash_available_usd": 400000.0})

    p_mala = cliente.post("/assess", json=mala).json()["survival_probability"]
    p_buena = cliente.post("/assess", json=buena).json()["survival_probability"]
    assert p_buena > p_mala
