# ==========================================
# EL CUESTIONARIO
#
# El modelo necesita numeros, como "product_market_fit_score de 0 a 10", pero un
# emprendedor no sabe cual es el suyo. Lo que si sabe es si tiene clientes que
# le pagan. Este archivo traduce una cosa en la otra.
#
# De donde salen los numeros de cada respuesta
# --------------------------------------------
# No estan inventados. Cada pregunta de cuatro opciones ubica a la startup en un
# cuarto de la poblacion del dataset, y el numero que se le asigna es el valor
# tipico de ese cuarto.
#
# Por ejemplo con product_market_fit_score: en los datos, el cuarto mas bajo
# ronda 1.9 y el mas alto 7.3. Entonces "todavia no tengo clientes" vale 1.9 y
# "crecen cada mes" vale 7.3. Responder la pregunta es ubicarse en la
# distribucion real.
#
# Ojo con esto: la traduccion es una regla de negocio, no algo que el modelo
# haya aprendido. Es una suposicion documentada, y como tal se declara en el
# informe.
# ==========================================

from ml import config


# Las preguntas de opcion multiple. Cada una alimenta una sola variable.
PREGUNTAS = [
    # ---------- Producto y clientes ----------
    {
        "id": "paying_customers",
        "section": "Your product and customers",
        "question": "Do you have paying customers yet?",
        "variable": "product_market_fit_score",
        "options": [
            {"label": "Not yet, still building", "value": 1.9},
            {"label": "A few, once in a while", "value": 3.9},
            {"label": "Yes, steady and they come back", "value": 5.3},
            {"label": "Yes, and growing every month", "value": 7.3},
        ],
    },
    {
        "id": "talked_to_customers",
        "section": "Your product and customers",
        "question": "Before building it, did you talk to people to confirm they wanted it?",
        "variable": "did_customer_validation",
        "options": [
            {"label": "No, we built it first", "value": 0},
            {"label": "Yes, we asked before building", "value": 1},
        ],
    },
    {
        "id": "margin_per_sale",
        "section": "Your product and customers",
        "question": "After paying everything it costs to serve one customer, do you make money?",
        "variable": "unit_economics_score",
        "options": [
            {"label": "No, we lose money on each one", "value": 2.6},
            {"label": "We roughly break even", "value": 4.4},
            {"label": "Yes, a small margin", "value": 5.9},
            {"label": "Yes, a healthy margin", "value": 7.8},
        ],
    },
    {
        "id": "how_customers_find_you",
        "section": "Your product and customers",
        "question": "How do customers reach you?",
        "variable": "marketing_effectiveness",
        "options": [
            {"label": "We chase them one by one", "value": 2.5},
            {"label": "Some referrals, some outreach", "value": 4.3},
            {"label": "Ads and word of mouth bring them", "value": 5.7},
            {"label": "They mostly come to us on their own", "value": 7.6},
        ],
    },
    {
        "id": "scaled_early",
        "section": "Your product and customers",
        "question": "Did you hire or spend heavily before having steady customers?",
        "variable": "premature_scaling",
        "options": [
            {"label": "No, we kept it lean", "value": 0},
            {"label": "Yes, we grew before the demand was there", "value": 1},
        ],
    },

    # ---------- Equipo ----------
    {
        "id": "founder_disagreements",
        "section": "Your team",
        "question": "How often do serious disagreements come up between the founders?",
        "variable": "cofounder_conflict",
        "options": [
            {"label": "Almost never", "value": 1.4},
            {"label": "Rarely", "value": 3.3},
            {"label": "Every now and then", "value": 4.8},
            {"label": "Often", "value": 6.7},
        ],
    },

    # ---------- Sobre el fundador ----------
    {
        "id": "founded_before",
        "section": "About you",
        "question": "Have you founded a company before?",
        "variable": "founder_prior_exits",
        "options": [
            {"label": "This is my first one", "value": 0},
            {"label": "Yes, but it did not take off", "value": 1},
            {"label": "Yes, and I sold it or it succeeded", "value": 2},
        ],
    },

    # ---------- Mercado ----------
    {
        "id": "how_many_customers",
        "section": "Your market",
        "question": "How many people or companies could buy what you sell?",
        "variable": "market_size_score",
        "options": [
            {"label": "A local niche", "value": 2.5},
            {"label": "A region or a city", "value": 4.3},
            {"label": "The whole country", "value": 5.7},
            {"label": "Several countries", "value": 7.5},
        ],
    },
    {
        "id": "how_many_competitors",
        "section": "Your market",
        "question": "How many companies already offer something similar?",
        "variable": "competition_intensity",
        "options": [
            {"label": "Almost none", "value": 3.1},
            {"label": "A handful", "value": 4.9},
            {"label": "Quite a few", "value": 6.2},
            {"label": "The market is crowded", "value": 7.9},
        ],
    },
    {
        "id": "fundraising_climate",
        "section": "Your market",
        "question": "Right now, how hard is it to find investment in your sector?",
        "variable": "macro_climate",
        "options": [
            {"label": "Very hard, nobody is investing", "value": 2.7},
            {"label": "Hard", "value": 4.4},
            {"label": "Normal", "value": 5.6},
            {"label": "Easy, there is money around", "value": 7.3},
        ],
    },
]


# Las cuatro areas que se le preguntan al equipo
AREAS_DEL_EQUIPO = ["Product or engineering", "Sales", "Operations", "Finance"]

# Cuanto vale team_completeness segun cuantas areas cubra el equipo.
# La posicion 0 es para cuando no cubre ninguna, la 4 para cuando las cubre todas.
PUNTAJE_POR_AREAS_CUBIERTAS = [1.0, 2.8, 4.6, 6.0, 7.9]

# Tope de meses de caja. El dataset no tiene startups con mas de 50 meses, asi
# que no tiene sentido pasarle al modelo un numero mas grande que ese.
MAXIMO_MESES_DE_CAJA = 50.0


def secciones():
    """Agrupa las preguntas por seccion, respetando el orden en que estan.

    La pagina web usa esto para dibujar un recuadro por cada seccion.
    """
    agrupadas = {}

    for pregunta in PREGUNTAS:
        seccion = pregunta["section"]
        if seccion not in agrupadas:
            agrupadas[seccion] = []
        agrupadas[seccion].append(pregunta)

    return agrupadas


def calcular_meses_de_caja(dinero_disponible, gasto_mensual):
    """Cuantos meses puede aguantar la startup con la plata que tiene.

    No se pregunta directamente porque nadie sabe de memoria su runway, pero
    todos saben cuanto dinero tienen y cuanto gastan al mes.
    """
    if gasto_mensual <= 0:
        # Si no gasta nada, nunca se queda sin caja. Ademas hay que evitar la
        # division entre cero, que reventaria el programa.
        return MAXIMO_MESES_DE_CAJA

    meses = dinero_disponible / gasto_mensual

    if meses > MAXIMO_MESES_DE_CAJA:
        return MAXIMO_MESES_DE_CAJA

    return round(meses, 1)


def traducir(respuestas):
    """Convierte las respuestas del cuestionario en las 17 variables del modelo.

    "respuestas" es un diccionario con el id de cada pregunta y el numero de la
    opcion que el usuario eligio, empezando desde cero, mas los datos que se
    piden directamente como numero.
    """
    variables = {}

    # ---------- 1. Las preguntas de opcion multiple ----------
    # De cada una se toma la opcion elegida y se guarda su valor.
    for pregunta in PREGUNTAS:
        opcion_elegida = respuestas[pregunta["id"]]
        valor = pregunta["options"][opcion_elegida]["value"]
        variables[pregunta["variable"]] = valor

    # ---------- 2. El equipo ----------
    # Mientras mas areas cubra, mas completo esta.
    areas_cubiertas = len(respuestas["team_skills"])
    variables["team_completeness"] = PUNTAJE_POR_AREAS_CUBIERTAS[areas_cubiertas]

    # ---------- 3. La caja ----------
    variables["runway_months"] = calcular_meses_de_caja(
        respuestas["cash_available_usd"],
        respuestas["monthly_spend_usd"],
    )
    variables["monthly_burn_rate"] = float(respuestas["monthly_spend_usd"])

    # ---------- 4. Lo que se pregunta tal cual ----------
    variables["total_raised_usd"] = float(respuestas["total_raised_usd"])
    variables["domain_experience_years"] = float(respuestas["years_in_industry"])
    variables["funding_path"] = respuestas["funding_path"]
    variables["industry"] = respuestas["industry"]

    # ---------- 5. Revision final ----------
    # Si faltara alguna variable, el modelo fallaria de una forma rara y dificil
    # de rastrear. Es mejor darse cuenta aca mismo.
    for columna in config.CARACTERISTICAS:
        if columna not in variables:
            raise ValueError(f"La traduccion no genero la variable {columna}")

    return variables
