# ==========================================
# TEXTOS DEL CUESTIONARIO EN ESPANOL
#
# Solo cambia lo que se ve en pantalla. Los ids de las preguntas y los valores
# que se mandan a la API son los mismos que en la version en ingles, asi que
# las dos paginas llaman a /assess exactamente igual y dan el mismo resultado.
#
# La API sigue en ingles, como pide la rubrica. Esta es una pagina adicional.
# ==========================================

# Nombre de cada seccion, con la seccion en ingles como clave
SECCIONES = {
    "Your product and customers": "Tu producto y tus clientes",
    "Your team": "Tu equipo",
    "About you": "Sobre ti",
    "Your market": "Tu mercado",
}

# Cada pregunta con sus opciones, en el mismo orden que en questionnaire.py.
# El orden importa: la opcion 0 de aca tiene que ser la opcion 0 de alla.
PREGUNTAS = {
    "paying_customers": {
        "question": "¿Ya tienes clientes que te pagan?",
        "options": ["Todavía no, sigo construyendo",
                    "Algunos, de vez en cuando",
                    "Sí, constantes y vuelven a comprar",
                    "Sí, y crecen cada mes"],
    },
    "talked_to_customers": {
        "question": "Antes de construirlo, ¿hablaste con gente para confirmar que lo querían?",
        "options": ["No, primero lo construimos",
                    "Sí, preguntamos antes de construir"],
    },
    "margin_per_sale": {
        "question": "Después de pagar todo lo que cuesta atender a un cliente, ¿te queda ganancia?",
        "options": ["No, perdemos plata con cada uno",
                    "Más o menos quedamos en cero",
                    "Sí, un margen pequeño",
                    "Sí, un margen saludable"],
    },
    "how_customers_find_you": {
        "question": "¿Cómo llegan los clientes a ti?",
        "options": ["Los buscamos uno por uno",
                    "Algunos por recomendación, otros los buscamos",
                    "Nos llegan por publicidad y boca a boca",
                    "La mayoría llega solo"],
    },
    "scaled_early": {
        "question": "¿Contrataste o gastaste fuerte antes de tener clientes constantes?",
        "options": ["No, nos mantuvimos austeros",
                    "Sí, crecimos antes de que hubiera demanda"],
    },
    "founder_disagreements": {
        "question": "¿Con qué frecuencia hay desacuerdos serios entre los socios?",
        "options": ["Casi nunca", "Rara vez", "De vez en cuando", "Seguido"],
    },
    "founded_before": {
        "question": "¿Has fundado una empresa antes?",
        "options": ["Esta es mi primera",
                    "Sí, pero no despegó",
                    "Sí, y la vendí o le fue bien"],
    },
    "how_many_customers": {
        "question": "¿Cuántas personas o empresas podrían comprar lo que vendes?",
        "options": ["Un nicho local",
                    "Una región o una ciudad",
                    "Todo el país",
                    "Varios países"],
    },
    "how_many_competitors": {
        "question": "¿Cuántas empresas ya ofrecen algo parecido?",
        "options": ["Casi ninguna", "Unas pocas", "Bastantes", "El mercado está saturado"],
    },
    "fundraising_climate": {
        "question": "Hoy, ¿qué tan difícil es conseguir inversión en tu sector?",
        "options": ["Muy difícil, nadie está invirtiendo",
                    "Difícil",
                    "Normal",
                    "Fácil, hay dinero dando vueltas"],
    },
}

# Las areas del equipo. La clave es la que se manda a la API, en ingles.
AREAS_DEL_EQUIPO = {
    "Product or engineering": "Producto o ingeniería",
    "Sales": "Ventas",
    "Operations": "Operaciones",
    "Finance": "Finanzas",
}

SECTORES = {
    "agriculture": "Agricultura",
    "biotech": "Biotecnología",
    "consumer_app": "App de consumo",
    "ecommerce": "Comercio electrónico",
    "fintech": "Fintech",
    "food_bev": "Alimentos y bebidas",
    "hardware": "Hardware",
    "healthtech": "Salud digital",
    "marketplace": "Marketplace",
    "services": "Servicios",
    "tech_saas": "Software (SaaS)",
}

FINANCIAMIENTO = {
    "angel": "Inversionista ángel",
    "bootstrapped": "Recursos propios",
    "vc_seed": "Capital de riesgo, etapa semilla",
    "vc_series_a_plus": "Capital de riesgo, serie A o más",
}

# Nombre de cada variable del modelo, para la explicacion del resultado
VARIABLES = {
    "macro_climate": "Clima para conseguir inversión",
    "market_size_score": "Tamaño del mercado",
    "competition_intensity": "Competencia",
    "founder_prior_exits": "Experiencia fundando empresas",
    "domain_experience_years": "Años en el rubro",
    "cofounder_conflict": "Desacuerdos entre socios",
    "team_completeness": "Áreas cubiertas por el equipo",
    "product_market_fit_score": "Clientes que pagan",
    "did_customer_validation": "Validación con clientes",
    "premature_scaling": "Crecer antes de tiempo",
    "unit_economics_score": "Margen por venta",
    "marketing_effectiveness": "Cómo llegan los clientes",
    "funding_path": "Forma de financiamiento",
    "total_raised_usd": "Dinero levantado",
    "monthly_burn_rate": "Gasto mensual",
    "runway_months": "Meses de caja",
    "industry": "Sector",
    "departamento": "Departamento",
}

# Las etiquetas de riesgo que devuelve la API
RIESGO = {"LOW": "RIESGO BAJO", "MEDIUM": "RIESGO MEDIO", "HIGH": "RIESGO ALTO"}
