"""
Configuracion del proyecto. Todo lo que se puede querer cambiar esta aca,
para no andar buscandolo por el codigo.
"""

import os
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "startup_survival_master.csv"
RUTA_MODELO = RAIZ / "models" / "model.joblib"
RUTA_METRICAS = RAIZ / "models" / "metrics.json"

# Base de datos donde la aplicacion guarda cada consulta que recibe. Vive en
# una carpeta aparte porque en Docker se monta como volumen, para que sobreviva
# cuando se reemplaza el contenedor.
RUTA_CONSULTAS = Path(os.environ.get("RUTA_CONSULTAS", RAIZ / "consultas" / "consultas.db"))

# Token que protege la vista de administracion, donde se marcan los desenlaces.
# Si no se define, la vista queda abierta, lo que sirve para desarrollar pero
# nunca para produccion.
TOKEN_ADMIN = os.environ.get("TOKEN_ADMIN", "")

SEMILLA = 42        # para que los resultados salgan iguales cada vez


# --- Que se predice ------------------------------------------------------
# El dataset trae la columna 'failed': 1 si la startup fracaso, 0 si no.
# Se predice el fracaso y la aplicacion muestra lo contrario, o sea la
# probabilidad de sobrevivir, que es 1 menos la probabilidad de fracaso.
OBJETIVO = "failed"

# Estas columnas NO se pueden usar: solo se conocen despues de que la startup
# ya fracaso o sobrevivio. Usarlas seria hacer trampa, el modelo estaria
# leyendo la respuesta en vez de predecirla.
COLUMNAS_CON_FUGA = [
    "startup_id",        # identificador, no aporta nada
    "outcome",           # el desenlace en texto: failed, still_operating, acquired, ipo
    "failure_reason",    # por que fracaso, obviamente solo se sabe si fracaso
    "years_to_outcome",  # cuantos anios tardo en llegar al desenlace
    "survived_5y",       # si seguia viva a los 5 anios
    "failed",            # la variable objetivo
]

# Estas se midieron y no aportan nada al modelo, asi que se descartan para
# que el formulario de la web no pida datos inutiles.
COLUMNAS_SIN_APORTE = [
    "pivots",
    "technical_cofounder",
    "prior_failures",
    "weekly_hours",
    "founding_year",
    "n_cofounders",
    "age_of_founder",
]


# --- Las variables agrupadas ---------------------------------------------
# El diccionario del dataset agrupa las variables por naturaleza. Esa
# agrupacion es el corazon del analisis: permite medir cuanto pesa la suerte
# frente a la ejecucion, y ademas organiza el formulario de la web en secciones.
GRUPOS = {
    "circumstance": ["macro_climate", "market_size_score", "competition_intensity"],
    "founder":      ["founder_prior_exits", "domain_experience_years"],
    "team":         ["cofounder_conflict", "team_completeness"],
    "execution":    ["product_market_fit_score", "did_customer_validation",
                     "premature_scaling", "unit_economics_score", "marketing_effectiveness"],
    "funding":      ["funding_path", "total_raised_usd", "monthly_burn_rate", "runway_months"],
    "context":      ["industry", "departamento"],
}

# Nombre legible de cada grupo, para mostrarlo en la web y en el informe.
NOMBRES_DE_GRUPO = {
    "circumstance": "Circumstance and luck",
    "founder":      "Founder background",
    "team":         "Founding team",
    "execution":    "Execution",
    "funding":      "Funding and cash",
    "context":      "Context",
}

# La lista completa de variables sale de aplanar los grupos.
CARACTERISTICAS = []
for columnas_del_grupo in GRUPOS.values():
    CARACTERISTICAS.extend(columnas_del_grupo)

CATEGORICAS = ["industry", "funding_path", "departamento"]

# Los departamentos del Peru, mas Callao. Se usan para armar el formulario.
#
# Ojo con algo: el dataset original no trae esta columna, asi que las 48 000
# filas historicas quedan como "unknown". Solo las consultas que recibe la
# aplicacion traen el departamento de verdad. El modelo recien podra aprender
# algo de esta variable cuando se acumulen suficientes consultas reales con su
# desenlace conocido.
DEPARTAMENTOS = [
    "Amazonas", "Ancash", "Apurimac", "Arequipa", "Ayacucho", "Cajamarca",
    "Callao", "Cusco", "Huancavelica", "Huanuco", "Ica", "Junin",
    "La Libertad", "Lambayeque", "Lima", "Loreto", "Madre de Dios", "Moquegua",
    "Pasco", "Piura", "Puno", "San Martin", "Tacna", "Tumbes", "Ucayali",
]
NUMERICAS = [c for c in CARACTERISTICAS if c not in CATEGORICAS]
