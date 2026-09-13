# ==========================================
# LA API
# Levantar:  python -m uvicorn app.main:app --reload --port 8000
#
# Tiene tres tipos de rutas:
#   - las paginas web que ve la gente        /  y  /expert
#   - las que calculan la prediccion         /assess  y  /predict
#   - las que sirven para monitorear         /health, /model-info, /feature-groups
#
# Todo lo que se ve en pantalla va en ingles, que es lo que pide la rubrica.
# Los comentarios van en espanol.
# ==========================================

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.questionnaire import AREAS_DEL_EQUIPO, secciones, traducir
from app.schemas import AssessmentRequest, AssessmentResponse, HealthResponse
from app.schemas import PredictionResponse, StartupRequest
from ml import config

VERSION_API = "1.0.0"

CARPETA_PLANTILLAS = Path(__file__).parent / "templates"
PLANTILLAS = Jinja2Templates(directory=str(CARPETA_PLANTILLAS))

app = FastAPI(
    title="Startup Survival Prediction API",
    description="Estimates the probability that a startup survives, from the "
                "characteristics it has at founding time.",
    version=VERSION_API,
)


# ==========================================
# EL MODELO
# ==========================================

# Aca se guarda el modelo una vez cargado. Empieza vacio y se llena la primera
# vez que alguien lo pide. Se hace asi para no leer el archivo del disco en
# cada peticion, que seria lentisimo.
ARTEFACTO = None


def cargar_artefacto():
    """Devuelve el modelo. La primera vez lo lee del disco, despues lo reutiliza.

    Si el archivo no existe devuelve None y la API sigue funcionando, pero en
    estado degradado. Es a proposito: es mejor un servicio que responde "no
    tengo modelo" a uno que ni siquiera arranca, porque asi el monitoreo puede
    darse cuenta del problema.
    """
    global ARTEFACTO

    if ARTEFACTO is None:
        if config.RUTA_MODELO.exists():
            ARTEFACTO = joblib.load(config.RUTA_MODELO)
        else:
            print(f"AVISO: no hay modelo en {config.RUTA_MODELO}")
            print("Corre primero: python -m ml.train")

    return ARTEFACTO


def calcular(variables):
    """Le pasa las 17 variables al modelo y devuelve la respuesta ya armada.

    Lo usan las dos rutas de prediccion, /predict y /assess, para que las dos
    hagan exactamente lo mismo y no puedan dar resultados distintos.
    """
    artefacto = cargar_artefacto()
    if artefacto is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # El modelo espera una tabla, aunque sea de una sola fila, y con las
    # columnas en el mismo orden en que fue entrenado.
    fila = pd.DataFrame([variables])[artefacto["features"]]

    # predict_proba devuelve dos numeros: la probabilidad de sobrevivir y la de
    # fracasar. La columna 1 es la de fracasar, que es lo que el modelo predice.
    probabilidad_de_fracaso = float(artefacto["pipeline"].predict_proba(fila)[0][1])
    umbral = artefacto["threshold"]

    return {
        "survival_probability": round(1 - probabilidad_de_fracaso, 4),
        "failure_probability": round(probabilidad_de_fracaso, 4),
        "at_risk": probabilidad_de_fracaso >= umbral,
        "risk_level": nivel_de_riesgo(probabilidad_de_fracaso, umbral),
        "threshold": umbral,
        "model_version": artefacto["model_version"],
    }


def nivel_de_riesgo(probabilidad_de_fracaso, umbral):
    """Convierte la probabilidad en una etiqueta que se entienda de un vistazo."""
    if probabilidad_de_fracaso < umbral * 0.8:
        return "LOW"
    if probabilidad_de_fracaso < umbral * 1.1:
        return "MEDIUM"
    return "HIGH"


# ==========================================
# RUTAS DE MONITOREO
# ==========================================

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Dice si el servicio esta vivo y si tiene el modelo cargado.

    La consultan el healthcheck de Docker cada 30 segundos y el pipeline de
    despliegue para saber si el contenedor nuevo arranco bien.
    """
    artefacto = cargar_artefacto()

    if artefacto is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version=None,
            api_version=VERSION_API,
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=artefacto["model_version"],
        api_version=VERSION_API,
    )


@app.get("/model-info", tags=["monitoring"])
def model_info():
    """La ficha del modelo que esta en produccion ahora mismo."""
    artefacto = cargar_artefacto()
    if artefacto is None:
        raise HTTPException(status_code=503, detail="Model not available")

    return {
        "model_name": artefacto["model_name"],
        "model_version": artefacto["model_version"],
        "trained_at": artefacto["trained_at"],
        "training_rows": artefacto["rows"],
        "failure_rate": artefacto["failure_rate"],
        "threshold": artefacto["threshold"],
        "hyperparameters": artefacto["hyperparameters"],
        "metrics": artefacto["metrics"],
    }


@app.get("/feature-groups", tags=["monitoring"])
def feature_groups():
    """Cuanto aporta cada grupo de variables.

    Es el resultado central del analisis y responde la pregunta del proyecto:
    que caracteristicas permiten anticipar la supervivencia.
    """
    artefacto = cargar_artefacto()
    if artefacto is None:
        raise HTTPException(status_code=503, detail="Model not available")

    return artefacto["group_study"]


# ==========================================
# RUTAS DE PREDICCION
# ==========================================

@app.post("/assess", response_model=AssessmentResponse, tags=["assessment"])
def assess(respuestas: AssessmentRequest):
    """Evalua una startup a partir de respuestas en lenguaje llano.

    Es la puerta de entrada para emprendedores. Primero traduce las respuestas
    a las 17 variables y despues usa el mismo calculo que /predict.
    """
    variables = traducir(respuestas.model_dump())
    resultado = calcular(variables)

    # Se devuelven tambien las variables para que el usuario pueda ver con que
    # numeros se calculo su resultado. Sin esto seria una caja negra.
    resultado["derived_features"] = variables

    return AssessmentResponse(**resultado)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(startup: StartupRequest):
    """Evalua una startup a partir de los 17 valores numericos directos.

    Es para analistas que ya tienen los puntajes evaluados y no necesitan el
    cuestionario.
    """
    variables = startup.model_dump()
    resultado = calcular(variables)

    return PredictionResponse(**resultado)


@app.get("/questionnaire", tags=["assessment"])
def questionnaire():
    """Las preguntas del cuestionario, para que la pagina se arme sola."""
    return {"sections": secciones(), "team_areas": AREAS_DEL_EQUIPO}


@app.get("/form-options", tags=["prediction"])
def form_options():
    """Los valores validos y los rangos de cada campo."""
    artefacto = cargar_artefacto()
    if artefacto is None:
        raise HTTPException(status_code=503, detail="Model not available")

    return {
        "categorical": artefacto["categorical_options"],
        "numeric": artefacto["numeric_ranges"],
    }


# ==========================================
# PAGINAS WEB
# ==========================================

@app.get("/", response_class=HTMLResponse, tags=["web"])
def home(request: Request):
    """La pagina principal: el cuestionario para emprendedores."""
    contexto = {
        "model": cargar_artefacto(),
        "secciones": secciones(),
        "areas_equipo": AREAS_DEL_EQUIPO,
    }
    return PLANTILLAS.TemplateResponse(request=request, name="assess.html", context=contexto)


@app.get("/expert", response_class=HTMLResponse, tags=["web"])
def expert(request: Request):
    """La vista tecnica, con los 17 valores numericos directos."""
    contexto = {"model": cargar_artefacto()}
    return PLANTILLAS.TemplateResponse(request=request, name="index.html", context=contexto)
