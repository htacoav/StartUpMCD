"""
Contratos de entrada y salida de la API.

Estan en ingles porque son la cara publica del servicio: lo que ve quien la
consume y lo que aparece en la documentacion automatica de /docs.

Pydantic valida cada campo antes de que llegue al modelo. Si alguien manda un
puntaje de 50 donde la escala llega a 10, la API responde un error claro en vez
de darle basura al modelo y devolver una prediccion sin sentido.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StartupRequest(BaseModel):
    """Los 17 datos de una startup al momento de fundarse."""

    # --- Circumstance and luck: el entorno que le toco ---
    macro_climate: float = Field(..., ge=0, le=10, examples=[5.0],
                                 description="Funding and market climate when founded (0-10)")
    market_size_score: float = Field(..., ge=0, le=10, examples=[6.0],
                                     description="Addressable market size (0-10)")
    competition_intensity: float = Field(..., ge=0, le=10, examples=[5.5],
                                         description="How crowded the market is (0-10)")

    # --- Founder background: quien es el fundador ---
    founder_prior_exits: int = Field(..., ge=0, le=2, examples=[0],
                                     description="0 first-time, 1 prior venture, 2 prior successful exit")
    domain_experience_years: float = Field(..., ge=0, le=40, examples=[4.4],
                                           description="Years of experience in the domain")

    # --- Founding team: como esta el equipo ---
    cofounder_conflict: float = Field(..., ge=0, le=10, examples=[4.0],
                                      description="Level of conflict between co-founders (0-10)")
    team_completeness: float = Field(..., ge=0, le=10, examples=[5.3],
                                     description="How complete the founding team is (0-10)")

    # --- Execution: que esta haciendo ---
    product_market_fit_score: float = Field(..., ge=0, le=10, examples=[4.6],
                                            description="Product-market fit (0-10)")
    did_customer_validation: Literal[0, 1] = Field(..., examples=[1],
                                                   description="1 if demand was validated before building")
    premature_scaling: Literal[0, 1] = Field(..., examples=[0],
                                             description="1 if it scaled before reaching product-market fit")
    unit_economics_score: float = Field(..., ge=0, le=10, examples=[5.2],
                                        description="Health of the unit economics (0-10)")
    marketing_effectiveness: float = Field(..., ge=0, le=10, examples=[5.0],
                                           description="Marketing effectiveness (0-10)")

    # --- Funding and cash: de cuanto dinero dispone ---
    funding_path: str = Field(..., examples=["angel"],
                              description="bootstrapped, angel, vc_seed or vc_series_a_plus")
    total_raised_usd: float = Field(..., ge=0, examples=[111000.0],
                                    description="Total capital raised in USD")
    monthly_burn_rate: float = Field(..., ge=0, examples=[6300.0],
                                     description="Monthly cash burn in USD")
    runway_months: float = Field(..., ge=0, le=60, examples=[18.8],
                                 description="Months of runway at founding")

    # --- Context: en que sector opera ---
    industry: str = Field(..., examples=["tech_saas"], description="Industry sector")


class PredictionResponse(BaseModel):
    """Lo que devuelve la API al predecir.

    Se informan las dos caras de la misma moneda porque cada una se lee mejor
    en un contexto distinto: la de supervivencia es la que le interesa al
    emprendedor y la de fracaso es con la que trabaja el modelo.
    """

    survival_probability: float = Field(..., ge=0, le=1,
                                        description="Probability that the startup survives")
    failure_probability: float = Field(..., ge=0, le=1,
                                       description="Probability that it fails")
    at_risk: bool = Field(..., description="True if failure probability is above the threshold")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Human readable band")
    threshold: float = Field(..., description="Decision threshold used")
    model_version: str = Field(..., description="Version of the model that answered")


class HealthResponse(BaseModel):
    """Estado del servicio. Lo consultan el monitoreo y el despliegue."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
    api_version: str


class AssessmentRequest(BaseModel):
    """Las respuestas del cuestionario para usuarios finales.

    A diferencia de StartupRequest, que pide los 17 numeros que usa el modelo,
    aca se piden respuestas que un emprendedor si puede dar. La traduccion de
    unas a otros la hace app/questionnaire.py.

    Cada campo de opcion multiple es el numero de la opcion elegida, empezando
    en cero.
    """

    # Producto y clientes
    paying_customers: int = Field(..., ge=0, le=3, examples=[2],
                                  description="0 none, 1 a few, 2 steady, 3 growing")
    talked_to_customers: int = Field(..., ge=0, le=1, examples=[1],
                                     description="0 built first, 1 asked before building")
    margin_per_sale: int = Field(..., ge=0, le=3, examples=[2],
                                 description="0 loses money, 3 healthy margin")
    how_customers_find_you: int = Field(..., ge=0, le=3, examples=[1],
                                        description="0 we chase them, 3 they come to us")
    scaled_early: int = Field(..., ge=0, le=1, examples=[0],
                              description="0 stayed lean, 1 grew before demand")

    # Equipo
    founder_disagreements: int = Field(..., ge=0, le=3, examples=[1],
                                       description="0 almost never, 3 often")
    team_skills: List[str] = Field(default_factory=list,
                                   examples=[["Product or engineering", "Sales"]],
                                   description="Areas the founding team covers")

    # Sobre el fundador
    founded_before: int = Field(..., ge=0, le=2, examples=[0],
                                description="0 first company, 1 prior venture, 2 prior success")
    years_in_industry: float = Field(..., ge=0, le=40, examples=[4.0],
                                     description="Years working in this industry")

    # Mercado
    how_many_customers: int = Field(..., ge=0, le=3, examples=[2],
                                    description="0 local niche, 3 several countries")
    how_many_competitors: int = Field(..., ge=0, le=3, examples=[1],
                                      description="0 almost none, 3 crowded market")
    fundraising_climate: int = Field(..., ge=0, le=3, examples=[2],
                                     description="0 very hard, 3 easy")
    industry: str = Field(..., examples=["tech_saas"], description="Industry sector")

    # Dinero
    total_raised_usd: float = Field(..., ge=0, examples=[100000.0],
                                    description="Total money raised so far, in USD")
    cash_available_usd: float = Field(..., ge=0, examples=[50000.0],
                                      description="Cash available right now, in USD")
    monthly_spend_usd: float = Field(..., ge=0, examples=[5000.0],
                                     description="How much is spent per month, in USD")
    funding_path: str = Field(..., examples=["angel"],
                              description="bootstrapped, angel, vc_seed or vc_series_a_plus")


class AssessmentResponse(PredictionResponse):
    """La prediccion mas las variables que salieron de traducir las respuestas.

    Se devuelven tambien las variables para que el resultado sea auditable: quien
    use la API puede ver exactamente con que numeros se calculo su probabilidad.
    """

    derived_features: dict = Field(..., description="The 17 model inputs derived from the answers")
