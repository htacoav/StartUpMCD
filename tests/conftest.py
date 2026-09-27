"""Piezas compartidas por todas las pruebas."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml import config


@pytest.fixture(autouse=True, scope="session")
def base_de_pruebas(tmp_path_factory):
    """Manda la base de consultas a una carpeta temporal.

    Sin esto, cada corrida de las pruebas escribiria consultas de mentira en la
    base de verdad, y despues esas filas terminarian dentro del entrenamiento.
    """
    config.RUTA_CONSULTAS = tmp_path_factory.mktemp("consultas") / "consultas.db"
    return config.RUTA_CONSULTAS


@pytest.fixture(scope="session")
def cliente():
    """Un cliente HTTP contra la aplicacion, sin levantar un servidor de verdad."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def startup():
    """Una startup promedio, con los valores medianos del dataset.

    Cada prueba parte de esta y cambia solo lo que quiere examinar, asi se sabe
    que la diferencia en el resultado viene de ese cambio y no de otra cosa.
    """
    return {
        "macro_climate": 5.0,
        "market_size_score": 6.0,
        "competition_intensity": 5.5,
        "founder_prior_exits": 0,
        "domain_experience_years": 4.4,
        "cofounder_conflict": 4.0,
        "team_completeness": 5.3,
        "product_market_fit_score": 4.6,
        "did_customer_validation": 1,
        "premature_scaling": 0,
        "unit_economics_score": 5.2,
        "marketing_effectiveness": 5.0,
        "funding_path": "angel",
        "total_raised_usd": 111000.0,
        "monthly_burn_rate": 6300.0,
        "runway_months": 18.8,
        "industry": "tech_saas",
        "departamento": "Lima",
    }
