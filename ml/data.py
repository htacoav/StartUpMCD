"""
Carga y preparacion de los datos.

Son dos funciones nada mas: una lee el archivo y la otra separa las variables
predictoras de la que se quiere predecir.
"""

import pandas as pd

from ml import config


def cargar_datos():
    """Lee el CSV y devuelve la tabla completa, con sus 30 columnas."""
    if not config.RUTA_DATOS.exists():
        raise SystemExit(
            f"No encuentro {config.RUTA_DATOS}.\n"
            "Descarga el dataset y dejalo en la carpeta data/."
        )
    return pd.read_csv(config.RUTA_DATOS)


def preparar(tabla):
    """Separa las variables predictoras (X) de lo que se quiere predecir (y).

    De las 30 columnas del archivo solo se usan 17. Las otras 13 se descartan
    por dos motivos distintos, explicados en config.py: seis son columnas con
    fuga de informacion y siete no aportan nada al modelo.
    """
    y = tabla[config.OBJETIVO]
    X = tabla[config.CARACTERISTICAS].copy()

    # Las categoricas se pasan a texto por si vinieran vacias. Un formulario web
    # tambien puede llegar incompleto, asi que conviene tener un valor de relleno.
    for columna in config.CATEGORICAS:
        X[columna] = X[columna].fillna("unknown").astype(str)

    # Las numericas que vengan mal se convierten a numero, y si no se puede, a cero.
    for columna in config.NUMERICAS:
        X[columna] = pd.to_numeric(X[columna], errors="coerce").fillna(0)

    return X, y


def opciones_de_formulario(X):
    """Valores posibles de las dos variables categoricas.

    Sirven para armar los desplegables de la pagina web sin escribirlos a mano.
    """
    opciones = {}
    for columna in config.CATEGORICAS:
        opciones[columna] = sorted(X[columna].unique().tolist())
    return opciones


def rangos_numericos(X):
    """Minimo, maximo y mediana de cada variable numerica.

    La pagina web los usa para poner limites a los campos y para rellenarlos
    con un valor razonable por defecto.
    """
    rangos = {}
    for columna in config.NUMERICAS:
        rangos[columna] = {
            "min": float(X[columna].min()),
            "max": float(X[columna].max()),
            "median": float(X[columna].median()),
        }
    return rangos
