# ==========================================
# DONDE SE GUARDAN LAS CONSULTAS
#
# Cada vez que alguien usa la aplicacion se guarda lo que respondio y lo que el
# modelo le contesto. Meses despues, cuando se sepa si ese emprendimiento
# sobrevivio o no, alguien de la incubadora marca el desenlace desde la vista
# de administracion.
#
# Por que hace falta esto
# -----------------------
# Una respuesta al cuestionario da las 17 variables, pero no dice como termino
# la historia. Y sin eso no se puede entrenar: un modelo necesita saber la
# respuesta correcta para aprender. El desenlace solo se conoce uno o dos anios
# despues, asi que hay que guardarlo aparte y esperar.
#
# Mientras tanto, las consultas sin desenlace igual sirven: con ellas se mide la
# deriva de datos, que no necesita saber como termino cada caso, solo compara
# como son los emprendimientos que llegan hoy contra los del entrenamiento.
#
# Se usa SQLite porque es un archivo y nada mas. No hay que instalar ni
# mantener un servidor de base de datos para guardar unas miles de filas.
# ==========================================

import json
import sqlite3
from datetime import datetime, timezone

from ml import config


def conectar():
    """Abre la base y se asegura de que la tabla exista."""
    config.RUTA_CONSULTAS.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(config.RUTA_CONSULTAS)

    # Las 17 variables se guardan juntas como texto JSON en vez de una columna
    # por variable. Asi, si manana el modelo usa otras variables, la tabla no
    # hay que cambiarla.
    conexion.execute("""
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            origen TEXT NOT NULL,
            variables TEXT NOT NULL,
            probabilidad_de_fracaso REAL NOT NULL,
            en_riesgo INTEGER NOT NULL,
            version_del_modelo TEXT NOT NULL,
            desenlace INTEGER,
            fecha_del_desenlace TEXT
        )
    """)
    conexion.commit()
    return conexion


def guardar_consulta(variables, resultado, origen):
    """Guarda una consulta y devuelve su numero.

    "origen" dice por donde entro: el cuestionario o la vista tecnica. Sirve
    para saber despues de donde vienen los datos.
    """
    conexion = conectar()
    cursor = conexion.execute(
        """INSERT INTO consultas
           (fecha, origen, variables, probabilidad_de_fracaso, en_riesgo, version_del_modelo)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            origen,
            json.dumps(variables),
            resultado["failure_probability"],
            int(resultado["at_risk"]),
            resultado["model_version"],
        ),
    )
    numero = cursor.lastrowid
    conexion.commit()
    conexion.close()
    return numero


def marcar_desenlace(numero, fracaso):
    """Anota como termino un emprendimiento.

    "fracaso" es True si cerro y False si sigue operando. Es el dato que
    convierte una consulta guardada en una fila que sirve para entrenar.

    Devuelve False si esa consulta no existe.
    """
    conexion = conectar()
    cursor = conexion.execute(
        "UPDATE consultas SET desenlace = ?, fecha_del_desenlace = ? WHERE id = ?",
        (int(fracaso), datetime.now(timezone.utc).isoformat(timespec="seconds"), numero),
    )
    conexion.commit()
    encontrada = cursor.rowcount > 0
    conexion.close()
    return encontrada


def listar(solo_sin_desenlace=False, limite=200):
    """Devuelve las consultas guardadas, de la mas nueva a la mas vieja."""
    conexion = conectar()
    conexion.row_factory = sqlite3.Row

    consulta = "SELECT * FROM consultas"
    if solo_sin_desenlace:
        consulta += " WHERE desenlace IS NULL"
    consulta += " ORDER BY id DESC LIMIT ?"

    filas = conexion.execute(consulta, (limite,)).fetchall()
    conexion.close()

    salida = []
    for fila in filas:
        registro = dict(fila)
        registro["variables"] = json.loads(registro["variables"])
        salida.append(registro)
    return salida


def contar():
    """Cuantas consultas hay en total, cuantas etiquetadas y cuantas no."""
    conexion = conectar()
    total = conexion.execute("SELECT COUNT(*) FROM consultas").fetchone()[0]
    etiquetadas = conexion.execute(
        "SELECT COUNT(*) FROM consultas WHERE desenlace IS NOT NULL").fetchone()[0]
    conexion.close()
    return {"total": total, "con_desenlace": etiquetadas, "sin_desenlace": total - etiquetadas}


def filas_para_entrenar():
    """Las consultas que ya tienen desenlace, listas para sumarse al dataset.

    Devuelve una lista de diccionarios con las 17 variables mas la columna
    objetivo, con el mismo nombre que usa el dataset original.
    """
    conexion = conectar()
    filas = conexion.execute(
        "SELECT variables, desenlace FROM consultas WHERE desenlace IS NOT NULL").fetchall()
    conexion.close()

    salida = []
    for variables_json, desenlace in filas:
        fila = json.loads(variables_json)
        fila[config.OBJETIVO] = int(desenlace)
        salida.append(fila)
    return salida
