"""Pruebas del mantenimiento y de la integracion continua.

Son los tres casos que se exigen para la Unidad II:

  Caso 1  la puerta de calidad no deja pasar un modelo peor que el actual
  Caso 2  la deriva de datos se detecta cuando los datos cambian de verdad
  Caso 3  el despliegue vuelve a la version anterior si la nueva no responde

Se agregan dos pruebas mas sobre la configuracion de los flujos, porque un
workflow mal encadenado desplegaria sin pasar por las pruebas y eso no se nota
hasta que ya paso.
"""

import json
import os
import stat
import subprocess

import pandas as pd
import pytest
import yaml

from ml import config
from ml.deriva import detectar_deriva
from ml.mantenimiento import MINIMOS, decidir, pasa_los_minimos

RAIZ = config.RAIZ


# ==========================================================================
# CASO 1: la puerta de calidad
# ==========================================================================

@pytest.fixture
def notas_de_produccion():
    """Las notas del modelo que esta en produccion ahora mismo."""
    return json.loads(config.RUTA_METRICAS.read_text(encoding="utf-8"))


@pytest.fixture
def candidato_bueno(notas_de_produccion):
    """Un modelo candidato que mejora al que esta en produccion."""
    notas = dict(notas_de_produccion["metrics"])
    notas["roc_auc"] = notas["roc_auc"] + 0.02
    return notas


def test_caso1_no_promueve_un_modelo_peor(candidato_bueno, notas_de_produccion):
    """Un candidato con peor AUC se queda afuera.

    Es la proteccion mas importante del pipeline. Sin ella, un reentrenamiento
    con datos malos degradaria el servicio de forma automatica y silenciosa.
    """
    # Se le baja el AUC lo suficiente para que sea peor, pero no tanto como
    # para que caiga bajo el minimo absoluto. Asi se comprueba la comparacion
    # contra produccion y no la otra puerta.
    peor = dict(candidato_bueno)
    peor["roc_auc"] = notas_de_produccion["metrics"]["roc_auc"] - 0.02

    decision = decidir(peor, notas_de_produccion)

    assert decision["promover"] is False
    assert "no mejora" in decision["motivo"]


def test_caso1_tampoco_promueve_una_mejora_insignificante(candidato_bueno, notas_de_produccion):
    """Una diferencia de milesimas es ruido, no una mejora.

    Si se promoviera por eso, el sistema estaria redesplegando en produccion
    todas las semanas sin ninguna ganancia real.
    """
    casi_igual = dict(candidato_bueno)
    casi_igual["roc_auc"] = notas_de_produccion["metrics"]["roc_auc"] + 0.0001

    assert decidir(casi_igual, notas_de_produccion)["promover"] is False


def test_caso1_si_promueve_una_mejora_real(candidato_bueno, notas_de_produccion):
    """El caso contrario: si de verdad mejora, tiene que pasar."""
    decision = decidir(candidato_bueno, notas_de_produccion)

    assert decision["promover"] is True
    assert "mejora el AUC" in decision["motivo"]


def test_caso1_rechaza_un_modelo_degenerado(candidato_bueno):
    """Un modelo que marca casi todo como fracaso no entra, aunque tenga buen AUC.

    Es la leccion del umbral de la Unidad I convertida en regla automatica: con
    clases desbalanceadas un modelo puede verse bien en las metricas habituales
    y no servir para nada.
    """
    degenerado = dict(candidato_bueno)
    degenerado["flagged_rate"] = 0.95

    motivos = pasa_los_minimos(degenerado)

    assert motivos != []
    assert "marca el" in motivos[0]


def test_caso1_decide_sobre_las_consultas_reales_cuando_las_hay(candidato_bueno, notas_de_produccion):
    """Cuando hay consultas reales suficientes, la comparacion se hace con ellas.

    Esto salio de la simulacion de dos anios: comparando sobre el dataset
    completo, el candidato daba 0.7431 contra 0.7469 y se rechazaba siempre,
    aunque sobre las consultas reales diera 0.6650 contra 0.6484. Midiendo
    sobre la poblacion equivocada, el sistema nunca habria aprendido nada de
    sus propios datos.
    """
    mejor_en_la_realidad = {"auc_produccion": 0.6484, "auc_candidato": 0.6650,
                            "filas_del_examen": 1890}
    decision = decidir(candidato_bueno, notas_de_produccion, mejor_en_la_realidad)
    assert decision["promover"] is True
    assert "consultas reales" in decision["motivo"]

    peor_en_la_realidad = {"auc_produccion": 0.6650, "auc_candidato": 0.6484,
                           "filas_del_examen": 1890}
    decision = decidir(candidato_bueno, notas_de_produccion, peor_en_la_realidad)
    assert decision["promover"] is False


def test_caso1_los_minimos_mandan_aunque_gane_en_la_realidad(candidato_bueno, notas_de_produccion):
    """Un modelo degenerado no entra ni aunque gane la comparacion real."""
    degenerado = dict(candidato_bueno)
    degenerado["flagged_rate"] = 0.95

    comparacion = {"auc_produccion": 0.60, "auc_candidato": 0.70, "filas_del_examen": 1890}
    assert decidir(degenerado, notas_de_produccion, comparacion)["promover"] is False


def test_caso1_rechaza_un_modelo_bajo_el_minimo_absoluto(candidato_bueno):
    """Aunque no hubiera modelo en produccion, hay un piso que respetar."""
    malo = dict(candidato_bueno)
    malo["roc_auc"] = MINIMOS["roc_auc"] - 0.05

    assert decidir(malo, None)["promover"] is False


# ==========================================================================
# CASO 2: la deriva de datos
# ==========================================================================

@pytest.fixture(scope="module")
def muestra():
    """Una muestra chica del dataset, para que las pruebas corran rapido."""
    return pd.read_csv(config.RUTA_DATOS).sample(3000, random_state=1)


def test_caso2_no_avisa_cuando_los_datos_no_cambiaron(muestra):
    """Si los datos nuevos se parecen a los viejos, no hay alerta.

    Sin esta prueba el detector podria estar avisando siempre, que es tan
    inutil como no avisar nunca.
    """
    mitad = len(muestra) // 2
    resultado = detectar_deriva(muestra.iloc[:mitad], muestra.iloc[mitad:])

    assert resultado["hay_deriva"] is False
    assert resultado["columnas_con_deriva"] <= 2


def test_caso2_avisa_cuando_los_datos_cambiaron(muestra):
    """Se alteran varias columnas a proposito y el detector tiene que verlo.

    Simula lo que pasaria en la realidad si cambiara el perfil de los
    emprendimientos que usan la aplicacion.
    """
    mitad = len(muestra) // 2
    referencia = muestra.iloc[:mitad]
    nuevos = muestra.iloc[mitad:].copy()

    # Startups mucho mas grandes, con peor encaje y menos caja
    nuevos["product_market_fit_score"] = nuevos["product_market_fit_score"] * 0.3
    nuevos["runway_months"] = nuevos["runway_months"] * 0.4
    nuevos["monthly_burn_rate"] = nuevos["monthly_burn_rate"] * 12
    nuevos["total_raised_usd"] = nuevos["total_raised_usd"] * 15
    nuevos["cofounder_conflict"] = 9.5
    nuevos["team_completeness"] = 1.0

    resultado = detectar_deriva(referencia, nuevos)

    assert resultado["hay_deriva"] is True
    assert resultado["columnas_con_deriva"] >= 6

    derivadas = [c["columna"] for c in resultado["detalle"] if c["derivo"]]
    assert "product_market_fit_score" in derivadas
    assert "monthly_burn_rate" in derivadas


def test_caso2_no_mide_deriva_con_pocas_consultas():
    """Con pocas filas reales no se mide deriva, se avisa y se sigue.

    Esto salio de una prueba real: con 4 consultas guardadas, Evidently reporto
    deriva en las 17 columnas, que era falso. Medir deriva con un punado de
    filas no da un resultado malo, da uno inventado, y el pipeline abriria un
    issue de alarma todas las semanas durante los primeros meses.
    """
    from ml.mantenimiento import MINIMO_PARA_MEDIR_DERIVA, hay_suficientes

    assert hay_suficientes(4, "produccion") is False
    assert hay_suficientes(MINIMO_PARA_MEDIR_DERIVA - 1, "produccion") is False
    assert hay_suficientes(MINIMO_PARA_MEDIR_DERIVA, "produccion") is True


def test_caso2_avisa_si_se_mueve_una_variable_critica(muestra):
    """Pocas columnas derivadas alcanzan si son de las que mas pesan.

    Esto salio de la simulacion de dos anios: el clima macroeconomico y el
    gasto mensual derivaron, pero como eran solo 2 de 18 columnas la
    proporcion daba 11 por ciento y la alerta nunca se activaba. Y son dos de
    las cinco variables mas importantes del modelo.
    """
    mitad = len(muestra) // 2
    referencia = muestra.iloc[:mitad]
    nuevos = muestra.iloc[mitad:].copy()

    # Solo dos columnas, pero las dos criticas
    nuevos["macro_climate"] = nuevos["macro_climate"] * 0.3
    nuevos["monthly_burn_rate"] = nuevos["monthly_burn_rate"] * 10

    resultado = detectar_deriva(referencia, nuevos)

    assert resultado["proporcion"] < 0.30      # no llega por cantidad
    assert resultado["hay_deriva"] is True     # pero avisa igual
    assert "macro_climate" in resultado["criticas_derivadas"]


def test_caso2_el_reporte_visual_se_guarda(muestra, tmp_path):
    """El reporte HTML de Evidently es lo que se adjunta al informe."""
    ruta = tmp_path / "reportes" / "deriva.html"
    mitad = len(muestra) // 2

    detectar_deriva(muestra.iloc[:mitad], muestra.iloc[mitad:], ruta_html=ruta)

    assert ruta.exists()
    assert "<html" in ruta.read_text(encoding="utf-8")[:2000].lower()


# ==========================================================================
# CASO 3: el despliegue vuelve atras si la version nueva falla
# ==========================================================================

def preparar_imitaciones(carpeta, curl_responde_bien):
    """Crea comandos falsos de docker y curl para probar el script sin servidor.

    El script de despliegue llama a docker y a curl. Aca se ponen dos programas
    con esos nombres al principio del PATH, que en vez de hacer algo real
    anotan en un archivo lo que se les pidio. Asi se puede comprobar que el
    script hace lo correcto sin necesidad de un VPS ni de contenedores.
    """
    binarios = carpeta / "bin"
    binarios.mkdir()
    registro = carpeta / "registro.txt"

    docker = binarios / "docker"
    docker.write_text(f"""#!/bin/bash
echo "docker $@" >> {registro}
if [ "$1" = "inspect" ]; then
    echo "ghcr.io/htacoav/startup-mcd:versionvieja"
fi
if [ "$1" = "compose" ]; then
    echo "IMAGEN=$IMAGEN" >> {registro}
fi
exit 0
""", encoding="utf-8")

    salida_curl = 0 if curl_responde_bien else 1
    cuerpo = '{"status":"ok","model_loaded":true}' if curl_responde_bien else ""
    curl = binarios / "curl"
    curl.write_text(f"""#!/bin/bash
echo "curl $@" >> {registro}
echo '{cuerpo}'
exit {salida_curl}
""", encoding="utf-8")

    for programa in (docker, curl):
        programa.chmod(programa.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return binarios, registro


def correr_despliegue(carpeta, curl_responde_bien):
    """Ejecuta el script de despliegue con los comandos falsos."""
    binarios, registro = preparar_imitaciones(carpeta, curl_responde_bien)

    entorno = dict(os.environ)
    entorno["PATH"] = f"{binarios}:{entorno['PATH']}"

    proceso = subprocess.run(
        ["bash", str(RAIZ / "servidor" / "desplegar_remoto.sh"),
         "ghcr.io/htacoav/startup-mcd", "abc123"],
        cwd=str(RAIZ), env=entorno, capture_output=True, text=True, timeout=180,
    )
    return proceso, registro.read_text(encoding="utf-8")


def test_caso3_despliegue_correcto_cuando_la_version_nueva_responde(tmp_path):
    """Si la version nueva contesta el healthcheck, el despliegue termina bien."""
    proceso, registro = correr_despliegue(tmp_path, curl_responde_bien=True)

    assert proceso.returncode == 0
    assert "Despliegue correcto" in proceso.stdout
    assert "docker pull ghcr.io/htacoav/startup-mcd:abc123" in registro
    # No tuvo que volver atras, asi que la unica imagen levantada es la nueva
    assert "IMAGEN=ghcr.io/htacoav/startup-mcd:versionvieja" not in registro


def test_caso3_vuelve_atras_cuando_la_version_nueva_no_responde(tmp_path):
    """El caso que importa: la version nueva no arranca y hay que rescatar el servicio.

    El script tiene que terminar con error, para que GitHub marque el despliegue
    como fallido, y ademas tiene que dejar corriendo la version anterior. Si
    solo fallara sin volver atras, la aplicacion quedaria caida.
    """
    proceso, registro = correr_despliegue(tmp_path, curl_responde_bien=False)

    assert proceso.returncode == 1
    assert "Volviendo a" in proceso.stdout
    # La ultima imagen levantada tiene que ser la vieja
    levantadas = [linea for linea in registro.splitlines() if linea.startswith("IMAGEN=")]
    assert levantadas[-1] == "IMAGEN=ghcr.io/htacoav/startup-mcd:versionvieja"


# ==========================================================================
# LOS FLUJOS ESTAN BIEN ENCADENADOS
# ==========================================================================

@pytest.fixture(scope="module")
def flujo_de_integracion():
    ruta = RAIZ / ".github" / "workflows" / "integracion-continua.yml"
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def test_nunca_se_despliega_sin_pasar_las_pruebas(flujo_de_integracion):
    """El trabajo de despliegue depende del de imagen, y ese del de pruebas.

    Si alguien rompiera esa cadena, se podria desplegar codigo que no paso las
    pruebas sin que nadie lo note.
    """
    trabajos = flujo_de_integracion["jobs"]

    assert trabajos["imagen"]["needs"] == "pruebas"
    assert trabajos["desplegar"]["needs"] == "imagen"
    assert "refs/heads/main" in trabajos["desplegar"]["if"]


def test_el_mantenimiento_esta_programado():
    """El pipeline de mantenimiento tiene que correr solo, no a mano."""
    ruta = RAIZ / ".github" / "workflows" / "mantenimiento.yml"
    flujo = yaml.safe_load(ruta.read_text(encoding="utf-8"))

    # PyYAML convierte la palabra "on" en True, por eso se busca asi
    disparadores = flujo.get("on", flujo.get(True))
    assert "schedule" in disparadores
    assert disparadores["schedule"][0]["cron"]
