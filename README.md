# Proyecto Final - Prediccion de Supervivencia de Startups

Maestria en Ciencia de Datos, Universidad Nacional del Altiplano

Aplicacion que estima la probabilidad de que un emprendimiento sobreviva, a partir de lo
que se sabe de el al momento de fundarse. Desplegada en produccion con mantenimiento e
integracion continua automatizados.

## Equipo

| Integrante |
|---|
| Taco Aviles Milton Hiroshi |
| |
| |

## La pregunta del proyecto

Que caracteristicas permiten anticipar que emprendimiento tiene mayor probabilidad de
sobrevivir.

Para responderla, el entrenamiento no solo ajusta un modelo: mide cuanto aporta cada grupo
de variables por separado. Ese estudio se publica en el endpoint /feature-groups y se
muestra en la pagina web.

## La respuesta

| Grupo | AUC solo con ese grupo | Se pierde al quitarlo |
|---|---|---|
| Ejecucion | 0.6465 | -0.0449 |
| Financiamiento y caja | 0.6420 | -0.0470 |
| Circunstancia y suerte | 0.5974 | -0.0187 |
| Equipo fundador | 0.5702 | -0.0112 |
| Perfil del fundador | 0.5430 | -0.0039 |
| Contexto, industria | 0.5347 | -0.0030 |

Lo que se hace importa mas que quien lo hace. Quitar todas las variables del fundador
(salidas exitosas previas, fracasos anteriores, anios de experiencia, cofundador tecnico,
edad, horas trabajadas) cuesta 0.0039 de AUC, practicamente nada. La suerte del clima
macroeconomico predice mejor que el curriculum del fundador.

Las variables que mas pesan individualmente son el encaje producto-mercado, la tasa de
quema mensual, los meses de caja, el clima macro y el conflicto entre cofundadores. No
tienen efecto medible: pivots, cofundador tecnico, fracasos previos, horas semanales,
anio de fundacion, numero de cofundadores y edad del fundador.

## El modelo

| Modelo | ROC-AUC | F1 | Precision | Recall | Umbral |
|---|---|---|---|---|---|
| LogisticRegression | 0.7469 | 0.8084 | 0.6975 | 0.9612 | 0.37 |
| HistGradientBoosting | 0.7363 | 0.8072 | 0.6978 | 0.9571 | 0.43 |
| RandomForest | 0.7305 | 0.8061 | 0.7014 | 0.9476 | 0.45 |

Gana la regresion logistica, que es el modelo mas simple de los tres. Vale la pena
subrayarlo: en datos tabulares suele ganar el boosting, y que aca no lo haga indica que
las relaciones son basicamente aditivas.

Particion 60 entrenamiento, 20 validacion, 20 prueba. Los hiperparametros se buscan con
RandomizedSearchCV sobre validacion cruzada de 5 pliegues. El umbral de decision se elige
en el conjunto de validacion maximizando F1, nunca en el de prueba.

## Sobre el dataset

startup_survival_master.csv, 48 000 emprendimientos con 30 columnas, de las cuales se usan
17. Se descartan seis por fuga de informacion (solo se conocen despues del desenlace) y
siete porque no aportan nada al modelo.

El conjunto es una simulacion calibrada sobre la literatura de fracaso de startups, no son
empresas reales. Eso hay que declararlo en el informe. Lo que si se verifico es que la
simulacion tiene ruido genuino: un modelo lineal comete 13 842 errores sobre las 48 000
filas y las clases se solapan. O sea que no hay formula escondida y el problema de
modelado es real.

La consecuencia para la redaccion es de una frase: no se afirma que la ejecucion importe
mas que el perfil del fundador en el mundo real, se afirma que el modelo recupera esa
estructura del conjunto de datos, y que coincide con lo que reporta la literatura.

## Como correr

### Con Docker, que es como corre en produccion

    docker compose up -d --build     # construye y levanta
    docker compose ps                # estado, debe decir healthy
    docker compose logs -f           # ver los logs
    docker compose down              # detener

Y se abre http://localhost:8000

Medido en local: imagen de 765 MB, el contenedor usa 122 MB de RAM y pasa a healthy en
unos 10 segundos.

### La exploracion de datos

    jupyter notebook notebooks/exploracion.ipynb

El cuaderno explora el dataset y ademas genera las 7 figuras del informe, que quedan en
docs/figuras/. Hacerlo asi evita que los graficos del informe queden desactualizados
cuando se reentrena el modelo: se vuelve a ejecutar el cuaderno y se regenera el
documento, y los numeros siempre coinciden.

### Sin Docker, para desarrollar

    python -m pytest                                    # 28 pruebas
    python -m ml.train                                  # reentrenar, unos 27 s
    python -m uvicorn app.main:app --reload --port 8000

### Dos casos para la demostracion

Startup sana: encaje producto-mercado 9, conflicto 2, equipo 8, caja 30 meses, vc_seed.
Devuelve 85.8 por ciento de supervivencia, riesgo LOW.

Startup en problemas: encaje 1, conflicto 9, equipo 3, caja 4 meses, bootstrapped,
escalamiento prematuro. Devuelve 1.5 por ciento, riesgo HIGH.

## Organizacion del codigo

| Carpeta | Que hay |
|---|---|
| ml/ | configuracion, carga de datos y entrenamiento |
| app/ | la API FastAPI y la pagina web |
| tests/ | 28 pruebas con pytest |
| models/ | el modelo entrenado y sus metricas |
| data/ | el dataset, no se sube a GitHub |
| docs/ | el informe y sus figuras |
| notebooks/ | la exploracion de datos |

## Las dos vistas

La aplicacion tiene dos caras porque tiene dos tipos de usuario.

La raiz es el cuestionario para emprendedores. Nadie sabe cual es su
"product market fit score de 0 a 10", pero todos saben si tienen clientes que
pagan. Son 14 preguntas en lenguaje llano que se traducen a las 17 variables
del modelo.

/expert es la vista tecnica, con los 17 valores numericos directos. Sirve para
un analista de la incubadora que ya tiene los puntajes evaluados.

Las dos terminan en el mismo modelo, y hay una prueba que verifica que dan
exactamente el mismo resultado para la misma startup.

### De donde salen los numeros del cuestionario

Cada pregunta de cuatro opciones ubica a la startup en un cuarto de la poblacion
del dataset, y el valor asignado es el punto medio de ese cuartil. Por ejemplo,
el cuarto mas bajo de product_market_fit_score ronda 1.9 y el mas alto 7.3, asi
que "todavia no tengo clientes" vale 1.9 y "crecen cada mes" vale 7.3.

Los meses de caja no se preguntan, se calculan dividiendo el efectivo disponible
entre el gasto mensual, que son dos cosas que un emprendedor si sabe.

Esta traduccion es una regla de negocio documentada, no algo que el modelo haya
aprendido, y como tal se declara en el informe. El endpoint /assess devuelve las
17 variables que uso, para que el resultado sea auditable.

## Rutas de la aplicacion

| Ruta | Que hace |
|---|---|
| / | el cuestionario para emprendedores |
| /expert | la vista tecnica con los 17 valores numericos |
| /assess | evalua desde las respuestas del cuestionario |
| /questionnaire | las preguntas, para armar la pagina |
| /predict | devuelve la probabilidad desde los 17 numeros |
| /health | estado del servicio, lo usa el healthcheck de Docker |
| /model-info | version del modelo, hiperparametros y metricas |
| /feature-groups | el estudio que responde la pregunta del proyecto |
| /docs | documentacion interactiva de la API |

## El informe

El informe esta en docs/, en Word y en PDF, en formato APA 7. Las 7 figuras que usa
salen del cuaderno notebooks/exploracion.ipynb, que las guarda en docs/figuras/. Si se
reentrena el modelo, basta con volver a correr el cuaderno para tener las figuras al dia.

La presentacion de la Unidad I tambien esta en docs/.
