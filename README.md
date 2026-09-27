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

    python -m pytest                                    # 59 pruebas
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
| ml/ | configuracion, datos, entrenamiento, deriva y mantenimiento |
| app/ | la API FastAPI y las tres paginas web |
| tests/ | 59 pruebas con pytest |
| models/ | el modelo entrenado y sus metricas |
| data/ | el dataset y su diccionario |
| docs/ | los informes, la presentacion y las figuras |
| notebooks/ | la exploracion y el entrenamiento paso a paso |
| .github/workflows/ | integracion continua y mantenimiento automatico |
| servidor/ | scripts y compose para el VPS |

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
| / | el cuestionario para emprendedores, en ingles |
| /admin | seguimiento: se marca como termino cada emprendimiento evaluado |
| /es | el mismo cuestionario en espanol |
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

## Integracion continua y mantenimiento

Son dos flujos automaticos, los dos en .github/workflows/.

### Integracion continua

Se dispara con cada push a main y con cada pull request. Tiene tres trabajos
encadenados: si uno falla, los siguientes no corren.

| Trabajo | Que hace |
|---|---|
| pruebas | revisa el estilo con ruff y corre las 59 pruebas |
| imagen | construye la imagen Docker y la sube a ghcr.io |
| desplegar | la instala en el VPS y comprueba que responda |

El despliegue usa servidor/desplegar_remoto.sh, que antes de cambiar nada anota
que imagen estaba corriendo. Si la version nueva no responde en 60 segundos,
vuelve sola a la anterior.

### Mantenimiento del modelo

Corre todos los lunes a las 6 de la manana, y tambien se puede disparar a mano.

    python -m ml.mantenimiento      # el mismo pipeline, en local

Los pasos son: revisar si los datos nuevos cambiaron respecto de los del
entrenamiento, entrenar un candidato sin tocar el modelo de produccion,
compararlo, registrar todo en MLflow y decidir.

El candidato solo reemplaza al que esta en produccion si mejora el AUC en al
menos 0.002 y ademas pasa los minimos absolutos: AUC de 0.70, exactitud
balanceada de 0.65 y menos del 80 por ciento de startups senaladas. Si no pasa,
o si hay deriva de datos, se abre un issue en vez de desplegar.

Cuando el modelo si mejora, el flujo lo commitea, y ese commit dispara la
integracion continua. O sea que el mantenimiento no despliega por su cuenta:
pasa por las mismas pruebas y la misma vuelta atras que cualquier otro cambio.

### Secretos que hacen falta en GitHub

| Secreto | Para que |
|---|---|
| VPS_HOST | la IP del servidor |
| VPS_USER | el usuario con el que entra |
| VPS_SSH_KEY | la llave privada del despliegue |
| VPS_DOMINIO | el dominio publico, para la comprobacion final |
| MLFLOW_TRACKING_URI | la direccion del servidor MLflow |

## De donde salen los datos para reentrenar

Cada evaluacion se guarda en una base SQLite, que vive en un volumen de Docker
para que sobreviva a los despliegues. Pero una evaluacion guardada todavia no
sirve para entrenar: tiene las 17 variables, no tiene el desenlace. Eso se sabe
uno o dos anios despues.

Por eso existe la vista /admin, donde la incubadora marca si cada emprendimiento
cerro o sigue operando. Recien ahi esa fila sirve para entrenar.

| Ruta | Que hace |
|---|---|
| /admin | la vista donde se marcan los desenlaces |
| /outcome | registra el desenlace de una evaluacion |
| /consultations | devuelve las evaluaciones guardadas, la usa el pipeline |

Las tres se protegen con la variable TOKEN_ADMIN. Si no se define, quedan
abiertas, lo que sirve para desarrollar pero nunca para produccion.

El pipeline busca los datos nuevos en cuatro lugares, en este orden: la API de
produccion, la base local, un archivo dejado a mano, y por ultimo una
simulacion. Y aplica dos minimos: 100 evaluaciones para medir deriva y 500 con
desenlace conocido para sumarlas al entrenamiento.

El primer minimo salio de probarlo: con 4 evaluaciones reales, Evidently
reporto deriva en las 17 columnas, que era falso. Sin ese piso, el flujo abriria
una alarma todas las semanas durante los primeros meses.

## La simulacion de dos anios

    python -m simulacion.simular_dos_anios

Simula 24 meses de operacion: 3500 emprendedores responden el mes 0, la
incubadora completa los desenlaces a los 10 meses, siguen llegando 200 por mes,
y el mantenimiento corre en los meses 10, 12, 18 y 24. Deja los graficos y la
bitacora en simulacion/resultados/.

Trabaja sobre copias aisladas del modelo y de la base, asi que no toca nada de
produccion.

Las caracteristicas salen de filas reales del dataset. Lo simulado es el
departamento, el desenlace y la deriva del tiempo, cada uno con una regla
explicita escrita en el codigo. El desenlace NO se genera con el modelo: si se
hiciera asi, el modelo acertaria por construccion.

### Que encontro

Promovio un modelo nuevo en 3 de las 4 corridas. Sobre las consultas reales
apartadas, el AUC paso de 0.6518 a 0.6746 en la primera. En el mes 12 el
candidato salio peor que el vigente (0.6622 contra 0.6644) y la puerta lo
rechazo, que es lo que tiene que pasar.

Las dos cifras salen del subconjunto que el pipeline aparta y no usa para
entrenar. Una version anterior de la simulacion las calculaba sobre todas las
consultas etiquetadas, incluidas las que el candidato acababa de ver, y por eso
informaba mejoras mas grandes que las reales.

Pero lo que mas sirvio fueron los dos errores que destapo, que ninguna prueba
unitaria habria encontrado porque solo aparecen con el tiempo:

La puerta de calidad comparaba sobre el dataset completo. Asi nunca promovio
nada en dos anios: el candidato daba 0.7431 contra 0.7469. Sobre las consultas
reales ese mismo candidato daba 0.6650 contra 0.6484, o sea 0.0166 mejor. Las
48 000 filas historicas tapaban la señal de unos pocos miles de casos reales.
Ahora la comparacion se hace sobre consultas reales apartadas.

La alerta de deriva solo miraba la proporcion de columnas. La deriva simulada
movio 2 de 18 columnas, o sea 11 por ciento, y nunca aviso, aunque eran el
clima macroeconomico y el gasto mensual, dos de las cinco variables mas
importantes. Ahora tambien avisa si se mueve cualquiera de esas cinco.
