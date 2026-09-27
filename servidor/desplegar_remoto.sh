#!/bin/bash
# ==========================================
# DESPLIEGUE EN EL VPS, CON VUELTA ATRAS
#
# Lo ejecuta GitHub Actions por SSH. Recibe dos datos:
#   $1  el nombre de la imagen, por ejemplo ghcr.io/htacoav/startup-mcd
#   $2  el hash del commit que se quiere desplegar
#
# La idea es simple: antes de cambiar nada se anota que imagen estaba
# corriendo. Si la version nueva no responde en 60 segundos, se vuelve a la
# anterior. Asi un despliegue malo no deja la aplicacion caida.
# ==========================================

set -e

NOMBRE_IMAGEN="$1"
COMMIT="$2"
IMAGEN_NUEVA="$NOMBRE_IMAGEN:$COMMIT"

echo "==> 1. Anotando que version esta corriendo ahora"
# Si es el primer despliegue todavia no hay contenedor, y el comando falla.
# El "|| echo" evita que el script se corte en ese caso.
IMAGEN_ANTERIOR=$(docker inspect --format '{{.Config.Image}}' startup-survival-api 2>/dev/null || echo "")
if [ -z "$IMAGEN_ANTERIOR" ]; then
    echo "    no hay despliegue previo, este es el primero"
else
    echo "    version actual: $IMAGEN_ANTERIOR"
fi

echo "==> 2. Bajando la imagen nueva"
docker pull "$IMAGEN_NUEVA"

echo "==> 3. Levantando la version nueva"
IMAGEN="$IMAGEN_NUEVA" docker compose -f docker-compose.prod.yml up -d

echo "==> 4. Esperando a que responda"
SALUDABLE="no"
for intento in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health | grep -q '"model_loaded":true'; then
        SALUDABLE="si"
        break
    fi
    sleep 2
done

if [ "$SALUDABLE" = "si" ]; then
    echo "==> Despliegue correcto: $IMAGEN_NUEVA"
    # Se borran las imagenes viejas que ya nadie usa, para no llenar el disco
    docker image prune -f > /dev/null
    exit 0
fi

echo "==> La version nueva no respondio en 60 segundos"

if [ -z "$IMAGEN_ANTERIOR" ]; then
    echo "==> No hay version anterior a la cual volver"
    docker compose -f docker-compose.prod.yml logs --tail 40 api
    exit 1
fi

echo "==> Volviendo a $IMAGEN_ANTERIOR"
docker compose -f docker-compose.prod.yml logs --tail 40 api
IMAGEN="$IMAGEN_ANTERIOR" docker compose -f docker-compose.prod.yml up -d
exit 1
