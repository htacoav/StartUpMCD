#!/bin/bash
# ==========================================
# DESPLIEGUE AL VPS
#
# Uso:   ./desplegar.sh root@TU_IP
#
# Copia el proyecto al servidor, construye la imagen y levanta el contenedor.
# En la Unidad II este mismo script lo va a ejecutar GitHub Actions.
# ==========================================

set -e   # si algo falla, el script se detiene aca mismo

SERVIDOR="$1"
# Carpeta donde vive el proyecto en el servidor. Va en /opt porque ahi se
# guardan las aplicaciones que uno instala aparte del sistema. Escribir en /opt
# necesita root, asi que el despliegue se hace entrando como root.
CARPETA_REMOTA="/opt/startup-survival"

if [ -z "$SERVIDOR" ]; then
    echo "Falta decir a que servidor. Ejemplo:"
    echo "   ./desplegar.sh root@191.101.1.1"
    exit 1
fi

echo "==> 1. Copiando el proyecto a $SERVIDOR"
# --delete borra en el servidor lo que ya no existe aca, para que no queden
# archivos viejos dando vueltas. El dataset no se copia: la API no lo necesita,
# solo hace falta para reentrenar.
rsync -az --delete \
    --exclude 'data/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.DS_Store' \
    --exclude '.git/' \
    --exclude 'docs/' \
    ./ "$SERVIDOR:$CARPETA_REMOTA/"

echo "==> 2. Construyendo y levantando el contenedor"
ssh "$SERVIDOR" "cd $CARPETA_REMOTA && docker compose up -d --build"

echo "==> 3. Esperando a que el servicio responda"
ssh "$SERVIDOR" "for i in \$(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo 'El servicio responde'
        exit 0
    fi
    sleep 2
done
echo 'El servicio no respondio en 60 segundos'
exit 1"

echo "==> 4. Estado final"
ssh "$SERVIDOR" "cd $CARPETA_REMOTA && docker compose ps && curl -s http://localhost:8000/health"
echo ""
echo "Listo. La aplicacion esta en http://${SERVIDOR#*@}:8000"
