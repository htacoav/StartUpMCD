#!/bin/bash
# ==========================================
# INSTALA DOCKER EN UN UBUNTU LIMPIO
#
# Uso (desde el servidor, como root):
#     bash instalar_docker.sh
#
# Usa el repositorio oficial de Docker y no el paquete docker.io de Ubuntu,
# porque ese trae una version vieja y sin el plugin compose, y el despliegue
# necesita poder escribir "docker compose" en dos palabras.
# ==========================================

set -e   # si algo falla, se detiene aca mismo

echo "==> 1. Paquetes basicos que hacen falta para agregar el repositorio"
apt-get update
apt-get install -y ca-certificates curl gnupg

echo "==> 2. Guardando la llave con la que Docker firma sus paquetes"
# Sin esta llave apt no confia en el repositorio y rechaza la instalacion.
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "==> 3. Agregando el repositorio de Docker"
# dpkg --print-architecture dice si el server es amd64 o arm64.
# VERSION_CODENAME es el nombre de la version de Ubuntu (jammy, noble, etc).
ARQUITECTURA=$(dpkg --print-architecture)
. /etc/os-release
VERSION_UBUNTU=$VERSION_CODENAME

echo "deb [arch=$ARQUITECTURA signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_UBUNTU stable" > /etc/apt/sources.list.d/docker.list

echo "==> 4. Instalando Docker"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> 5. Dejando el servicio prendido, tambien despues de reiniciar"
systemctl enable --now docker

echo "==> 6. Comprobando"
docker --version
docker compose version
docker run --rm hello-world

echo ""
echo "Listo. Docker quedo instalado."
