# Imagen slim: la maquina destino tiene 3.7 GB de RAM y poco margen.
FROM python:3.11-slim

WORKDIR /app

# Las dependencias van antes que el codigo para aprovechar la cache de capas:
# si solo cambia el codigo, Docker no vuelve a instalar todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY app/ ./app/
COPY models/ ./models/

# Usuario sin privilegios, no conviene correr como root en produccion.
RUN useradd --create-home apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8000

# El healthcheck lo usa Docker para reiniciar el contenedor si la API deja de responder.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
