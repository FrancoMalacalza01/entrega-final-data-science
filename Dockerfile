# Imagen para desplegar la API de prediccion.
#
# Usa requirements-api.txt y no requirements.txt: el servicio no necesita jupyter ni
# las librerias de graficos, que solo harian la imagen mas pesada y el build mas lento.

FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# models/ viene versionado en el repositorio, asi que normalmente esto no hace nada.
# Si el artefacto faltara, se entrena durante la construccion y la imagen igual queda
# autocontenida.
RUN test -f models/modelo.joblib || python src/entrenar.py

# Render y Hugging Face Spaces inyectan el puerto por la variable PORT. Si no esta
# definida —al correr la imagen a mano— usa 8000.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
