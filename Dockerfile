# Imagen para desplegar la API. El modelo se entrena al construir, de modo que la
# imagen queda autocontenida y no depende de que models/ este versionado.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/
COPY models/ ./models/

# Si models/ llegara vacio, se entrena durante la construccion.
RUN python -c "from pathlib import Path; import sys; sys.exit(0 if Path('models/modelo.joblib').exists() else 1)" \
    || python src/entrenar.py

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
