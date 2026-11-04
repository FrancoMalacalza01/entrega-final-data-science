"""Carga el modelo entrenado y predice la producción de un pozo.

Separado de `app.py` a propósito: acá está todo lo que tiene que ver con el modelo y nada
que tenga que ver con HTTP. Se puede importar desde un notebook, un script o un test sin
levantar un servidor.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.caracteristicas import CATEGORICAS, NUMERICAS, fila_desde_diseno

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "modelo.joblib"
RUTA_METADATA = RAIZ / "models" / "metadata.json"

_modelo = None
_metadata = None


class ModeloNoEntrenado(FileNotFoundError):
    """El artefacto no está en disco: hay que correr `python src/entrenar.py`."""


def cargar():
    """Carga el modelo una sola vez y lo deja en memoria."""
    global _modelo, _metadata
    if _modelo is None:
        if not RUTA_MODELO.exists():
            raise ModeloNoEntrenado(
                f"No se encontro {RUTA_MODELO}. Ejecuta: python src/entrenar.py")
        _modelo = joblib.load(RUTA_MODELO)
        _metadata = json.loads(RUTA_METADATA.read_text())
    return _modelo, _metadata


def metadata():
    """Ficha del modelo: cuándo se entrenó, con cuántos pozos y con qué métricas."""
    return cargar()[1]


def operadoras_conocidas():
    """Las operadoras que el modelo vio al entrenar.

    Una que no esté en la lista no rompe —el encoder la codifica como todo ceros— pero la
    predicción pierde el aporte de esa variable, así que conviene avisarlo.
    """
    return metadata()["operadoras_conocidas"]


def predecir(diseno):
    """Estima los m³ de petróleo de los primeros 12 meses a partir del diseño del pozo.

    Devuelve también un rango: cada árbol del bosque vota un número, y el desacuerdo entre
    ellos da una idea de cuánta confianza merece la estimación. No es un intervalo de
    confianza estadístico, y el diagnóstico de error del notebook 02 sigue valiendo: en los
    extremos de producción el modelo comprime sus predicciones hacia el centro.
    """
    modelo, meta = cargar()
    fila = fila_desde_diseno(diseno)

    prediccion = float(modelo.predict(fila)[0])

    # Voto de cada arbol, para el rango.
    transformada = modelo.named_steps["pre"].transform(fila)
    votos = np.array([arbol.predict(transformada)[0]
                      for arbol in modelo.named_steps["mod"].estimators_])

    derivadas = {c: float(fila.iloc[0][c])
                 for c in NUMERICAS if c not in diseno}

    return {
        "prediccion_m3": round(prediccion, 1),
        "rango_p10_p90_m3": [round(float(np.percentile(votos, 10)), 1),
                             round(float(np.percentile(votos, 90)), 1)],
        "variables_derivadas": {k: round(v, 3) for k, v in derivadas.items()},
        "operadora_conocida": diseno.get("operadora") in meta["operadoras_conocidas"],
    }
