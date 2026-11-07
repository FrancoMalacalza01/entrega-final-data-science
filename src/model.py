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

from src.caracteristicas import (CRUDAS, NUMERICAS, fila_desde_diseno,
                                 normalizar_operadora)

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


def rangos_entrenamiento():
    """Hasta dónde llegan los datos con los que se entrenó, variable por variable."""
    return metadata().get("rangos_entrenamiento", {})


def _aviso(variable, valor, rango):
    """Devuelve el aviso que corresponde a un valor, o None si está dentro de lo visto.

    Dos niveles, porque no es lo mismo pedirle al modelo algo raro que pedirle algo que
    nunca existió:

    * `extrapolacion` — el valor queda fuera del mínimo y el máximo del entrenamiento. El
      bosque no puede extrapolar: sus hojas guardan promedios de los pozos que vio, así que
      a partir del borde devuelve siempre lo mismo por más que se siga estirando el valor.
    * `poca_evidencia` — el valor cae dentro, pero en el 1% extremo. Hay datos, pero tan
      pocos que la estimación se apoya en un puñado de pozos.
    """
    if not rango:
        return None
    if valor < rango["min"] or valor > rango["max"]:
        lado = "por debajo" if valor < rango["min"] else "por encima"
        return {
            "variable": variable, "valor": round(float(valor), 3),
            "nivel": "extrapolacion",
            "rango_entrenamiento": [rango["min"], rango["max"]],
            "mensaje": (f"{variable} = {valor:,.1f} queda {lado} de todo lo que el modelo "
                        f"vio al entrenar ({rango['min']:,.1f} a {rango['max']:,.1f}). "
                        f"Un bosque no extrapola: a partir del borde devuelve siempre lo "
                        f"mismo. La estimación no es confiable."),
        }
    if valor < rango["p1"] or valor > rango["p99"]:
        lado = "por debajo" if valor < rango["p1"] else "por encima"
        return {
            "variable": variable, "valor": round(float(valor), 3),
            "nivel": "poca_evidencia",
            "rango_entrenamiento": [rango["p1"], rango["p99"]],
            "mensaje": (f"{variable} = {valor:,.1f} cae en el 1% extremo del entrenamiento "
                        f"({lado} de {rango['p1']:,.1f} a {rango['p99']:,.1f}). Hay datos, "
                        f"pero pocos: la estimación se apoya en un puñado de pozos."),
        }
    return None


def avisos(diseno, fila, operadora_conocida):
    """Los avisos que acompañan a una estimación: qué datos caen fuera de lo entrenado.

    Se revisan las nueve variables que el modelo ve y también las que el usuario escribió,
    porque son distintas y las dos ayudan: alargar la rama sin subir la arena deja la
    longitud dentro de rango pero saca de rango la arena por metro, y eso es justamente lo
    que conviene que se vea.
    """
    rangos = rangos_entrenamiento()
    valores = {c: float(diseno[c]) for c in CRUDAS if c in diseno}
    valores.update({c: float(fila.iloc[0][c]) for c in NUMERICAS})

    salida = [a for a in (_aviso(v, valores[v], rangos.get(v)) for v in valores) if a]
    if not operadora_conocida:
        salida.append({
            "variable": "operadora", "valor": None, "nivel": "poca_evidencia",
            "rango_entrenamiento": None,
            "mensaje": ("La operadora no está entre las que el modelo vio al entrenar. No "
                        "rompe nada, pero la estimación pierde el aporte de esa variable, "
                        "que es de las que más pesan. En GET /operadoras está la lista."),
        })
    # Primero lo que el modelo nunca vio, que es lo que de verdad invalida la estimación.
    return sorted(salida, key=lambda a: (a["nivel"] != "extrapolacion", a["variable"]))


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

    operadora = normalizar_operadora(diseno.get("operadora", ""))
    conocida = operadora in meta["operadoras_conocidas"]

    return {
        "prediccion_m3": round(prediccion, 1),
        "rango_p10_p90_m3": [round(float(np.percentile(votos, 10)), 1),
                             round(float(np.percentile(votos, 90)), 1)],
        "variables_derivadas": {k: round(v, 3) for k, v in derivadas.items()},
        "operadora_normalizada": operadora,
        "operadora_conocida": conocida,
        "avisos": avisos(diseno, fila, conocida),
    }
