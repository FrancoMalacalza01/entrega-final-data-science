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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.caracteristicas import (CRUDAS, NUMERICAS, fila_desde_diseno,
                                 filas_desde_disenos, normalizar_operadora)

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


def _num(x):
    """Formato de número en castellano: 10.596,5 y no 10,596.5."""
    return f"{x:,.1f}".translate(str.maketrans(",.", ".,"))


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
            "mensaje": (f"{variable} = {_num(valor)} queda {lado} de todo lo que el modelo "
                        f"vio al entrenar ({_num(rango['min'])} a {_num(rango['max'])}). "
                        f"Un bosque no extrapola: a partir del borde devuelve siempre lo "
                        f"mismo. La estimación no es confiable."),
        }
    if valor < rango["p1"] or valor > rango["p99"]:
        lado = "por debajo" if valor < rango["p1"] else "por encima"
        return {
            "variable": variable, "valor": round(float(valor), 3),
            "nivel": "poca_evidencia",
            "rango_entrenamiento": [rango["p1"], rango["p99"]],
            "mensaje": (f"{variable} = {_num(valor)} cae en el 1% extremo del entrenamiento "
                        f"({lado} de {_num(rango['p1'])} a {_num(rango['p99'])}). Hay datos, "
                        f"pero pocos: la estimación se apoya en un puñado de pozos."),
        }
    return None


def avisos_de_valores(valores):
    """Los avisos que corresponden a un diccionario {variable: valor}."""
    rangos = rangos_entrenamiento()
    return [a for a in (_aviso(v, valores[v], rangos.get(v)) for v in valores) if a]


def avisos_de_rango(diseno, fila):
    """Los avisos que vienen de los números: qué datos caen fuera de lo entrenado.

    Se revisan las nueve variables que el modelo ve y también las que el usuario escribió,
    porque son distintas y las dos ayudan: alargar la rama sin subir la arena deja la
    longitud dentro de rango pero saca de rango la arena por metro, y eso es justamente lo
    que conviene que se vea.
    """
    valores = {c: float(diseno[c]) for c in CRUDAS if c in diseno}
    valores.update({c: float(fila.iloc[0][c]) for c in NUMERICAS})
    return avisos_de_valores(valores)


def nivel_mas_grave(lista):
    """El peor de los niveles de una lista de avisos, o None si no hay ninguno."""
    niveles = {a["nivel"] for a in lista}
    return ("extrapolacion" if "extrapolacion" in niveles
            else "poca_evidencia" if niveles else None)


def avisos(diseno, fila, operadora_conocida):
    """Los avisos de rango más el de la operadora desconocida, ordenados por gravedad."""
    salida = avisos_de_rango(diseno, fila)
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


ESCALAN_CON_LA_RAMA = ("cantidad_fracturas", "arena_bombeada_nacional_tn",
                       "arena_bombeada_importada_tn", "agua_inyectada_m3")


def _mover(diseno, variable, valor, mantener_intensidad):
    """Copia del diseño con una variable movida a `valor`.

    Con `mantener_intensidad`, mover la rama arrastra las etapas, la arena y el agua en la
    misma proporción. Es la comparación que interesa: un pozo más largo completado con la
    misma receta por metro. Sin eso, alargar la rama diluye el diseño —la arena por metro
    cae— y la estimación baja, que es un resultado correcto pero responde otra pregunta.
    """
    d = dict(diseno)
    d[variable] = valor
    if variable == "longitud_rama_horizontal_m" and mantener_intensidad:
        k = valor / diseno["longitud_rama_horizontal_m"]
        d["cantidad_fracturas"] = max(1, round(diseno["cantidad_fracturas"] * k))
        for c in ESCALAN_CON_LA_RAMA[1:]:
            d[c] = round(diseno[c] * k, 1)
    return d


def barrido(diseno, variable="longitud_rama_horizontal_m", desde=None, hasta=None,
            puntos=41, mantener_intensidad=True):
    """La curva de la estimación a lo largo de una variable, con su banda y sus avisos.

    Las `puntos` filas se arman primero y se pasan al modelo **en un solo lote**, que es lo
    que hace viable dibujar la curva en vivo: el ColumnTransformer y los 600 árboles
    vectorizan sobre las filas, así que una curva de 41 puntos con banda tarda unos 50 ms,
    contra los casi cinco segundos que lleva pedir los 41 puntos de a uno.

    Cada punto trae `nivel_aviso` —`extrapolacion`, `poca_evidencia` o None— para que quien
    dibuje la curva pueda marcar dónde el modelo se queda sin datos.
    """
    modelo, meta = cargar()
    rango = rangos_entrenamiento().get(variable, {})
    if desde is None:
        desde = rango.get("min", diseno[variable] * 0.5)
    if hasta is None:
        hasta = rango.get("max", diseno[variable] * 1.5)
    if puntos < 2:
        raise ValueError("un barrido necesita al menos dos puntos")

    paso = (hasta - desde) / (puntos - 1)
    valores = [desde + paso * i for i in range(puntos)]
    # El pozo actual viaja como una fila mas del mismo lote: pedirlo aparte costaria tanto
    # como los 41 puntos de la curva juntos.
    disenos = [_mover(diseno, variable, v, mantener_intensidad) for v in valores]
    filas = filas_desde_disenos(disenos + [diseno])

    centrales = modelo.predict(filas)
    transformada = modelo.named_steps["pre"].transform(filas)
    votos = np.array([arbol.predict(transformada)
                      for arbol in modelo.named_steps["mod"].estimators_])
    p10, p90 = np.percentile(votos[:, :puntos], [10, 90], axis=0)

    # Las derivadas salen de la matriz y no de filas.iloc[i]: indexar pandas 41 veces
    # cuesta mas que todo el resto del barrido junto.
    derivadas = filas[NUMERICAS].to_numpy(float)
    todos = []
    for i, d in enumerate(disenos):
        vals = {c: float(d[c]) for c in CRUDAS if c in d}
        vals.update(zip(NUMERICAS, derivadas[i]))
        todos.append(vals)

    # Para sombrear el grafico interesa lo que se mueve a lo largo del barrido. Un dato que
    # esta fuera de rango pero vale lo mismo en los 41 puntos pintaria la curva entera y no
    # diria nada de este tramo; sigue estando en los avisos del pozo actual.
    movibles = [c for c in todos[0] if len({v[c] for v in todos}) > 1]

    curva = []
    for i, (valor, vals) in enumerate(zip(valores, todos)):
        # Solo los avisos de rango: el de la operadora desconocida valdria para todos
        # los puntos por igual y no dice nada sobre este tramo de la curva.
        curva.append({
            "valor": round(float(valor), 2),
            "prediccion_m3": round(float(centrales[i]), 1),
            "p10": round(float(p10[i]), 1),
            "p90": round(float(p90[i]), 1),
            "nivel_aviso": nivel_mas_grave(avisos_de_valores(vals)),
            "nivel_eje": nivel_mas_grave(
                avisos_de_valores({c: vals[c] for c in movibles})),
        })

    return {
        "variable": variable,
        "mantener_intensidad": bool(mantener_intensidad),
        "rango_entrenamiento": rango or None,
        "curva": curva,
        "actual": _armar_respuesta(diseno, filas.iloc[[puntos]],
                                   centrales[puntos], votos[:, puntos]),
    }


def _armar_respuesta(diseno, fila, central, votos):
    """Arma la respuesta de una estimación, a partir de su fila y del voto de los árboles.

    La comparten `predecir`, que trabaja sobre una fila sola, y `barrido`, que saca el punto
    actual del mismo lote que la curva. Así el número que muestra el gráfico y el que
    devuelve /predict no pueden diferir.
    """
    conocida = (normalizar_operadora(diseno.get("operadora", ""))
                in metadata()["operadoras_conocidas"])
    derivadas = {c: float(fila.iloc[0][c]) for c in NUMERICAS if c not in diseno}
    return {
        "prediccion_m3": round(float(central), 1),
        "rango_p10_p90_m3": [round(float(np.percentile(votos, 10)), 1),
                             round(float(np.percentile(votos, 90)), 1)],
        "variables_derivadas": {k: round(v, 3) for k, v in derivadas.items()},
        "operadora_normalizada": normalizar_operadora(diseno.get("operadora", "")),
        "operadora_conocida": conocida,
        "avisos": avisos(diseno, fila, conocida),
    }


def predecir(diseno):
    """Estima los m³ de petróleo de los primeros 12 meses a partir del diseño del pozo.

    Devuelve también un rango: cada árbol del bosque vota un número, y el desacuerdo entre
    ellos da una idea de cuánta confianza merece la estimación. No es un intervalo de
    confianza estadístico, y el diagnóstico de error del notebook 02 sigue valiendo: en los
    extremos de producción el modelo comprime sus predicciones hacia el centro.
    """
    modelo, _ = cargar()
    fila = fila_desde_diseno(diseno)
    central = float(modelo.predict(fila)[0])

    # Voto de cada arbol, para el rango.
    transformada = modelo.named_steps["pre"].transform(fila)
    votos = np.concatenate([arbol.predict(transformada)
                            for arbol in modelo.named_steps["mod"].estimators_])
    return _armar_respuesta(diseno, fila, central, votos)
