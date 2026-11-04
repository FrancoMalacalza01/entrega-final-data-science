"""Funciones compartidas por los notebooks del proyecto.

Lo que vive acá es plomería: leer los archivos, unificar razones sociales de las
operadoras, agrupar categorías poco frecuentes y calcular métricas. Son cosas que
se usan en más de un notebook y que no aportan nada a la lectura del análisis, así
que separarlas deja los cuadernos con las decisiones y los resultados.

Lo que NO vive acá, a propósito: la construcción de la variable objetivo, los
guardas de truncamiento, la partición temporal y la definición de los modelos. Son
decisiones del análisis y tienen que leerse en el cuaderno donde se justifican.

Uso desde `notebooks/`:

    import sys
    sys.path.insert(0, "..")
    from src.utils import leer, metricas_regresion
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

__all__ = [
    "URL_REPO", "carpeta_datos", "leer",
    "NOMBRES_UNIFICADOS", "MIN_POZOS_CATEGORIA", "unificar_razones_sociales",
    "agrupar_categorias", "con_operadora_normalizada",
    "metricas_regresion", "metricas_clasificacion",
    "f3", "vif",
]


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #

URL_REPO = ("https://raw.githubusercontent.com/FrancoMalacalza01/"
            "entrega-final-data-science/main/data/")

# Los notebooks se abren desde notebooks/, pero los scripts corren desde la raiz.
CANDIDATOS_DATA = (Path("data"), Path("..") / "data")


def carpeta_datos():
    """Devuelve la carpeta data/ local, o None si no esta (por ejemplo, en Colab)."""
    return next((p for p in CANDIDATOS_DATA if p.exists()), None)


def leer(nombre, **kw):
    """Lee un archivo de data/.

    Se intenta primero la copia local, asi el notebook corre sin conexion; si no
    aparece —por ejemplo al abrirlo en Colab— se lee directamente del repositorio
    por su URL. Los kwargs van tal cual a `pd.read_csv`.
    """
    carpeta = carpeta_datos()
    if carpeta is not None and (carpeta / nombre).exists():
        print(f"  {nombre:32s} <- {carpeta / nombre}")
        return pd.read_csv(carpeta / nombre, **kw)
    print(f"  {nombre:32s} <- repositorio (URL)")
    return pd.read_csv(URL_REPO + nombre, **kw)


# --------------------------------------------------------------------------- #
# Razones sociales y categorias
# --------------------------------------------------------------------------- #

# Una misma operadora declarada bajo dos razones sociales distintas. Se detecto en
# el punto 11 del notebook 02: las dos filas comparten yacimientos y perfil de
# diseño, y el cambio de nombre ocurre en 2023.
NOMBRES_UNIFICADOS = {"vista oil & gas argentina sau": "vista energy argentina sau"}

# Minimo de pozos para que una operadora tenga categoria propia. Por debajo, el
# one-hot gasta una columna en un puñado de filas y el modelo no puede aprender nada.
MIN_POZOS_CATEGORIA = 40


def unificar_razones_sociales(serie):
    """Unifica las razones sociales que corresponden a una misma operadora."""
    return serie.replace(NOMBRES_UNIFICADOS)


def agrupar_categorias(serie, minimo=MIN_POZOS_CATEGORIA, conteo=None,
                       etiqueta="otras"):
    """Agrupa bajo una sola etiqueta los valores con menos de `minimo` registros.

    `conteo` permite contar sobre un conjunto distinto del que se transforma, que es
    lo que hace falta para que train y test compartan exactamente las mismas
    categorias.
    """
    if conteo is None:
        conteo = serie.value_counts()
    return serie.where(serie.map(conteo) >= minimo, etiqueta)


def con_operadora_normalizada(d, conteo, minimo=MIN_POZOS_CATEGORIA):
    """Devuelve una copia con la columna operadora_norm (razones sociales unificadas).

    Trabaja sobre una copia a proposito: compara encodings sin tocar el dataset de
    modelado ni la particion.
    """
    e = unificar_razones_sociales(d.empresa_informante)
    return d.assign(operadora_norm=agrupar_categorias(e, minimo, conteo))


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #

def metricas_regresion(y_real, y_pred):
    """MAE y RMSE en m3 (la unidad del objetivo), y R2."""
    return {"MAE": mean_absolute_error(y_real, y_pred),
            "RMSE": mean_squared_error(y_real, y_pred) ** 0.5,
            "R2": r2_score(y_real, y_pred)}


def metricas_clasificacion(c_real, c_pred, probabilidad):
    """Las cuatro metricas sobre la clase positiva, mas el AUC sobre la probabilidad."""
    return {"exactitud": accuracy_score(c_real, c_pred),
            "precision": precision_score(c_real, c_pred, zero_division=0),
            "sensibilidad": recall_score(c_real, c_pred, zero_division=0),
            "F1": f1_score(c_real, c_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(c_real, probabilidad)}


# --------------------------------------------------------------------------- #
# Presentacion y diagnostico
# --------------------------------------------------------------------------- #

def f3(x):
    """Formato de tabla: enteros con separador arriba de 1.000, 3 decimales abajo.

    Hace falta porque en la misma tabla convive un MAE de 10.596 m3 con un R2 de
    0,263: con un unico formato de dos decimales, el MAE se llena de ceros y —peor—
    el R2 se imprime como 0,26, de modo que una afirmacion del texto como "sube de
    0,263 a 0,285" no se puede verificar contra la tabla.
    """
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.3f}"


def vif(df, columnas):
    """VIF = 1/(1-R2) de regresar cada variable contra todas las demas."""
    X = df[columnas].dropna().to_numpy(float)
    resultado = {}
    for i, c in enumerate(columnas):
        objetivo_i = X[:, i]
        resto = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
        beta, *_ = np.linalg.lstsq(resto, objetivo_i, rcond=None)
        r2 = 1 - ((objetivo_i - resto @ beta)**2).sum() / \
                 ((objetivo_i - objetivo_i.mean())**2).sum()
        resultado[c] = 1 / max(1e-12, 1 - r2)
    return pd.Series(resultado).sort_values(ascending=False).round(2)
