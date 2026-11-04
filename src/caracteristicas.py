"""Construcción de las variables de intensidad.

Vive en su propio módulo porque la usan dos caminos distintos y tienen que calcular
exactamente lo mismo: el entrenamiento, que la aplica sobre los 1.917 pozos del dataset,
y la API, que la aplica sobre un pozo que todavía no se fracturó.

Es la misma lección que el `Pipeline` de scikit-learn: los datos de entrenamiento y los de
predicción siguen el mismo camino. Si estas cinco fórmulas estuvieran escritas dos veces,
tarde o temprano se separarían.
"""

import numpy as np
import pandas as pd

# Lo que la operadora decide antes de fracturar, tal como lo declara al organismo.
CRUDAS = [
    "longitud_rama_horizontal_m",
    "cantidad_fracturas",
    "arena_bombeada_nacional_tn",
    "arena_bombeada_importada_tn",
    "agua_inyectada_m3",
    "presion_maxima_psi",
    "potencia_equipos_fractura_hp",
    "duracion_fractura_dias",
]

# Las nueve que ve el modelo, en el orden del notebook 02.
NUMERICAS = [
    "longitud_rama_horizontal_m",     # escala de la operacion
    "arena_por_metro",                # intensidad: arena
    "agua_por_metro",                 # intensidad: agua
    "espaciamiento_etapas_m",         # intensidad: densidad de etapas
    "ratio_agua_arena",               # tipo de fluido (slickwater vs gel)
    "ratio_arena_importada",          # calidad de la arena
    "presion_maxima_psi",             # condicion de la roca y del bombeo
    "potencia_equipos_fractura_hp",   # capacidad del equipo
    "duracion_fractura_dias",         # como salio la operacion
]

CATEGORICAS = ["operadora"]
OBJETIVO = "prod_pet_acum_12m"


def agregar_variables_intensidad(df):
    """Agrega al DataFrame las variables derivadas que usa el modelo.

    Normalizan el diseño por la escala del pozo, de modo que dos pozos de 1.500 y 4.000
    metros con la misma receta de fractura quedan descritos igual. Es el aporte de feature
    engineering de la Pre-entrega 2.
    """
    d = df.copy()
    d["arena_total_tn"] = (d.arena_bombeada_nacional_tn + d.arena_bombeada_importada_tn)
    d["arena_por_metro"] = d.arena_total_tn / d.longitud_rama_horizontal_m
    d["agua_por_metro"] = d.agua_inyectada_m3 / d.longitud_rama_horizontal_m
    d["espaciamiento_etapas_m"] = (d.longitud_rama_horizontal_m / d.cantidad_fracturas)
    d["ratio_arena_importada"] = (d.arena_bombeada_importada_tn
                                  / d.arena_total_tn.replace(0, np.nan)).fillna(0)
    d["ratio_agua_arena"] = d.agua_inyectada_m3 / d.arena_total_tn.replace(0, np.nan)
    return d


def fila_desde_diseno(diseno):
    """Convierte el diseño de un pozo (un diccionario) en la fila que espera el modelo."""
    d = pd.DataFrame([diseno])
    d = agregar_variables_intensidad(d)
    return d[NUMERICAS + CATEGORICAS]
