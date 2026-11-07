"""Entrena el modelo que sirve la API y lo guarda en models/.

Reproduce el camino del notebook 02 sin el relato: reconstruye el dataset desde los
archivos originales, mide el desempeño con la partición temporal y guarda el modelo.

    python src/entrenar.py

Dos pasos y la diferencia importa:

1. **Medir.** Se entrena con los pozos fracturados hasta 2023 y se evalúa una sola vez
   sobre los de 2024-2025. Las métricas que la API declara salen de acá.
2. **Servir.** Se vuelve a entrenar con los 1.917 pozos, porque un modelo que va a estimar
   pozos nuevos no tiene por qué ignorar los dos últimos años. El artefacto guardado es
   éste, y la metadata deja constancia de las dos cosas.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.caracteristicas import (CATEGORICAS, CRUDAS, NUMERICAS, OBJETIVO,
                                 agregar_variables_intensidad, normalizar_operadora)
from src.utils import agrupar_categorias, leer

SEMILLA = 42
CORTE = 2024
UMBRAL_RAMA_M = 500
RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "models"

AGREGACION = {
    "longitud_rama_horizontal_m": "max", "presion_maxima_psi": "max",
    "potencia_equipos_fractura_hp": "max", "cantidad_fracturas": "sum",
    "arena_bombeada_nacional_tn": "sum", "arena_bombeada_importada_tn": "sum",
    "agua_inyectada_m3": "sum", "fecha_inicio_fractura": "min", "fecha_fin_fractura": "max",
    "formacion_productiva": "first", "empresa_informante": "first",
}


def construir_dataset():
    """Mismo pipeline que el notebook 02: de los archivos crudos a los 1.917 pozos."""
    fractura = leer("fractura_adjunto_iv.csv", low_memory=False)
    prod = leer("produccion_filtrada.csv.gz")

    df = fractura.copy()
    for c in ["fecha_inicio_fractura", "fecha_fin_fractura", "fecha_data"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # La referencia para descartar fechas imposibles sale del archivo, no del reloj.
    fecha_corte = df.fecha_data.max()
    df.loc[df.fecha_fin_fractura < df.fecha_inicio_fractura, "fecha_fin_fractura"] = pd.NaT
    df.loc[df.fecha_inicio_fractura > fecha_corte, "fecha_inicio_fractura"] = pd.NaT

    for c in ["formacion_productiva", "empresa_informante"]:
        df[c] = normalizar_operadora(df[c]).replace({"nan": "sin dato", "": "sin dato"})

    df = df[df.longitud_rama_horizontal_m >= UMBRAL_RAMA_M]
    pozos = df.groupby("idpozo").agg(**{k: (k, v) for k, v in AGREGACION.items()}).reset_index()

    # Ceros centinela y extremos del 1%, imputados por mediana de formacion.
    for c in ["presion_maxima_psi", "potencia_equipos_fractura_hp", "agua_inyectada_m3"]:
        falta = (pozos[c] == 0) | pozos[c].isna()
        mediana = pozos.loc[~falta].groupby("formacion_productiva")[c].median()
        pozos.loc[falta, c] = pozos.loc[falta, "formacion_productiva"].map(mediana)
        pozos[c] = pozos[c].fillna(pozos.loc[~falta, c].median())
    for c in ["presion_maxima_psi", "potencia_equipos_fractura_hp"]:
        lo, hi = pozos[c].quantile([0.01, 0.99])
        extremo = (pozos[c] < lo) | (pozos[c] > hi)
        mediana = pozos.loc[~extremo].groupby("formacion_productiva")[c].median()
        pozos.loc[extremo, c] = pozos.loc[extremo, "formacion_productiva"].map(mediana)
        pozos[c] = pozos[c].fillna(pozos.loc[~extremo, c].median())

    pozos["duracion_fractura_dias"] = (pozos.fecha_fin_fractura
                                       - pozos.fecha_inicio_fractura).dt.days
    pozos["anio_fractura"] = pozos.fecha_inicio_fractura.dt.year
    pozos = agregar_variables_intensidad(pozos)

    # Variable objetivo: m3 de petroleo de los primeros 12 meses, con los dos guardas.
    prod["periodo"] = pd.to_datetime(dict(year=prod.anio, month=prod.mes, day=1)).dt.to_period("M")
    activos = prod[prod.prod_pet > 0].copy()
    primer_mes = activos.groupby("idpozo").periodo.min().rename("primer_mes")
    activos = activos.merge(primer_mes, on="idpozo")
    activos["mes_de_vida"] = ((activos.periodo.dt.year - activos.primer_mes.dt.year) * 12
                              + (activos.periodo.dt.month - activos.primer_mes.dt.month) + 1)
    ventana = activos[activos.mes_de_vida.between(1, 12)]
    objetivo = ventana.groupby("idpozo").agg(
        prod_pet_acum_12m=("prod_pet", "sum"), prod_gas_acum_12m=("prod_gas", "sum"),
        primer_mes=("primer_mes", "first")).reset_index()
    p_min, p_max = prod.periodo.min(), prod.periodo.max()
    objetivo = objetivo[~(objetivo.primer_mes <= p_min)
                        & ~((objetivo.primer_mes + 11) > p_max)].copy()

    datos = pozos.merge(objetivo, on="idpozo", how="inner")

    # Los pozos cuyo objetivo es el gas se miden con la metrica equivocada: quedan afuera.
    ratio = datos.prod_gas_acum_12m / datos.prod_pet_acum_12m.replace(0, np.nan)
    datos = datos[~((datos.prod_pet_acum_12m < 1000) & (ratio > 5))].copy()

    datos["operadora"] = agrupar_categorias(datos.empresa_informante)
    return datos


def rangos_observados(datos):
    """Hasta dónde llegan los datos con los que se entrenó, variable por variable.

    La API los usa para avisar cuando un pozo cae afuera. Se guardan dos pares: el mínimo y
    el máximo observados, que marcan el borde más allá del cual el modelo extrapola, y los
    percentiles 1 y 99, que marcan dónde empieza el 1% extremo —ahí dentro hay datos, pero
    tan pocos que la estimación se apoya en un puñado de pozos—.
    """
    rangos = {}
    for c in sorted(set(NUMERICAS + CRUDAS)):
        s = datos[c].dropna()
        if s.empty:
            continue
        p1, p99 = s.quantile([0.01, 0.99])
        rangos[c] = {"min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                     "p1": round(float(p1), 4), "p99": round(float(p99), 4)}
    return rangos


def nuevo_modelo():
    """El Random Forest del punto 13 del notebook 02, con sus hiperparametros."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
    return Pipeline([("pre", pre),
                     ("mod", RandomForestRegressor(
                         n_estimators=600, min_samples_leaf=8, max_features=0.4,
                         random_state=SEMILLA, n_jobs=-1))])


def main():
    print("Reconstruyendo el dataset...")
    datos = construir_dataset()
    columnas = NUMERICAS + CATEGORICAS
    print(f"  {len(datos):,} pozos de petroleo con produccion de 12 meses medida")

    # --- 1. Medir, con la particion temporal -------------------------------------
    train = datos[datos.anio_fractura < CORTE]
    test = datos[datos.anio_fractura >= CORTE]
    print(f"\nParticion temporal: {len(train):,} pozos hasta {CORTE - 1}, "
          f"{len(test):,} de {CORTE} en adelante")

    medicion = nuevo_modelo().fit(train[columnas], train[OBJETIVO])
    pred = medicion.predict(test[columnas])
    metricas = {
        "MAE": float(mean_absolute_error(test[OBJETIVO], pred)),
        "RMSE": float(mean_squared_error_raiz(test[OBJETIVO], pred)),
        "R2": float(r2_score(test[OBJETIVO], pred)),
    }
    referencia = float(mean_absolute_error(
        test[OBJETIVO], np.full(len(test), train[OBJETIVO].median())))
    metricas["MAE_referencia"] = referencia
    metricas["mejora_vs_referencia"] = 1 - metricas["MAE"] / referencia
    print(f"  MAE  {metricas['MAE']:,.0f} m3   R2 {metricas['R2']:.3f}   "
          f"({metricas['mejora_vs_referencia']:.1%} mejor que la referencia)")

    # --- 2. Servir, reentrenando con todo ----------------------------------------
    print("\nReentrenando con los 1.917 pozos para servir...")
    final = nuevo_modelo().fit(datos[columnas], datos[OBJETIVO])

    MODELOS.mkdir(exist_ok=True)
    ruta = MODELOS / "modelo.joblib"
    joblib.dump(final, ruta, compress=3)

    metadata = {
        "entrenado": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "modelo": "RandomForestRegressor",
        "objetivo": "m3 de petroleo acumulados en los primeros 12 meses",
        "pozos_entrenamiento": int(len(datos)),
        "variables": columnas,
        "operadoras_conocidas": sorted(datos.operadora.unique().tolist()),
        "rangos_entrenamiento": rangos_observados(datos),
        "metricas_validacion_temporal": metricas,
        "nota_metricas": (f"Medidas sobre un modelo entrenado solo con los {len(train):,} pozos "
                          f"fracturados hasta {CORTE - 1} y evaluado sobre los {len(test):,} de "
                          f"{CORTE}-2025. El modelo que sirve la API se reentreno con los "
                          f"{len(datos):,} pozos."),
        "versiones": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__},
        "semilla": SEMILLA,
    }
    (MODELOS / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    print(f"\nGuardado: {ruta}  ({ruta.stat().st_size / 1e6:.1f} MB)")
    print(f"Guardado: {MODELOS / 'metadata.json'}")


def mean_squared_error_raiz(y, p):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y, p) ** 0.5


if __name__ == "__main__":
    main()
