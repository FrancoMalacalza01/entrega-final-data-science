"""API de estimación de producción de pozos no convencionales.

Levantarla en local:

    uvicorn src.app:app --reload

Y abrir http://127.0.0.1:8000/docs para probarla desde el navegador.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import model

app = FastAPI(
    title="Productividad de pozos en Vaca Muerta",
    description=(
        "Estima los metros cúbicos de petróleo que un pozo no convencional va a producir "
        "en sus primeros doce meses, a partir de las decisiones de diseño de completación "
        "que se toman **antes** de fracturarlo.\n\n"
        "El modelo explica el 26% de la variabilidad: sirve para comparar diseños y ordenar "
        "prioridades, **no para dictaminar sobre un pozo individual**. Lo que no explica es "
        "sobre todo la geología, que no está en los datos abiertos.\n\n"
        "Cada respuesta trae un campo `avisos` que señala los datos del pedido que caen "
        "fuera del rango con el que se entrenó. Un pozo de 8.000 metros de rama se estima "
        "igual, pero con el aviso de que el más largo del entrenamiento mide 4.965: un "
        "bosque no extrapola, así que a partir de ahí devuelve siempre lo mismo."
    ),
    version="1.0.0",
)


class Diseno(BaseModel):
    """El diseño de completación de un pozo, tal como se declara al organismo."""

    longitud_rama_horizontal_m: float = Field(
        ..., ge=500, le=8000, description="Metros del tramo horizontal", examples=[2800])
    cantidad_fracturas: int = Field(
        ..., ge=1, le=200, description="Cantidad de etapas de fractura", examples=[48])
    arena_bombeada_nacional_tn: float = Field(
        ..., ge=0, le=50000, description="Toneladas de arena nacional", examples=[10900])
    arena_bombeada_importada_tn: float = Field(
        ..., ge=0, le=50000, description="Toneladas de arena importada", examples=[0])
    agua_inyectada_m3: float = Field(
        ..., ge=0, le=500000, description="Metros cúbicos de agua inyectada", examples=[69400])
    presion_maxima_psi: float = Field(
        ..., ge=0, le=20000, description="Presión máxima alcanzada", examples=[11000])
    potencia_equipos_fractura_hp: float = Field(
        ..., ge=0, le=200000, description="Potencia de los equipos", examples=[25000])
    duracion_fractura_dias: float = Field(
        ..., ge=0, le=365, description="Días que duró la operación", examples=[40])
    operadora: str = Field(
        ...,
        description=("Razón social de la operadora. No distingue mayúsculas ni espacios de "
                     "más: «YPF S.A.» y «ypf s.a.» son la misma. La lista de las que el "
                     "modelo conoce está en GET /operadoras; una que no esté no rompe, pero "
                     "la respuesta lo avisa."),
        examples=["YPF S.A."])


class Respuesta(BaseModel):
    prediccion_m3: float = Field(..., description="m³ de petróleo estimados a 12 meses")
    rango_p10_p90_m3: list[float] = Field(
        ..., description="Rango entre los percentiles 10 y 90 del voto de los árboles")
    variables_derivadas: dict = Field(
        ..., description="Variables de intensidad calculadas a partir del diseño")
    operadora_normalizada: str = Field(
        ..., description="La razón social tal como se la buscó: en minúsculas y sin espacios de más")
    operadora_conocida: bool = Field(
        ..., description="Si la operadora estaba entre las que el modelo vio al entrenar")
    avisos: list[dict] = Field(
        default_factory=list,
        description=("Datos del pedido que caen fuera del rango de entrenamiento. Vacío "
                     "cuando el pozo está dentro de lo que el modelo vio. Cada aviso trae "
                     "la variable, su valor, el rango y un `nivel`: `extrapolacion` si "
                     "quedó fuera del mínimo y el máximo observados, `poca_evidencia` si "
                     "cayó en el 1% extremo."))


@app.get("/", tags=["info"])
def raiz():
    """Ficha del servicio."""
    return {
        "servicio": app.title,
        "version": app.version,
        "documentacion": "/docs",
        "endpoints": {"POST /predict": "estimar un pozo", "GET /health": "estado",
                      "GET /modelo": "ficha del modelo",
                      "GET /operadoras": "las operadoras que el modelo conoce"},
    }


@app.get("/health", tags=["info"])
def health():
    """Estado del servicio. Devuelve 503 si el modelo no está entrenado."""
    try:
        model.cargar()
    except model.ModeloNoEntrenado as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"estado": "ok", "modelo_cargado": True}


@app.get("/modelo", tags=["info"])
def ficha_del_modelo():
    """Cuándo se entrenó el modelo, con cuántos pozos y con qué métricas."""
    try:
        return model.metadata()
    except model.ModeloNoEntrenado as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/operadoras", tags=["info"])
def operadoras():
    """Las operadoras que el modelo conoce, para saber qué escribir en el campo.

    Se puede mandar cualquier otra: no rompe nada, pero la estimación pierde el aporte de
    esa variable y la respuesta lo avisa.
    """
    try:
        return {"operadoras_conocidas": model.operadoras_conocidas(),
                "nota": ("No distingue mayúsculas. «otras» es la categoría en la que el "
                         "entrenamiento agrupó a las operadoras con menos de 40 pozos.")}
    except model.ModeloNoEntrenado as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=Respuesta, tags=["prediccion"])
def predict(diseno: Diseno):
    """Estima la producción de los primeros doce meses a partir del diseño.

    Las variables de intensidad —arena por metro, agua por metro, espaciamiento entre
    etapas— se calculan acá, con las mismas fórmulas que se usaron para entrenar.

    La respuesta incluye `avisos`: los datos del pedido que caen fuera del rango de
    entrenamiento. La estimación se devuelve igual, porque rechazar el pedido sería peor
    —el usuario no se enteraría de por qué—, pero con el aviso de cuánta confianza merece.
    """
    try:
        return model.predecir(diseno.model_dump())
    except model.ModeloNoEntrenado as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo predecir: {e}")
