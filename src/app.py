"""API de estimación de producción de pozos no convencionales.

Levantarla en local:

    uvicorn src.app:app --reload

Y abrir http://127.0.0.1:8000/docs para probarla desde el navegador.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import model

RUTA_DEMO = Path(__file__).resolve().parent / "demo.html"
RUTA_ESTILO_DOCS = Path(__file__).resolve().parent / "estilo_docs.html"

CINTA_DOCS = """<div class="cinta-pozos"><div class="interior"><h1>Productividad de pozos en Vaca Muerta</h1><p>Proyecto final de Ciencia de Datos &middot; EnergIA Digital 2026</p><a href="/demo">Probarlo con deslizadores &#9654;</a></div></div>"""

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
    # La pagina de /docs la arma esta aplicacion, mas abajo, para poder vestirla.
    docs_url=None,
    swagger_ui_parameters={
        # El bloque "Schemas" repite al pie lo que cada endpoint ya muestra desplegado.
        "defaultModelsExpandDepth": -1,
        # "Try it out" ya abierto: el visitante viene a probar, no a leer.
        "tryItOutEnabled": True,
        "displayRequestDuration": True,
        "docExpansion": "list",
        "defaultModelRendering": "example",
        "syntaxHighlight": {"theme": "tomorrow-night"},
    },
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

    # Un pozo entero y coherente, para que «Try it out» arranque con algo real en vez de
    # con el campo por campo que arma Swagger solo.
    model_config = {"json_schema_extra": {"example": {
        "longitud_rama_horizontal_m": 2800,
        "cantidad_fracturas": 48,
        "arena_bombeada_nacional_tn": 10900,
        "arena_bombeada_importada_tn": 0,
        "agua_inyectada_m3": 69400,
        "presion_maxima_psi": 11000,
        "potencia_equipos_fractura_hp": 25000,
        "duracion_fractura_dias": 40,
        "operadora": "YPF S.A.",
    }}}


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

    # La respuesta real del pozo de ejemplo, copiada del servicio.
    model_config = {"json_schema_extra": {"example": {
        "prediccion_m3": 38506.0,
        "rango_p10_p90_m3": [26915.1, 49433.9],
        "variables_derivadas": {
            "arena_por_metro": 3.893, "agua_por_metro": 24.786,
            "espaciamiento_etapas_m": 58.333, "ratio_agua_arena": 6.367,
            "ratio_arena_importada": 0.0,
        },
        "operadora_normalizada": "ypf s.a.",
        "operadora_conocida": True,
        "avisos": [],
    }}}


class PedidoBarrido(BaseModel):
    """Qué curva de sensibilidad dibujar: un diseño, una variable y un tramo."""

    diseno: Diseno
    variable: str = Field(
        "longitud_rama_horizontal_m",
        description="Qué variable barrer: longitud_rama_horizontal_m o cantidad_fracturas",
        examples=["longitud_rama_horizontal_m"])
    desde: float | None = Field(
        None, description="Dónde empieza el barrido. Por defecto, el mínimo del entrenamiento")
    hasta: float | None = Field(
        None, description="Dónde termina. Por defecto, el máximo del entrenamiento")
    puntos: int = Field(41, ge=2, le=120, description="Cuántos puntos tiene la curva")
    mantener_intensidad: bool = Field(
        True,
        description=("Al mover la rama, escalar también etapas, arena y agua, de modo que el "
                     "diseño por metro no cambie. Sin esto, alargar el pozo lo diluye y la "
                     "curva baja: es correcto, pero responde otra pregunta. No aplica a las "
                     "demás variables."))

    @model_validator(mode="after")
    def _tramo_valido(self):
        if self.variable not in ("longitud_rama_horizontal_m", "cantidad_fracturas"):
            raise ValueError("variable tiene que ser longitud_rama_horizontal_m "
                             "o cantidad_fracturas")
        if self.desde is not None and self.hasta is not None and self.hasta <= self.desde:
            raise ValueError("hasta tiene que ser mayor que desde")
        return self


@app.get("/", tags=["info"])
def raiz():
    """Ficha del servicio."""
    return {
        "servicio": app.title,
        "version": app.version,
        "documentacion": "/docs",
        "endpoints": {"POST /predict": "estimar un pozo", "GET /health": "estado",
                      "GET /modelo": "ficha del modelo",
                      "GET /operadoras": "las operadoras que el modelo conoce",
                      "POST /barrido": "la curva de la estimacion a lo largo de una variable",
                      "GET /demo": "panel interactivo"},
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


RESPUESTAS_PREDICT = {
    422: {
        "description": "El diseño no pasa las validaciones",
        "content": {"application/json": {"example": {"detail": [{
            "type": "greater_than_equal",
            "loc": ["body", "longitud_rama_horizontal_m"],
            "msg": "Input should be greater than or equal to 500",
            "input": 120,
        }]}}},
    },
    503: {
        "description": "El modelo no está entrenado",
        "content": {"application/json": {"example": {
            "detail": "No se encontro models/modelo.joblib. Ejecuta: python src/entrenar.py"}}},
    },
}


@app.post("/predict", response_model=Respuesta, tags=["prediccion"],
          response_description=("La estimación a doce meses, con su rango, las variables de "
                                "intensidad que calculó el servicio y los avisos"),
          responses=RESPUESTAS_PREDICT)
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


@app.post("/barrido", tags=["prediccion"])
def barrido(pedido: PedidoBarrido):
    """La curva de la estimación a lo largo de una variable, con su banda y sus avisos.

    Es lo que alimenta el panel de `/demo`. Las filas se arman y se pasan al modelo en un
    solo lote, así que una curva de 41 puntos cuesta más o menos lo mismo que una sola
    estimación pedida de a una, y el deslizador se puede mover en vivo.

    Cada punto trae `nivel_aviso`, que permite marcar en el gráfico el tramo donde el
    modelo se queda sin datos.
    """
    try:
        return model.barrido(pedido.diseno.model_dump(), variable=pedido.variable,
                             desde=pedido.desde, hasta=pedido.hasta, puntos=pedido.puntos,
                             mantener_intensidad=pedido.mantener_intensidad)
    except model.ModeloNoEntrenado as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo barrer: {e}")


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def documentacion():
    """La documentación interactiva de Swagger, con la paleta del proyecto.

    FastAPI la genera sola a partir del esquema OpenAPI; acá sólo se le inyectan los
    estilos y una cinta de encabezado, de modo que no hay nada escrito a mano que pueda
    quedar desactualizado respecto de los endpoints.
    """
    pagina = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} · documentación",
        swagger_ui_parameters=app.swagger_ui_parameters,
    ).body.decode()
    try:
        estilo = RUTA_ESTILO_DOCS.read_text(encoding="utf-8")
    except FileNotFoundError:
        estilo = ""                      # sin el archivo, Swagger sin vestir sigue sirviendo
    pagina = pagina.replace("</head>", estilo + "</head>", 1)
    pagina = pagina.replace('<div id="swagger-ui">', CINTA_DOCS + '<div id="swagger-ui">', 1)
    return HTMLResponse(pagina)


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    """Un panel para mover el diseño y ver cómo responde el modelo.

    Es una página suelta servida por esta misma aplicación: no agrega dependencias ni un
    segundo despliegue que mantener. Todo lo que muestra sale de `/barrido`.
    """
    try:
        return HTMLResponse(RUTA_DEMO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No se encontro src/demo.html")
