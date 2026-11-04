# Diseño de completación y productividad de pozos no convencionales en Vaca Muerta

Proyecto final del curso de Ciencia de Datos — **EnergIA Digital 2026**.

**Integrantes:** Carolina Bailón · Martín Gerbaldo · Franco Malacalza

Trabajo grupal. Las decisiones de alcance, criterio y metodología las discutimos y acordamos
entre los tres; nos apoyamos en herramientas de IA como copiloto, y lo documentamos al final
de este README.

**El modelo está desplegado y se puede probar ahora mismo:**
**<https://entregafinal-data-science.onrender.com/docs>** — se le pasa el diseño de un pozo
(metros de rama, etapas, arena, agua) y devuelve la producción estimada a doce meses. Corre en
una capa gratuita que duerme el servicio tras quince minutos sin uso, así que la primera
petición puede tardar hasta un minuto; las siguientes son inmediatas. El detalle está en
[La API de predicción](#la-api-de-predicción).

## El problema

Un pozo no convencional no produce por sí solo: la roca de Vaca Muerta tiene el petróleo
atrapado adentro y hay que agrietarla. Para eso se perfora un tramo horizontal de varios
kilómetros y se bombea agua con arena a alta presión, en decenas de etapas sucesivas. La arena
queda trabando las fracturas para que no se cierren.

Ese conjunto de decisiones —cuán largo es el tramo horizontal, cuántas etapas, cuánta arena y
cuánta agua por etapa— es el **diseño de completación**, y concentra buena parte del costo de
terminar el pozo. La operadora lo define antes de saber cuánto va a producir: una vez bombeada,
la arena no se recupera. Sobreinvertir es plata enterrada; subinvertir deja petróleo en el
subsuelo.

## Objetivo

Estimar **cuánto petróleo va a producir un pozo en su primer año a partir de las decisiones de
diseño que se toman antes de fracturarlo**, y entender cómo cambió ese diseño en la última década.

Es un problema de **aprendizaje supervisado**, y lo abordamos de las dos formas. Como **regresión**,
prediciendo los metros cúbicos de petróleo acumulados en los primeros 12 meses. Y como
**clasificación**, prediciendo si el pozo va a superar la producción típica. No son la misma pregunta
con otra cara: se miden distinto y, como mostramos, no dan el mismo resultado.

Sobre el final damos vuelta la pregunta y sacamos la variable objetivo de la ecuación: con
**aprendizaje no supervisado** preguntamos si los pozos se agrupan solos en tipos de diseño, sin
decirle al algoritmo cuánto produjo ninguno.

## Datos

Tres datasets abiertos de la Secretaría de Energía, todos incluidos en el repositorio:

| Fuente | Contenido | Período | Registros |
|---|---|---|---|
| **Datos de fractura de pozos (Adjunto IV)** | Una fila por operación de fractura: longitud de rama, etapas, arena nacional e importada, agua, presión máxima, potencia de equipos, operadora y formación | 2009 → 2026 | 4.890 |
| **Producción de pozos de gas y petróleo** | Una fila por pozo y mes: producción de petróleo y gas | 2006 → 2026 | 136.380 |
| **Listado de pozos** | Coordenadas, cota y profundidad de cada pozo | — | 2.282 |

- https://datos.energia.gob.ar/dataset/datos-de-fractura-de-pozos-adjunto-iv
- https://datos.energia.gob.ar/dataset/produccion-de-pozos-de-gas-y-petroleo-por-pozo-y-mes
- https://datos.energia.gob.ar/dataset/listado-de-pozos-cargados-por-empresas-operadoras

El dataset de producción original pesa **1,4 GB** repartido en archivos anuales que cubren todos los
pozos del país. Lo que está en el repositorio es un recorte propio a los pozos de nuestro alcance y
a las cinco columnas que usamos: **1,3 MB** con la misma información útil. El código del recorte
está documentado en el anexo del notebook de modelado.

## Alcance elegido

**Pozos horizontales con rama de al menos 500 m.** El dataset original mezcla dos poblaciones que
no son comparables: 2.054 de los 4.890 registros (42%) son pozos verticales convencionales, con una
escala de operación completamente distinta. Además, en un pozo vertical las variables de intensidad
por metro serían indefinidas, porque habría que dividir por cero.

Tras la limpieza nos quedamos con **2.603 pozos**: 2.590 en Cuenca Neuquina y 2.517 con Vaca Muerta
como formación productiva.

## Estructura del repositorio

```
entrega-final-data-science/
├── README.md
├── requirements.txt
├── data/
│   ├── fractura_adjunto_iv.csv       # dataset original del portal, sin modificar
│   ├── produccion_filtrada.csv.gz    # producción mensual, recortada a nuestro alcance
│   └── pozos_ubicacion.csv           # coordenadas y cota de los pozos
├── docs/
│   ├── DESPLIEGUE.md                 # cómo poner la API en un servidor público
│   ├── space_README.md               # README listo para Hugging Face Spaces
│   ├── guia_de_defensa.pdf           # el proyecto explicado sin jerga, para la presentación
│   ├── presentacion.pdf              # 17 diapositivas con las que se presenta el trabajo
│   └── presentacion.pptx             # las mismas diapositivas, con las notas del orador
├── Dockerfile                        # para desplegar la API
├── requirements-api.txt              # dependencias del servicio, sin las del análisis
├── ejemplo_pozo.json                 # cuerpo de ejemplo para POST /predict
├── models/
│   ├── modelo.joblib                 # el Random Forest entrenado que sirve la API
│   └── metadata.json                 # ficha del modelo: fecha, métricas, versiones
├── src/
│   ├── utils.py                      # funciones compartidas por los cuatro notebooks
│   ├── caracteristicas.py            # variables de intensidad: entrenamiento y API
│   ├── entrenar.py                   # entrena y guarda el modelo
│   ├── model.py                      # carga el modelo y predice
│   └── app.py                        # la API FastAPI
└── notebooks/
    ├── 01_eda_preentrega2.ipynb         # análisis exploratorio y limpieza
    ├── 02_modelado_preentrega3.ipynb    # regresión: cuántos m³ produce un pozo
    ├── 03_clasificacion_preentrega3.ipynb  # clasificación: ¿va a ser uno de los buenos?
    └── 04_clustering_preentrega4.ipynb   # clustering: ¿hay tipos de diseño?
```

Al ejecutar el primer notebook se crea `data/processed/pozos_analitico.csv` con el dataset limpio y
enriquecido. No lo versionamos porque se regenera solo.

### Qué vive en `src/utils.py` y qué no

En el módulo está lo que se repetía entre notebooks y no aporta nada a la lectura del análisis: la
**carga** de los archivos con su respaldo por URL, la **normalización de razones sociales** de las
operadoras y la agrupación de categorías poco frecuentes, las **funciones de métricas** de regresión
y de clasificación, el formateador de tablas y el cálculo de VIF.

Lo que deliberadamente **no** está ahí es todo lo que constituye una decisión del análisis: la
construcción de la variable objetivo, los dos guardas de truncamiento, la partición temporal y la
definición de los modelos. Eso tiene que leerse en el notebook donde se justifica, no escondido en
un import.

Los notebooks siguen siendo autocontenidos: si se abren en Colab, donde el repositorio no está
clonado, bajan `src/utils.py` del repositorio igual que hacen con los datos.

## Cómo reproducir

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_preentrega2.ipynb
jupyter notebook notebooks/02_modelado_preentrega3.ipynb
jupyter notebook notebooks/03_clasificacion_preentrega3.ipynb
jupyter notebook notebooks/04_clustering_preentrega4.ipynb
```

Los cuatro notebooks corren de punta a punta con los archivos que ya están en el repositorio: no hace
falta descargar nada. Del segundo en adelante son autocontenidos —vuelven a aplicar el pipeline de
limpieza del primero— así que pueden correrse solos. Los cuatro están ejecutados, así que los
resultados y los gráficos se ven directamente en GitHub.

**Las versiones están fijadas con `==`**, y ejecutado así el resultado es idéntico al que está
guardado: los mismos números y los mismos gráficos, hasta el píxel. Hacen falta las dos cosas para
que eso valga. La semilla (`SEMILLA = 42`) fija los resultados numéricos, pero no el renderizado:
entre versiones de `matplotlib` cambia el antialiasing y hasta el ancho en píxeles de una figura.
Y ninguna fecha sale del reloj de la máquina —la referencia para descartar fechas imposibles es la
fecha de publicación del propio archivo—, así que correrlo mañana da lo mismo que correrlo hoy.

Está verificado con **Python 3.13.16**. Si tu versión de Python no tiene ruedas para alguno de los
paquetes, cambiá el `==` por `>=` en esa línea: el análisis no cambia, solo se re-renderizan los
gráficos.

## Qué hace cada notebook

### `01_eda_preentrega2.ipynb` — exploración y limpieza

Sigue la estructura de la guía de la cursada, en tres partes:

| Parte | Puntos | Contenido |
|---|---|---|
| **1 — Análisis exploratorio** | 1 a 12 | Conocer el dataset, identificar las variables, completitud, duplicados, categóricas, unicidad de escritura, distribuciones, atípicos y su comportamiento, relación entre variables, comparación de grupos y variable objetivo |
| **2 — Transformación y limpieza** | 13 a 17 | Conversión de fechas, normalización de categorías, faltantes, validación de errores y definición del alcance, consolidación de cargas parciales |
| **3 — Feature engineering** | 18 | Creación de las variables de intensidad y control de plausibilidad |

En la Parte 1 exploramos el dataset completo sin modificarlo. El filtro a pozos horizontales es una
decisión que tomamos y documentamos en la Parte 2, no un supuesto de partida.

### `02_modelado_preentrega3.ipynb` — modelos, evaluación y métricas

| Parte | Puntos | Contenido |
|---|---|---|
| **1** | 1 a 5 | Cruce con la producción, construcción de la variable objetivo con guardas de truncamiento, caracterización del objetivo y separación de los pozos de gas |
| **2** | 6 y 7 | Selección de variables: varianza nula, disponibilidad al momento de decidir, multicolinealidad medida con VIF |
| **3** | 8 a 11 | Partición temporal, preprocesamiento dentro del pipeline, métricas y modelo de referencia, encoding de la variable categórica |
| **4** | 12 a 14 | Regresión lineal, Ridge, Random Forest y Gradient Boosting; comparación |
| **5** | 15 a 19 | Validación cruzada con `TimeSeriesSplit`, optimización de hiperparámetros con `GridSearchCV`, sobreajuste, partición temporal contra aleatoria, validación expansiva y variantes descartadas |
| **6** | 20 a 22 | Importancia por permutación, coeficientes, diagnóstico de error por quintil y conclusiones |

### `03_clasificacion_preentrega3.ipynb` — clasificación

La misma pregunta planteada como categorías: **¿va a ser este pozo uno de los buenos?** Mismo dataset,
mismas predictoras y misma partición temporal que el notebook 02, para que las dos mitades sean
comparables. Lo único que cambia es la variable objetivo y las métricas.

| Parte | Puntos | Contenido |
|---|---|---|
| **1** | 1 a 3 | Reconstrucción del dataset y definición del objetivo binario con el umbral en la mediana de entrenamiento |
| **2** | 4 | Métricas de clasificación y modelo de referencia |
| **3** | 5 | Regresión logística y Random Forest; comparación |
| **4** | 6 y 7 | Validación cruzada con `TimeSeriesSplit` y optimización con `GridSearchCV` |
| **5** | 8 y 9 | Matriz de confusión, curva ROC e importancia por permutación |
| **6** | 10 y 11 | Clasificar contra regresionar, y conclusiones |

### `04_clustering_preentrega4.ipynb` — clustering

La pregunta sin variable objetivo: **¿existen tipos de diseño de completación, o cada pozo se diseña
distinto?** Como el clustering no necesita saber cuánto produjo cada pozo, trabaja sobre los **2.603
pozos del alcance** en lugar de los 1.917 con producción medida: 686 pozos más.

| Parte | Puntos | Contenido |
|---|---|---|
| **1** | 1 a 3 | Reconstrucción del dataset y elección de las variables de intensidad |
| **2** | 4 a 6 | Estandarización, diagnóstico de extremos y acotamiento |
| **3** | 7 a 9 | Inercia, silueta, Calinski-Harabasz, Davies-Bouldin y estabilidad entre semillas |
| **4** | 10 y 11 | Las cuatro familias, su perfil, PCA y su evolución en el tiempo |
| **5** | 12 y 13 | Reducción de dimensionalidad con SVD y supervivencia de las familias al comprimir |
| **6** | 14 a 16 | Ward, GMM y DBSCAN; validación externa contra la producción; conclusiones |

## Arquitectura del modelo

Todo el preprocesamiento vive **dentro** de un `Pipeline` de scikit-learn. No es una
preferencia de estilo: es lo que garantiza que el imputador y el codificador se ajusten
únicamente con los datos de entrenamiento. Calcularlos sobre el conjunto completo filtraría
información del test al modelo, y el resultado dejaría de significar lo que dice.

```
Pipeline
├── ColumnTransformer
│   ├── num  → SimpleImputer(strategy="median")        sobre 9 variables numéricas
│   └── cat  → OneHotEncoder(handle_unknown="ignore")  sobre `operadora`
└── RandomForestRegressor(n_estimators=600, min_samples_leaf=8,
                          max_features=0.4, random_state=42)
```

| | |
|---|---|
| **Entrada** | 9 numéricas + `operadora` → **19 columnas** después del one-hot |
| **Salida** | m³ de petróleo acumulados en los primeros 12 meses |
| **Imputación** | mediana, y solo hace falta en `duracion_fractura_dias` (8 casos en train) |
| **Categórica** | `operadora`, agrupando bajo «otras» a las de menos de 40 pozos: 10 categorías |
| **`handle_unknown="ignore"`** | una operadora que el modelo no vio se codifica como todo ceros en lugar de romper |

Los modelos lineales usan el mismo esquema con dos cambios: `StandardScaler` después de
imputar, porque la escala sí les importa, y `drop="first"` en el codificador para evitar la
colinealidad perfecta entre las indicadoras.

**El mismo objeto `Pipeline` es el que sirve la API.** Se serializa entero con `joblib`, así
que el preprocesamiento viaja con el modelo y la predicción en producción recorre exactamente
el mismo camino que en el notebook.

## Resultados del modelado

**Modelo elegido: Random Forest.** Evaluado sobre los 619 pozos fracturados en 2024-2025, con
entrenamiento exclusivo en pozos anteriores:

| Modelo | MAE (m³) | RMSE (m³) | R² | Mejora de MAE vs. referencia |
|---|---|---|---|---|
| Referencia (mediana de train) | 14.138 | 18.782 | −0,18 | — |
| Regresión lineal | 10.776 | 15.019 | 0,244 | 24% |
| Ridge | 10.814 | 15.059 | 0,240 | 23% |
| **Random Forest** | **10.596** | **14.829** | **0,263** | **25%** |
| Gradient Boosting | 10.878 | 14.965 | 0,249 | 23% |

### El hallazgo central: la partición importa más que el modelo

El mismo modelo, con las mismas variables y los mismos hiperparámetros, evaluado con una **partición
aleatoria** da **R² 0,52** (promedio de 10 semillas) en lugar de 0,263. Exactamente el doble, y el
factor se repite en los tres modelos.

La diferencia es que la partición aleatoria le deja ver al modelo pozos del mismo período que
después usa para evaluarlo. Como el diseño de completación en Vaca Muerta cambió muchísimo —la rama
mediana pasó de 1.411 m en 2016 a 2.996 m en 2025— eso no refleja cómo se va a usar el modelo: en
la práctica se entrena con lo que ya produjo y se aplica a pozos que todavía no se fracturaron.

**La métrica no mide solo al modelo: mide al esquema de validación.**

### Validación cruzada: el mismo hallazgo sin tocar el test

El resultado anterior compara la partición temporal contra la aleatoria usando el conjunto de test.
Se puede comprobar lo mismo **sin que el test participe**, con validación cruzada sobre los 1.298
pozos de entrenamiento:

| Esquema de validación cruzada | R² medio | Desvío entre folds | Rango |
|---|---|---|---|
| `TimeSeriesSplit(5)` — temporal | **0,171** | 0,116 | 0,017 – 0,335 |
| `KFold(5, shuffle=True)` — aleatoria | **0,546** | 0,044 | 0,473 – 0,610 |

La validación aleatoria infla el resultado **3,2 veces**, y los pozos de 2024-2025 no intervinieron
en la medición. El desvío entre folds temporales es casi el triple que el de los aleatorios: cuando
el R² depende tanto de qué año toque predecir, reportar un número sin su dispersión da una falsa
sensación de precisión.

### Optimización de hiperparámetros: no había nada que ganar

Sobre esa validación temporal corrimos la búsqueda de hiperparámetros del Random Forest:

| | MAE de validación cruzada | MAE en test | R² en test |
|---|---|---|---|
| Hiperparámetros elegidos a mano | 11.765 | **10.596** | **0,263** |
| `GridSearchCV` — 120 combinaciones | **11.432** | 10.659 | 0,256 |
| `RandomizedSearchCV` — 60 sorteos | 11.517 | 10.554 | 0,255 |

**La optimización mejora la validación cruzada un 2,9% y no mejora el test.** Para saber si esas
diferencias significan algo, entrenamos el mismo modelo cambiando solo la semilla del bosque: el R²
de test se mueve con un desvío de **0,003**, en un rango de 0,252 a 0,264. El R² del modelo
optimizado —0,256— **cae dentro de ese rango**, a 1,9 desvíos del modelo a mano. Y reentrenar ese
mismo modelo optimizado con las filas ordenadas por año en lugar del orden original lo mueve otros
0,004, sin que cambie ningún hiperparámetro. La optimización movió la métrica en el mismo orden en
que la mueven detalles sin significado.

`RandomizedSearchCV` lo deja más claro todavía: es el **mejor MAE de los tres** (42 m³ por debajo del
modelo a mano) y a la vez el **peor R²**. Cuando el ordenamiento se da vuelta según qué métrica se
mire, ese ordenamiento no es información.

El óptimo es una meseta y no un pico: las cinco mejores combinaciones de la grilla están dentro de
40 m³ entre sí, con un desvío entre folds de unos 1.000 m³. Y el mejor de la grilla apunta a
`min_samples_leaf=2`, o sea hojas de dos pozos: más capacidad de memorizar, en un problema que ya
tiene una brecha train-test de 0,45. **Mantenemos los hiperparámetros elegidos a mano.** Lo que
aporta la búsqueda no es un modelo mejor, es la verificación de que no había uno mejor por encontrar.

### Encoding de la variable categórica

`operadora` es la única categórica del modelo: 10 categorías tras agrupar en `"otras"` las
operadoras con menos de 40 pozos. Comparados los esquemas sobre la misma partición temporal:

| Esquema | R² (Random Forest) | R² (Ridge) |
|---|---|---|
| Sin la categórica | 0,123 | 0,062 |
| **One-hot — el elegido** | **0,263** | **0,240** |
| Ordinal (códigos enteros) | 0,221 | **0,042** |
| Target encoding (CV interna) | 0,260 | 0,240 |

El encoding ordinal le cuesta poco al árbol, que puede partir la variable varias veces y reconstruir
los grupos, y **le rompe el modelo al lineal** (0,240 → 0,042), que queda obligado a tratar códigos
arbitrarios como si fueran una escala. El target encoding empata con one-hot usando una columna en
lugar de diez: con 10 categorías no vale el riesgo de filtración, pero es el camino para `yacimiento`
y sus 63 categorías.

Mirando la variable apareció además un defecto del dato: **Vista figura como dos operadoras
distintas** (`vista oil & gas argentina sau` y `vista energy argentina sau`) porque cambió de razón
social, y el corte es temporal —62 pozos en train y 0 en test la primera, 33 y 88 la segunda—, así
que esas dos columnas del one-hot funcionan como un marcador de época más que como la identidad de
la operadora. Unificarlas no cambia el resultado (R² 0,263 → 0,262), así que lo dejamos documentado
sin adoptarlo.

## Resultados de la clasificación

Misma partición temporal, con el umbral en la **mediana de entrenamiento (28.853 m³)**. El umbral se
calcula solo con train para no filtrar información del test.

| Modelo | Exactitud | Precisión | Sensibilidad | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Referencia (clase mayoritaria de train) | 0,313 | 0,000 | 0,000 | 0,000 | 0,500 |
| Regresión logística | 0,735 | 0,747 | 0,929 | 0,828 | 0,742 |
| **Random Forest** | **0,750** | **0,773** | 0,899 | **0,831** | **0,744** |

Dos aclaraciones para leer la tabla. La referencia saca 0,313 porque entrenamiento está empatado
649/649 y, ante el empate, predice "bajo" para todos: en test el 68,7% son altos. **La vara exigente
es ese 68,7%**, que es lo que se obtiene prediciendo "alto" siempre. El modelo le gana por 6 puntos.

Y ese 68,7% no es casualidad: el umbral calibrado con el pasado deja el test **69/31** porque los pozos
de 2024-2025 tienen la mediana en 35.681 m³ contra 28.853. Es la misma deriva temporal del notebook 02,
vista desde la clasificación.

### El modelo detecta buenos, no descarta malos

| Clase | Precisión | Sensibilidad | Pozos |
|---|---|---|---|
| bajo | 0,66 | **0,42** | 194 |
| alto | 0,77 | **0,90** | 425 |

Comete **112 falsos altos contra 43 falsos bajos**. Sirve para no perderse un pozo bueno; no sirve para
confiar en que los que marca como malos lo son. Es el mismo sesgo de sobreestimación del regresor,
con otra forma.

### El hallazgo: conviene regresionar y cortar, no clasificar

| | Exactitud | F1 | ROC-AUC |
|---|---|---|---|
| Clasificador entrenado para la tarea | 0,750 | 0,831 | 0,744 |
| **Regresor del notebook 02, con su predicción cortada en el umbral** | **0,763** | **0,844** | **0,791** |

**Tomar el modelo de regresión y comparar su predicción contra el umbral funciona mejor que entrenar un
clasificador**, y la diferencia de AUC —0,047— es veinte veces la que separa a los dos clasificadores
entre sí, y mayor que todo lo que mueve optimizar hiperparámetros.

La razón es que **binarizar el objetivo tira información**: para el clasificador, un pozo de 28.000 m³
y uno de 5.000 son los dos "bajo" y aprende lo mismo de ambos. El regresor sabe que uno está al borde
y el otro es un fracaso, y ese orden más fino se conserva al cortar.

No es una regla general: vale cuando la categoría se construyó cortando una continua que se puede
medir. Si las clases fueran genuinamente categóricas, no habría ningún número que regresionar.

## Resultados del clustering

**Cuatro familias de diseño**, encontradas usando solo las cinco variables de intensidad. Las columnas
de rama, etapas y año **no entraron al ajuste**: describen lo que el algoritmo encontró.

| Familia | Pozos | Espaciamiento | Arena/m | Agua/m | Arena imp. | Rama | Etapas | Año (mediana) |
|---|---|---|---|---|---|---|---|---|
| **Intensivo moderno** | 1.476 | 60 m | 3,91 t | 24,8 m³ | 0% | 2.800 m | 48 | 2024 |
| **Intermedio** | 911 | 76 m | 3,01 t | 18,4 m³ | 0% | 2.256 m | 29 | 2020 |
| **Arena importada** | 132 | 78 m | 2,82 t | 13,6 m³ | **90%** | 1.244 m | 15 | 2016 |
| **Baja intensidad** | 84 | 133 m | 0,93 t | 2,6 m³ | 0% | 1.017 m | 7 | 2017 |

El algoritmo nunca vio el año de fractura y sin embargo las medianas salen ordenadas: encontró la
**evolución histórica del diseño** que habíamos descrito a mano en el EDA.

### La validación externa: las familias producen distinto

La producción quedó fuera del ajuste, así que compararla entre familias es una prueba honesta:

| Familia | Pozos con producción | Mediana (m³) |
|---|---|---|
| Intensivo moderno | 1.078 | **36.760** |
| Intermedio | 716 | 26.422 |
| Arena importada | 92 | 15.638 |
| Baja intensidad | 31 | **3.311** |

Kruskal-Wallis H = 264,4 (p ≈ 5·10⁻⁵⁷). Un factor de once entre la primera y la última, **en el mismo
orden que la intensidad del diseño**. No prueba causalidad: las familias están confundidas con la
época.

### Una variable inútil para predecir, decisiva para describir

`ratio_arena_importada` es la variable que los modelos supervisados descartaron: +79 m³ en Ridge e
importancia **0,0000 exacta** en el clasificador. En el clustering **define una familia entera**. No
hay contradicción: predecir la producción y describir el diseño son preguntas distintas.

### Lo que costó llegar

- **Sin acotar los extremos, K-Means dedica un grupo entero a 3 pozos** y la silueta lo aplaude con
  0,62. Acotando al 1%–99% cae a 0,40, con un grupo más chico de 134 pozos en lugar de 3.
- **Los índices no deciden:** silueta y Calinski-Harabasz votan k=2, Davies-Bouldin vota k=6, la
  inercia no tiene codo. Elegimos k=4 por estabilidad, tamaño de grupo e interpretabilidad, y lo
  dejamos argumentado.
- **`n_init` no es decorativo:** con 10 arranques, k=4 daba agrupamientos distintos según la semilla
  (ARI 0,26); con 200, siempre el mismo (ARI 0,99).
- **Varianza explicada no es importancia.** Al comprimir con SVD, el agrupamiento sobrevive con tres
  componentes (ARI 0,958) y se rompe con dos (0,374). La componente que se pierde aporta apenas el
  17% de la varianza y es casi exclusivamente `ratio_arena_importada`: sin ella, esa familia se
  disuelve. Descartar componentes por su peso habría borrado el grupo más distintivo.

## Qué cambiaría una operadora: la lectura no técnica

El modelo no decide por nadie, pero ordena las decisiones de diseño por cuánto mueven la producción
del primer año. Leído sin jerga, dice tres cosas.

**1. Alargar la rama es la palanca más grande, y por lejos.** Por cada ~900 m adicionales de tramo
horizontal, el modelo asocia **unos 7.400 m³ más** de petróleo en los primeros doce meses, con el
resto del diseño constante: casi el triple que la segunda variable de la lista. Es también el camino
por el que la industria subió la productividad en la última década, y el EDA lo confirma: la rama
mediana se triplicó en diez años.

**2. Apretar las etapas suma; bombear más arena por metro, ya casi no.** Fracturar más denso —menos
metros entre etapas— aparece con signo positivo en los dos modelos. La arena por metro, en cambio,
tiene un efecto chico y con rendimientos decrecientes: el modelo lineal casi no la usa y el de
árboles la aprovecha solo en los tramos bajos. Coincide con lo que muestra el EDA, donde la
intensidad de arena dejó de crecer en 2019. Si hay un peso más para poner en el diseño, estos datos
sugieren que rinde más en metros de rama que en toneladas de arena por metro.

**3. Hay decisiones que no mueven la aguja.** La proporción de arena importada sobre nacional es
indistinguible de cero en los dos modelos. Es un resultado útil justamente por negativo: es una
decisión de compra que tiene costo y que, según estos datos, no se paga en producción.

Tres advertencias que van en la misma frase que los resultados:

- **El modelo explica el 26% de la variabilidad.** Sirve para comparar conjuntos de pozos y ordenar
  prioridades de diseño; no para dictaminar sobre un pozo individual. Y falla justo en los extremos:
  sobreestima a los peores pozos y subestima a los mejores.
- **La mayor parte de lo que no explica es geología**, que no está en los datos abiertos. Dos pozos
  con el mismo diseño en áreas distintas pueden producir muy distinto, y eso el modelo no lo ve.
- **El peso de la operadora no mide la habilidad de la empresa**: mide la calidad del área que le
  tocó. No se puede usar para rankear operadoras.
- **Son asociaciones, no causas.** Solo existen los pozos que se perforaron: no hay forma de saber
  qué habría producido uno de ellos con otro diseño.

Y una conclusión que no es sobre pozos sino sobre cómo se compra un modelo: **si hubiéramos validado
al azar, habríamos informado el doble de la precisión que el modelo tiene con pozos nuevos.** Un
modelo que se va a usar para diseñar los pozos del año que viene hay que evaluarlo con los pozos de
los años que todavía no vio. Es el hallazgo central del proyecto, y el que más fácil se traduce
fuera del ámbito técnico.

## Principales hallazgos del EDA

- **El dataset mezcla pozos verticales y horizontales.** Un vertical mediano bombea 0 tn de arena
  nacional y 430 m³ de agua en 3 etapas; uno horizontal, 8.576 tn y 54.314 m³ en 38 etapas. El cruce con
  `tipo_reservorio` lo confirma: los verticales son convencionales y los horizontales no.

- **El grano no es un pozo por fila.** Hay 4.890 filas para 4.646 pozos. Los 244 registros
  excedentes comparten pozo *y* fecha de fractura y reparten las etapas entre varias filas: son
  cargas parciales de una misma operación, no refracturas. Las consolidamos sumando las magnitudes
  aditivas y tomando el máximo de las propiedades del pozo.

- **Los ceros son el problema de calidad, no los nulos.** Solo tres columnas tienen nulos, pero
  hay ceros centinela en longitud de rama (el valor 3,0 m aparece 63 veces), presión, potencia y
  agua. Un equipo de fractura no opera a cero psi.

- **`co2_inyectado_m3` no aporta nada, pero no por lo obvio.** Tiene 159 registros con valor
  positivo, y los 159 son pozos verticales. Dentro de la población que analizamos es idénticamente
  cero. Si hubiéramos mirado solo el porcentaje de ceros global, la conclusión habría sido otra.

- **La intensidad de fractura se amesetó.** Entre 2016 y 2025 la arena por pozo se triplicó, pero
  eso es efecto de escala: la rama se duplicó. La arena *por metro* creció 29% hasta 2019 y
  apenas 9% en los seis años siguientes, mientras el espaciamiento entre etapas se apretaba de
  78 m a 59 m. La industria dejó de mejorar bombeando más arena por metro.

- **Cada operadora tiene una estrategia identificable, y se puede medir.** Vista usa la rama más
  larga (2.911 m), la mayor intensidad (4,17 tn/m) y el espaciamiento más apretado (58 m), y lo
  repite de forma muy estandarizada: su dispersión relativa de rama es 0,17, la más baja de las
  seis. Total Austral está en el extremo opuesto en todo —1.970 m, 2,48 tn/m, etapas a 83,6 m— con
  una dispersión de 0,42, dos veces y media la de Vista. La diferencia de intensidad entre las dos
  es de 1,7 veces, ya normalizada por metro de rama: es criterio de diseño, no escala.

- **Las variables de diseño están muy correlacionadas entre sí.** La longitud de rama correlaciona
  0,91 con la arena y 0,91 con las etapas, porque todas crecen con el tamaño del pozo. En el
  modelado conservamos una sola variable de escala más las de intensidad: con eso los VIF bajan de
  7,57 a menos de 5.

## Limitaciones y sesgos

- **Concentración en un operador.** YPF representa el 53% de los pozos, así que cualquier
  conclusión refleja sobre todo su forma de completar.
- **Concentración geológica.** El 97% de los pozos son de Vaca Muerta; las conclusiones no se
  extrapolan a otras formaciones no convencionales.
- **Diseño confundido con geología.** Cada operadora trabaja en áreas de calidad de roca distinta,
  así que el efecto del diseño está mezclado con el de la roca. Describimos asociaciones, no
  efectos causales.
- **No hay contrafactual.** El dataset solo contiene pozos efectivamente fracturados: no se puede
  saber qué habría producido un pozo con otro diseño.
- **Datos preliminares.** La Secretaría de Energía los publica como sujetos a revisión, y el año
  en curso está incompleto.
- **Falta la geología, que es el factor dominante.** Probamos agregar coordenadas, cota y área de
  concesión: el R² de test no se movió de 0,26 y el de entrenamiento subió de 0,71 a 0,81, es decir
  que solo agregan capacidad de memorizar. La posición en superficie no es la calidad de la roca a
  tres kilómetros de profundidad, y eso no está en datos abiertos. Es el techo del modelo.
- **La operadora pesa tanto como la mejor variable de diseño**, y no es causal: refleja la calidad
  del área que cada empresa tiene en concesión. No se puede usar para comparar operadoras.
- **El modelo es de petróleo.** Los 191 pozos cuyo objetivo es el gas quedan excluidos: medirlos con
  "petróleo acumulado a 12 meses" sería evaluarlos con la métrica equivocada.
- **Validez temporal acotada.** Está validado sobre pozos de 2024-2025. Si el diseño sigue
  cambiando, hay que reentrenar.

## La API de predicción

El modelo no se queda en el notebook: se sirve como una API HTTP, de modo que cualquier
sistema pueda pedirle una estimación para un diseño que todavía no se fracturó.

> 🟢 **Desplegada y en funcionamiento:** **<https://entregafinal-data-science.onrender.com/docs>**
>
> Es la documentación interactiva: se puede probar el endpoint desde el navegador, sin escribir
> código. Corre en la capa gratuita de Render, que **duerme el servicio tras quince minutos sin
> uso**: la primera petición puede tardar hasta un minuto en despertarlo. Las siguientes son
> inmediatas.

Para levantarla en local:

```bash
pip install -r requirements.txt
python src/entrenar.py          # entrena y guarda models/modelo.joblib
uvicorn src.app:app --reload    # levanta el servicio
```

Y en el navegador, **http://127.0.0.1:8000/docs**: la documentación interactiva que genera
FastAPI, donde se puede probar el endpoint sin escribir una línea de código.

### Los cuatro endpoints

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/` | Ficha del servicio |
| `GET` | `/health` | Estado. Devuelve 503 si el modelo no está entrenado |
| `GET` | `/modelo` | Cuándo se entrenó, con cuántos pozos, con qué métricas y qué operadoras conoce |
| `POST` | `/predict` | La estimación |

### Un ejemplo

```bash
curl -X POST https://entregafinal-data-science.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d @ejemplo_pozo.json
```

```json
{
  "prediccion_m3": 38506.0,
  "rango_p10_p90_m3": [26915.1, 49433.9],
  "variables_derivadas": {
    "arena_por_metro": 3.893,
    "agua_por_metro": 24.786,
    "espaciamiento_etapas_m": 58.333,
    "ratio_agua_arena": 6.367,
    "ratio_arena_importada": 0.0
  },
  "operadora_conocida": true
}
```

### Qué recibe y qué devuelve

La API pide **las decisiones de diseño tal como se declaran**, no las variables del modelo:
metros de rama, cantidad de etapas, toneladas de arena nacional e importada, metros cúbicos
de agua, presión, potencia, días de operación y operadora. Las **variables de intensidad las
calcula el servicio**, con las mismas fórmulas que se usaron para entrenar —viven en
`src/caracteristicas.py` y las importan tanto el entrenamiento como la API, para que no
puedan separarse— y las devuelve en la respuesta para que se vea qué entró al modelo.

Además de la estimación devuelve un **rango**: cada uno de los 600 árboles del bosque vota un
número, y los percentiles 10 y 90 de esos votos dan una idea del desacuerdo. No es un
intervalo de confianza estadístico, pero es más honesto que un número solo, sobre todo
porque el modelo comprime sus predicciones hacia el centro.

Y marca si la **operadora es conocida**. Una que el modelo no vio al entrenar no rompe nada
—el codificador la trata como todo ceros— pero la estimación pierde el aporte de esa
variable, y conviene saberlo.

### Lo que responde ante el diseño

Las dos palancas del modelo se ven pidiéndole estimaciones y cambiando una sola cosa:

| Alargar la rama (arena, agua y etapas en proporción) | Estimación |
|---|---|
| 2.000 m, 34 etapas | 29.224 m³ |
| 2.800 m, 48 etapas | 38.506 m³ |
| 3.600 m, 62 etapas | 49.903 m³ |

| Apretar las etapas (todo lo demás igual) | Estimación |
|---|---|
| 28 etapas, una cada 100 m | 33.365 m³ |
| 40 etapas, una cada 70 m | 36.871 m³ |
| 56 etapas, una cada 50 m | 38.067 m³ |

La primera tabla sube con fuerza y la segunda se aplana: es, en estimaciones concretas, el
mismo resultado que los coeficientes del punto 20.

### Cómo está armado

```
src/
├── caracteristicas.py   las variables de intensidad; las comparten entrenamiento y API
├── entrenar.py          reconstruye el dataset, mide, reentrena y guarda el artefacto
├── model.py             carga el modelo y predice; nada de HTTP
└── app.py               FastAPI; nada de modelo
models/
├── modelo.joblib        el Random Forest entrenado (3,7 MB)
└── metadata.json        ficha: fecha, pozos, métricas, operadoras, versiones
```

**`src/entrenar.py` hace dos cosas y la diferencia importa.** Primero entrena con los 1.298
pozos fracturados hasta 2023 y mide una sola vez sobre los 619 de 2024-2025: de ahí salen las
métricas que la API declara, y reproduce exactamente los números del notebook (MAE 10.596 m³,
R² 0,263). Después **reentrena con los 1.917 pozos** para servir, porque un modelo que va a
estimar pozos nuevos no tiene por qué ignorar los dos últimos años. `metadata.json` deja
constancia de las dos cosas.

### Despliegue

Con Docker:

```bash
docker build -t pozos-api .
docker run -p 8000:8000 pozos-api
```

En **Render** o **Hugging Face Spaces** alcanza con conectar el repositorio: el `Dockerfile`
ya define todo y `models/` está versionado, así que la plataforma no tiene que entrenar nada.

El servicio usa **`requirements-api.txt`** y no `requirements.txt`: no necesita jupyter ni las
librerías de gráficos, y sacarlas hace que la imagen entre holgada en cualquier capa gratuita.
Las versiones coinciden con las del análisis, y la de `scikit-learn` tiene que coincidir sí o
sí, porque es la que serializó el modelo.

📄 **[Guía de despliegue paso a paso](docs/DESPLIEGUE.md)** — Render, Hugging Face Spaces y
Docker, con qué revisar cuando algo falla.

## Uso responsable del modelo

La consigna pide una reflexión sobre sesgos e IA responsable, y acá va la nuestra, que es la
misma que atraviesa todo el proyecto: **el riesgo de este modelo no es que se equivoque, es
que se le crea más de lo que vale.**

**Qué puede hacer.** Ordenar decisiones de diseño por cuánto rinden, comparar carteras de
pozos y dar un orden de magnitud. Explica el 26% de la variabilidad.

**Qué no puede hacer, y no habría que dejarlo hacer:**

- **Decidir sobre un pozo individual.** El error se triplica en los extremos: sobreestima el
  quintil menos productivo en 13.053 m³ y subestima el más productivo en 11.461. Justo en los
  casos donde más plata hay en juego es donde peor anda.
- **Comparar operadoras.** La variable `operadora` pesa tanto como la mejor variable de
  diseño, pero mide **la calidad del área que cada empresa tiene en concesión**, no su
  habilidad. Usarla para rankear empresas sería leer geología como desempeño. Es el sesgo más
  peligroso del modelo porque es el más fácil de cometer.
- **Sostener una decisión causal.** Solo existen los pozos que se perforaron: no hay
  contrafactual. El modelo describe asociaciones.
- **Aplicarse fuera de su alcance.** Está entrenado con pozos horizontales de rama ≥ 500 m,
  en su enorme mayoría de Vaca Muerta y con el petróleo como objetivo. Un pozo vertical, de
  gas o de otra formación cae fuera, y la API no tiene cómo saberlo.

**Sesgos de los datos que el modelo hereda.** YPF representa el 53% de los pozos, así que el
modelo aprende sobre todo su forma de completar. El 97% son de Vaca Muerta. Y los datos son
los que las operadoras **declaran** al organismo: el modelo hereda cualquier sesgo sistemático
de esa declaración, que no podemos auditar.

**Qué hicimos al respecto.** El modelo se validó con pozos posteriores a los de entrenamiento,
que es el uso real; la API devuelve un rango y no solo un número; `metadata.json` deja a la
vista con qué se entrenó y cuándo; y las limitaciones están escritas en el README, en los
notebooks y en la descripción de la propia API, no en una nota al pie. Un modelo que se
presenta con sus límites adelante es más difícil de usar mal.

**Validez temporal.** Está validado sobre pozos de 2024-2025. El diseño de completación cambia
año a año —es el hallazgo central del proyecto— así que **esto se degrada con el tiempo**:
hay que reentrenar y volver a medir con la partición temporal, no asumir que sigue valiendo.

## Uso de herramientas de IA

Usamos un asistente de IA como **copiloto** durante las cuatro pre-entregas. Lo documentamos porque
la cursada lo pide y porque es parte honesta de cómo trabajamos.

### En qué nos ayudó y qué decidimos nosotros

| Nos apoyamos en la IA para | Lo resolvimos nosotros |
|---|---|
| Sintaxis y API de `pandas` y `scikit-learn` que no conocíamos (`ColumnTransformer`, `Pipeline`, `permutation_importance`) | La elección del tema, la pregunta de negocio y por qué vale la pena responderla |
| Proponer enfoques ante un problema planteado: cómo calcular el mes de vida de un pozo, cómo detectar el truncamiento de la ventana | La definición del alcance (pozos horizontales con rama ≥ 500 m) y su justificación técnica |
| Revisar nuestro código buscando errores y señalar casos borde que no habíamos previsto | La elección de la variable objetivo y de la ventana de 12 meses |
| Sugerir qué gráfico comunica mejor un hallazgo | La interpretación de los resultados y el criterio físico de qué tiene sentido en un pozo |
| Mejorar la redacción de las explicaciones para que se entiendan sin ser del rubro | Las decisiones metodológicas, incluida la de rechazar variantes que mejoraban la métrica |


### Cómo nos aseguramos de entender lo que entregamos

El criterio que seguimos fue que **ninguna afirmación entra si no la podemos sostener con la salida
del código**. Todos los números de los notebooks salen de celdas ejecutadas y visibles: si un
resultado del texto no coincidía con lo que imprimía la celda, corregíamos el texto, no el número.

Cuatro casos concretos donde revisar en lugar de aceptar nos cambió la conclusión:

- **La columna `co2_inyectado_m3`.** La primera lectura fue que era constante y no aportaba nada.
  Al mirar la salida, tenía 159 registros con valor positivo. Verificamos uno por uno: los 159 son
  pozos verticales, así que dentro de nuestro alcance es idénticamente cero. El hallazgo correcto
  es más fuerte que el inicial, y solo apareció por desconfiar del primero.

- **La transformación logarítmica de la variable objetivo.** La habíamos dado por necesaria al
  plantear el proyecto, porque es lo habitual en producción de hidrocarburos. Al medirla, la
  asimetría original era 0,57 y `log1p` la llevaba a −2,44: la empeoraba. Dejamos la verificación
  a la vista en el notebook en lugar de borrar el supuesto equivocado.

- **Agregar el año de fractura como variable.** Mejoraba el R² de 0,263 a 0,285. En vez de quedarnos
  con la mejora, la probamos en los otros cortes temporales: ahí cambiaba de signo. La descartamos
  y explicamos por qué, aunque el número fuera peor.

- **La silueta de 0,62 en el clustering.** El primer agrupamiento daba un valor altísimo para datos
  reales. Al mirar qué estaba premiando, el algoritmo había dedicado un grupo entero a **tres pozos**
  fracturados en una sola etapa. Acotando los extremos la silueta cayó a 0,40 y los grupos pasaron a
  tener cientos de pozos: nos quedamos con el número peor.
  

### Revisión entre nosotros

Cada entrega la revisó otro integrante del grupo antes de darla por cerrada, y de ahí salieron
cambios concretos: reemplazar el recorte de atípicos por imputación con la mediana del grupo —porque
el *winsorizing* no era una técnica vista en la cursada— y reescribir una sección entera de
comparación entre operadoras que no se entendía como estaba redactada.

Esa decisión volvió a discutirse en la cuarta pre-entrega y la revisamos: ahí **sí** acotamos los
valores extremos, porque el problema es otro. En los modelos supervisados los árboles son insensibles
a un valor extremo, así que recortarlo solo perdía información. En un algoritmo basado en distancias
un único pozo extremo deforma los grupos de todos los demás, y el punto 5 de ese notebook muestra
exactamente cómo. Cuando la técnica cambia, el criterio se vuelve a justificar en lugar de
arrastrarse.
