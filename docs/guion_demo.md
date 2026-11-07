# Guion de la demostración en vivo

Para la diapositiva **«El modelo, funcionando»**. Está pensado para imprimirse y tenerlo
al lado: todo lo que hay que pegar está acá, no hay que calcular nada delante de la clase.

**El panel:** <https://entregafinal-data-science.onrender.com/demo>
**La API:** <https://entregafinal-data-science.onrender.com/docs>

---

## Dos caminos, y cuál conviene

**Para mostrar en clase, el panel.** Se mueve un deslizador y el número cambia delante de
todos, con la curva entera a la vista. No hay que pegar nada ni leer JSON, y la zona sombreada
del gráfico cuenta sola el límite del modelo. Son los **pasos 1 y 2 en versión corta**:

1. Abrir **`/demo`**. Arranca en el pozo típico: **38.506 m³**.
2. Apretar **«Pozo corto»** (29.224) y **«Pozo largo»** (49.903). Tres clics, misma historia.
3. Arrastrar el deslizador de **longitud de rama** hacia la derecha. La curva sube, y al pasar
   los **4.965 metros** entra en la franja marcada: el número deja de moverse y aparecen los
   avisos. Ahí se dice la frase que importa: *«el pozo más largo del entrenamiento mide 4.965
   metros; de acá en más el modelo devuelve siempre lo mismo, porque un bosque no extrapola»*.
4. Apretar **«Etapas»** en *Qué barrer*: la curva sube y se aplana. La segunda palanca.

**Para mostrar que es una API de verdad, `/docs`.** Si el profesor pregunta por la
implementación, o si querés mostrar la respuesta cruda, están los pasos completos más abajo.
Sirven igual como respaldo: **si el panel no carga, la demostración sigue en `/docs`**.

---

## Antes de entrar al aula

- [ ] **Diez minutos antes**, abrir **las dos** direcciones en sendas pestañas y dejarlas abiertas. La capa
      gratuita de Render duerme el servicio tras quince minutos sin uso y la primera petición
      puede tardar casi un minuto. En vivo eso se ve como si nada funcionara.
- [ ] Tener **este guion abierto o impreso**, para copiar y pegar sin pensar.
- [ ] Comprobar que la pestaña responde: desplegar `GET /health` → *Try it out* → *Execute*.
      Tiene que devolver `{"estado":"ok","modelo_cargado":true}`.

---

## Paso 1 — El pozo base

Abrir **`POST /predict`** → **Try it out** → reemplazar el cuerpo por esto → **Execute**:

```json
{
  "longitud_rama_horizontal_m": 2800,
  "cantidad_fracturas": 48,
  "arena_bombeada_nacional_tn": 10900,
  "arena_bombeada_importada_tn": 0,
  "agua_inyectada_m3": 69400,
  "presion_maxima_psi": 11000,
  "potencia_equipos_fractura_hp": 25000,
  "duracion_fractura_dias": 40,
  "operadora": "ypf s.a."
}
```

**Responde 38.506 m³**, con rango 26.915 – 49.434.

Señalar **tres** cosas de la respuesta, en este orden:

1. **`prediccion_m3`** — la estimación de producción de petróleo en los primeros doce meses.
2. **`rango_p10_p90_m3`** — *«no damos una cifra sola: los 600 árboles del bosque votan y esto
   es el rango donde cae el 80% de los votos. Un pozo individual puede caer en cualquier punto
   de esa banda.»*
3. **`variables_derivadas`** — `arena_por_metro: 3.89`, `espaciamiento_etapas_m: 58.3`.
   *«El servicio no recibe estas variables: las calcula. Es el mismo código de feature
   engineering que usamos para entrenar, importado del mismo módulo. Por eso no hay forma de
   que el entrenamiento y la predicción se desincronicen.»*

---

## Paso 2 — El mismo diseño en un pozo más largo

**Ésta es la demostración que vale.** Son dos ejecuciones seguidas, pegando el cuerpo entero.

### 2a. Pozo corto — 2.000 metros

```json
{
  "longitud_rama_horizontal_m": 2000,
  "cantidad_fracturas": 34,
  "arena_bombeada_nacional_tn": 7786,
  "arena_bombeada_importada_tn": 0,
  "agua_inyectada_m3": 49571,
  "presion_maxima_psi": 11000,
  "potencia_equipos_fractura_hp": 25000,
  "duracion_fractura_dias": 40,
  "operadora": "ypf s.a."
}
```

**→ 29.224 m³**

### 2b. Pozo largo — 3.600 metros

```json
{
  "longitud_rama_horizontal_m": 3600,
  "cantidad_fracturas": 62,
  "arena_bombeada_nacional_tn": 14014,
  "arena_bombeada_importada_tn": 0,
  "agua_inyectada_m3": 89229,
  "presion_maxima_psi": 11000,
  "potencia_equipos_fractura_hp": 25000,
  "duracion_fractura_dias": 40,
  "operadora": "ypf s.a."
}
```

**→ 49.903 m³** — un **71% más**, veinte mil metros cúbicos de diferencia.

Qué decir: *«Los tres pozos tienen exactamente la misma intensidad de diseño: 3,89 toneladas
de arena por metro y una etapa cada 58 metros. Lo único que cambia es el largo. Por eso la
rama es la palanca más grande que tiene la operadora.»*

> ### ⚠️ Lo que no hay que hacer
>
> **No cambiar sólo el campo `longitud_rama_horizontal_m`.** Si se alarga la rama dejando la
> arena y el agua donde estaban, la estimación **baja** (de 34.971 a 28.333 m³), porque se
> está diluyendo el diseño: la arena por metro cae de 5,45 a 3,03.
>
> No es un error del modelo, es el hallazgo central: **lo que el modelo aprendió es la
> intensidad por metro, no el largo.** Pero contarlo así, en vivo y sin preparar, confunde.
> Por eso los tres pozos se pegan enteros.
>
> Si alguien lo pregunta, la respuesta corta es: *«un pozo el doble de largo con la misma
> arena no es un pozo mejor, es el mismo diseño estirado».*

---

## Paso 3 — Si sobra tiempo: rendimientos decrecientes

Sobre el **pozo base**, cambiar sólo `cantidad_fracturas` y ejecutar tres veces:

| `cantidad_fracturas` | Espaciamiento | Estimación | Diferencia |
|---:|---:|---:|---:|
| 28 | 100,0 m | 33.365 m³ | — |
| 48 | 58,3 m | 38.506 m³ | **+5.141** |
| 72 | 38,9 m | 38.476 m³ | **−30** |

*«De 28 a 48 etapas el modelo suma cinco mil metros cúbicos. De 48 a 72 no suma nada. Apretar
las etapas por debajo de los 58 metros de espaciamiento es gastar sin contrapartida.»*

---

## Paso 4 — Lo que el servicio avisa

Dos comportamientos que conviene mostrar si hay tiempo, porque son la respuesta a lo que
señaló la devolución del curso.

### 4a. El pozo imposible

Sobre el **pozo base**, cambiar `longitud_rama_horizontal_m` a **8.000** y ejecutar.

Responde igual —**24.156 m³**— pero el campo `avisos` ya no viene vacío:

> `longitud_rama_horizontal_m = 8.000,0 queda por encima de todo lo que el modelo vio al
> entrenar (522,0 a 4.965,0). Un bosque no extrapola: a partir del borde devuelve siempre lo
> mismo. La estimación no es confiable.`

*«El pozo más largo del entrenamiento mide 4.965 metros. Si le pedimos 6.000 u 8.000, el
modelo devuelve prácticamente el mismo número, porque un árbol de decisión no extrapola:
guarda promedios de los pozos que vio. Antes lo estimaba en silencio; ahora lo dice.»*

| Rama (con la intensidad en proporción) | Estimación | Aviso |
|---|---:|---|
| 4.000 m | 50.606 m³ | — |
| 4.965 m | 54.980 m³ | `poca_evidencia` |
| 6.000 m | 55.016 m³ | `extrapolacion` |
| 8.000 m | 54.980 m³ | `extrapolacion` |

### 4b. Las mayúsculas

Escribir la operadora como **`YPF S.A.`** en vez de `ypf s.a.`. Devuelve lo mismo —38.506 m³—
y en `operadora_normalizada` se ve `ypf s.a.`, la forma con la que buscó.

*«El archivo de la Secretaría trae las razones sociales en minúsculas y nadie las escribe
así. La normalización está en el mismo módulo que comparten el entrenamiento y la API.»*

---

## Si la conexión falla

**Seguir sin pedir disculpas.** Los números de la diapositiva son los reales, así que el
argumento se sostiene solo: *«los tengo acá, es lo que devuelve el servicio»*, y continuar.

---

## Lo que conviene no improvisar

Dos campos se comportan de manera que no se explica bien en diez segundos:

- **`arena_bombeada_importada_tn`.** Pasar de 0 a cualquier valor positivo mueve la estimación
  hacia arriba de golpe (38.506 → 42.626 m³) y después se queda plana: el modelo aprendió un
  salto de «hay o no hay importada», probablemente asociado a la época y a qué operadoras la
  usaron, no un efecto de la arena en sí. Es coherente con que sea la variable menos importante
  del modelo, pero la demostración no lo muestra bien.
- **`operadora`** con un nombre inventado. Funciona bien —devuelve 38.308 m³, marca
  `operadora_conocida: false` y agrega el aviso correspondiente— pero invita a la pregunta de
  por qué la operadora pesa tanto, y ésa merece la respuesta larga: **mide la calidad del área
  en concesión, no la habilidad de la empresa.** Si hay tiempo para una sola cosa extra,
  mejor el paso 4a.

---

## La pregunta más probable

**«Un R² de 0,26 es poquísimo.»**

Tres frases, en este orden:

1. Le gana a la referencia —predecir siempre la producción típica— por un **25% de error**.
2. **180 combinaciones de hiperparámetros** no lo movieron: no es un problema de ajuste.
3. Lo que falta es la **geología**, que no está en los datos abiertos. Un modelo que sólo ve
   el diseño no puede explicar la roca.

Y el cierre: *«por eso decimos que sirve para ordenar decisiones de diseño y comparar
conjuntos de pozos, no para dictaminar sobre un pozo individual.»*
