# Guion de la demostración en vivo

Para la diapositiva **«El modelo, funcionando»**. Está pensado para imprimirse y tenerlo
al lado: todo lo que hay que pegar está acá, no hay que calcular nada delante de la clase.

**Dirección:** <https://entregafinal-data-science.onrender.com/docs>

---

## Antes de entrar al aula

- [ ] **Diez minutos antes**, abrir la dirección en una pestaña y dejarla abierta. La capa
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
- **`operadora`** con un nombre inventado. Funciona —devuelve 38.308 m³ y marca
  `operadora_conocida: false`, que es la validación trabajando— pero invita a la pregunta de
  por qué la operadora pesa tanto, y ésa merece la respuesta larga: **mide la calidad del área
  en concesión, no la habilidad de la empresa.**

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
