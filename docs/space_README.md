---
title: Productividad de pozos en Vaca Muerta
emoji: 🛢️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Productividad de pozos no convencionales en Vaca Muerta

API que estima los **metros cúbicos de petróleo** que un pozo va a producir en sus primeros
doce meses, a partir de las decisiones de **diseño de completación** que se toman antes de
fracturarlo: metros de rama horizontal, cantidad de etapas, toneladas de arena, metros cúbicos
de agua, presión y potencia de los equipos.

👉 **La documentación interactiva está en [`/docs`](./docs)** — se puede probar el endpoint
desde el navegador, sin escribir código.

## Cómo probarlo

```bash
curl -X POST https://TU_USUARIO-pozos-vaca-muerta.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "longitud_rama_horizontal_m": 2800,
    "cantidad_fracturas": 48,
    "arena_bombeada_nacional_tn": 10900,
    "arena_bombeada_importada_tn": 0,
    "agua_inyectada_m3": 69400,
    "presion_maxima_psi": 11000,
    "potencia_equipos_fractura_hp": 25000,
    "duracion_fractura_dias": 40,
    "operadora": "ypf s.a."
  }'
```

Responde la estimación, un rango a partir del voto de los 600 árboles del bosque, y las
variables de intensidad que el servicio calculó a partir del diseño.

El nombre de la operadora **no distingue mayúsculas**, y la lista de las que el modelo conoce
está en `/operadoras`. Cada respuesta trae además un campo **`avisos`** con los datos del
pedido que caen fuera del rango de entrenamiento: un pozo de 8.000 metros de rama se estima
igual, pero con el aviso de que el más largo que vio el modelo mide 4.965, y que a partir de
ahí un bosque devuelve siempre lo mismo.

## Qué hay detrás

Un **Random Forest** entrenado con **1.917 pozos** de petróleo de Vaca Muerta, con datos
públicos de la Secretaría de Energía de la Nación. Validado con partición temporal —entrenando
con pozos fracturados hasta 2023 y evaluando sobre los de 2024-2025—: **MAE de 10.596 m³**,
un 25% mejor que predecir siempre la producción típica, y **R² de 0,263**.

## Para qué sirve y para qué no

El modelo explica el **26% de la variabilidad**. Sirve para **ordenar decisiones de diseño y
comparar conjuntos de pozos**; no para dictaminar sobre un pozo individual, donde el error se
triplica. Lo que no explica es sobre todo la geología, que no está en los datos abiertos.

La variable `operadora` pesa mucho, pero mide **la calidad del área que cada empresa tiene en
concesión**, no su habilidad: no se puede usar para comparar operadoras.

Son **asociaciones, no relaciones causales**: solo existen los pozos que efectivamente se
perforaron.

## El proyecto completo

Código, notebooks y documentación:
**[github.com/FrancoMalacalza01/entrega-final-data-science](https://github.com/FrancoMalacalza01/entrega-final-data-science)**

Proyecto final del curso de Ciencia de Datos — EnergIA Digital 2026.
Franco Malacalza · Martín Gerbaldo · Carolina Bailón
