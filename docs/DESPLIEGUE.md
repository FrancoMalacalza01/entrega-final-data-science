# Desplegar la API paso a paso

Guía para poner `src/app.py` en un servidor público y poder demostrarla en vivo.

> ✅ **Este proyecto ya está desplegado en Render:** <https://entregafinal-data-science.onrender.com/docs>
>
> Lo que sigue documenta cómo se hizo y cómo repetirlo, en Render o en las alternativas.

Hay dos caminos y los dos son gratuitos. **Render** es el más parecido a un servicio real y
es el que usamos. **Hugging Face Spaces** es más simple y no se duerme, que para una
demostración en clase es una ventaja concreta.

---

## Antes de empezar

Lo único que hace falta:

- El repositorio en GitHub, que ya está: `github.com/FrancoMalacalza01/preentrega-data-science`
- Una cuenta en la plataforma elegida (se crea con la cuenta de GitHub)

**No hace falta entrenar nada**: `models/modelo.joblib` está versionado en el repositorio, así
que la plataforma solo instala dependencias y levanta el servicio.

### Probarlo en local primero

Antes de pelear con una plataforma, conviene ver que funciona en la propia máquina:

```bash
pip install -r requirements-api.txt
uvicorn src.app:app --reload
```

Y abrir **http://127.0.0.1:8000/docs**. Si eso anda, lo que siga es configuración, no código.

---

## Camino A — Render

### 1. Crear el servicio

1. Entrar a **render.com** e iniciar sesión con GitHub.
2. **New** → **Web Service**.
3. Conectar el repositorio `preentrega-data-science`. Si no aparece, usar **Configure account**
   y darle acceso a Render.

### 2. Configurar

Render detecta el `Dockerfile` y propone **Docker** como entorno. Conviene aceptarlo: la imagen
ya está definida y no hay que escribir comandos.

| Campo | Valor |
|---|---|
| **Name** | `pozos-vaca-muerta` (define la URL) |
| **Region** | la más cercana |
| **Branch** | `main` |
| **Runtime / Language** | **Docker** |
| **Instance Type** | **Free** |

Si en lugar de Docker se elige el entorno de Python, hay que completar a mano:

- **Build Command:** `pip install -r requirements-api.txt`
- **Start Command:** `uvicorn src.app:app --host 0.0.0.0 --port $PORT`

> El `--host 0.0.0.0` no es opcional. Con el valor por defecto el servicio solo escucha
> conexiones internas y la plataforma lo da por caído.

### 3. Desplegar y esperar

**Create Web Service**. El primer despliegue tarda entre cinco y diez minutos: instala pandas,
numpy y scikit-learn, que son pesados. El registro se ve en vivo.

Cuando el estado pase a **Live**, la URL es `https://pozos-vaca-muerta.onrender.com`.

### 4. Comprobar

```bash
curl https://pozos-vaca-muerta.onrender.com/health
curl https://pozos-vaca-muerta.onrender.com/modelo
```

Y en el navegador, **`https://pozos-vaca-muerta.onrender.com/docs`**: ésa es la dirección que
va en la presentación.

### ⚠️ Lo que hay que saber antes de la demostración

**En el plan gratuito, Render apaga el servicio después de 15 minutos sin uso.** La primera
petición que llega lo despierta y **tarda entre 30 y 60 segundos** en responder.

En una demostración en vivo eso se ve como si nada funcionara.

**La solución es simple: despertarlo antes de entrar al aula.** Abrir la URL en una pestaña
diez minutos antes de presentar y dejarla ahí. Mientras haya tráfico no se duerme.

---

## Camino B — Hugging Face Spaces

Más simple y **no se duerme**, que para la clase 32 es la ventaja que importa. Son cuatro
pasos y el proyecto ya está preparado para los cuatro.

### 1. Crear el Space

1. Entrar a **huggingface.co**, iniciar sesión (se puede con la cuenta de GitHub).
2. Arriba a la derecha: **New** → **Space**.
3. Completar:

| Campo | Valor |
|---|---|
| **Owner** | tu usuario |
| **Space name** | `pozos-vaca-muerta` |
| **License** | `mit` |
| **Space SDK** | **Docker** → plantilla **Blank** |
| **Hardware** | *CPU basic · free* |
| **Visibility** | **Public** |

**Create Space**. Queda un repositorio vacío con un `README.md` generado por Hugging Face.

### 2. Generar un token de escritura

En **huggingface.co/settings/tokens** → **New token** → tipo **Write**. Copiarlo: se usa como
contraseña en el paso siguiente y no se vuelve a mostrar.

> El token es una credencial: no se comparte ni se pega en un chat, un ticket o un archivo del
> repositorio.

### 3. Subir el proyecto

Desde una copia local del repositorio, con `TU_USUARIO` reemplazado:

```bash
git remote add space https://huggingface.co/spaces/TU_USUARIO/pozos-vaca-muerta
git push space main --force
```

Pide usuario (el de Hugging Face) y contraseña (**el token**, no la contraseña de la cuenta).

El `--force` es a propósito: el Space arranca con un commit propio que no comparte historia
con el nuestro, así que hay que reemplazarlo.

Se suben unos **14 MB**, sobre todo el modelo entrenado (3,6 MB), los datos (3,5 MB) y los PDF
de `docs/` (3,6 MB). Ninguno llega al límite que obligaría a usar Git LFS, así que no hace
falta configurar nada.

### 4. Declarar el puerto

Éste es el paso que se olvida y hace que el Space quede en *Build error* o en blanco.

Hugging Face lee la configuración del Space de un **encabezado YAML al principio del
`README.md`**. Nuestro `README.md` es el del proyecto y no lo trae, así que el Space no sabe en
qué puerto escucha el contenedor y prueba el 7860, mientras nuestra imagen escucha en el 8000.

Ya está resuelto: **[`docs/space_README.md`](space_README.md)** es un README listo para el
Space, con el encabezado correcto y una descripción del servicio.

En la web del Space: pestaña **Files** → `README.md` → el lápiz de editar → **reemplazar todo
el contenido** por el de `docs/space_README.md` → **Commit changes to main**.

El encabezado que importa es éste:

```yaml
---
title: Productividad de pozos en Vaca Muerta
sdk: docker
app_port: 8000
---
```

`app_port: 8000` es el único valor que no se puede cambiar sin tocar también el `Dockerfile`.

### 5. Comprobar

El Space reconstruye solo y tarda entre cinco y diez minutos la primera vez. La pestaña
**Logs** muestra el avance; cuando termina, el estado pasa a **Running**.

- La aplicación: `https://TU_USUARIO-pozos-vaca-muerta.hf.space`
- La documentación interactiva: `https://TU_USUARIO-pozos-vaca-muerta.hf.space/docs`

```bash
curl https://TU_USUARIO-pozos-vaca-muerta.hf.space/health
```

Tiene que responder `{"estado":"ok","modelo_cargado":true}`.

## Camino C — Solo con Docker, en cualquier servidor

```bash
docker build -t pozos-api .
docker run -p 8000:8000 pozos-api
```

Y en otro puerto, si hiciera falta:

```bash
docker run -e PORT=9000 -p 9000:9000 pozos-api
```

---

## Después de desplegar

1. **Actualizar la presentación.** La diapositiva de demostración dice «documentación
   interactiva en /docs»; conviene reemplazarlo por la dirección real.
2. **Actualizar el README**, en la sección *La API de predicción*, con la URL pública.
3. **Dejar la pestaña abierta** antes de presentar, sobre todo con Render.

---

## Si algo falla

| Síntoma | Causa más probable | Qué hacer |
|---|---|---|
| El despliegue queda en *Build failed* | Falta una dependencia o la versión de Python no tiene ruedas | Mirar el registro: nombra el paquete. Probar `pip install -r requirements-api.txt` en local |
| Arranca y se cae enseguida | El servicio no escucha en `$PORT` o en `0.0.0.0` | Revisar el *Start Command*: tiene que llevar `--host 0.0.0.0 --port $PORT` |
| `/health` devuelve **503** | No está `models/modelo.joblib` | Verificar que `models/` esté versionado. Si no, correr `python src/entrenar.py` y commitear |
| `/predict` devuelve **422** | El cuerpo no cumple las validaciones | La respuesta dice qué campo y por qué. Rama mínima 500 m, etapas entre 1 y 200 |
| Avisos al cargar el modelo | La versión de scikit-learn del servidor no es la que serializó el artefacto | Tiene que ser `1.9.1` exacta, como está fijado en `requirements-api.txt` |
| La primera petición tarda un minuto | El plan gratuito de Render durmió el servicio | Es esperable. Despertarlo antes de presentar |
| Se queda sin memoria | La capa gratuita da 512 MB | No entrenar en el servidor: usar el `models/` versionado, que es lo que hace el `Dockerfile` |
| El Space queda en blanco o en *Build error* | Falta el encabezado YAML en el `README.md` del Space, o `app_port` no es 8000 | Paso 4 del camino B: reemplazar el README del Space por `docs/space_README.md` |
| `git push space` rechaza el envío | El Space arranca con una historia propia | Usar `git push space main --force`, como indica el paso 3 |

---

## Qué se verificó de esta guía y qué no

Honestidad sobre el alcance, porque una guía de despliegue que promete de más hace perder más
tiempo del que ahorra:

**Verificado en este proyecto:**

- La API levanta y responde `/health`, `/`, `/modelo` y `/predict` con datos reales.
- `uvicorn` respeta la variable `PORT`, que es el mecanismo que usan Render y Spaces.
- El camino de servicio **no importa** matplotlib, seaborn ni jupyter, así que
  `requirements-api.txt` alcanza para servir.
- La validación responde 422 ante un diseño fuera de alcance y marca las operadoras
  desconocidas.

**No verificado:**

- La construcción de la imagen Docker, porque el entorno donde se preparó el proyecto no tiene
  un demonio de Docker disponible. El `Dockerfile` es estándar, pero **el primer `docker build`
  conviene correrlo en local antes de subirlo a una plataforma**: es donde aparecería, por
  ejemplo, que alguna de las versiones fijadas no tenga rueda para Python 3.13 en Linux.
- Las pantallas de Render y de Hugging Face, que cambian seguido. Los nombres de los campos
  pueden diferir; lo que no cambia es qué hay que configurar.
