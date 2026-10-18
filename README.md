# Optimización del transporte público de Bogotá con IA y análisis de opiniones

Proyecto del estudio de caso de **Inteligencia Artificial** (UNIMINUTO).
Plataforma en Python que **recolecta** comentarios públicos sobre TransMilenio/SITP
(Selenium), los **analiza** con técnicas de PLN (limpieza, sentimiento, clasificación
temática, extracción de corredores y esperas, TF-IDF + K-means) y **genera propuestas
operativas** de rutas y horarios que no requieren obra civil.

> **Aviso sobre los datos.** `data/muestra_comentarios.csv` es una muestra **simulada**
> (la crea `src/muestra.py`). Sirve para ejecutar y probar el pipeline. Sus resultados
> **no** describen la opinión real de los usuarios. Para un análisis real, recolecte datos
> con `src/recoleccion.py` y pase el CSV con `--entrada`.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm   # modelo de spaCy en español
```

Para recolectar con Selenium se necesita Google Chrome instalado (Selenium Manager
descarga el driver automáticamente).

## Uso

```bash
python -m src.main --generar-muestra     # crea la muestra simulada y analiza
python -m src.main                       # analiza data/muestra_comentarios.csv
python -m src.main --entrada data/mis_datos.csv --salida resultados_reales
python -m src.main --comparar-anexo     # contrasta el código base del Anexo 1 (spaCy)
```

Salidas en `resultados/`: `comentarios_analizados.csv`, `ranking_rutas.csv`,
`propuestas.csv` y `figuras/*.png`.

### Recolección real (ejemplo)

```python
from src.recoleccion import ConfigFuente, recolectar, guardar_csv

cfg = ConfigFuente(
    nombre="foro_ejemplo",
    url_base="https://foro.ejemplo.co/transmilenio?pagina={pagina}",
    selector_item="div.comentario",       # ajustar al HTML del sitio
    selector_texto="p.texto",
    selector_fecha="time",
    max_paginas=5,
)
df = recolectar([cfg])                    # respeta robots.txt y pausa entre páginas
guardar_csv(df, "data/mis_datos.csv")
```

Antes de recolectar revise los términos de uso del sitio. Las redes sociales como X o
Facebook exigen sesión y restringen la extracción automática; prefiera foros públicos,
APIs oficiales o datos abiertos. No se almacenan nombres de usuario (Ley 1581 de 2012).

## Pruebas

```bash
python -m pytest                                  # 123 pruebas
python -m pytest --hypothesis-show-statistics     # estadísticas de Hypothesis
```

* `tests/test_*.py`: pruebas unitarias con **PyTest** (parametrizadas, fixtures, mocks).
* `tests/test_propiedades.py`: pruebas basadas en propiedades con **Hypothesis**.
* `tests/test_recoleccion.py`: usa un navegador falso, no necesita red ni Chrome.
* `tests/test_spacy.py`: spaCy y línea base del Anexo (nlp falso; integración con el modelo real si está instalado).
* `tests/test_validacion.py`: piso de regresión sobre `data/validacion_manual.csv`.

## Estructura

```
src/
  config.py          recursos léxicos: corredores, temas, léxico, stopwords
  recoleccion.py     Selenium + robots.txt + CSV
  muestra.py         generador de muestra simulada
  preprocesamiento.py limpieza y tokenización
  analisis.py        sentimiento, temas, entidades, TF-IDF + K-means
  ner_spacy.py       lugares con spaCy (NER) y comparación con el catálogo
  anexo_referencia.py réplica del código base del Anexo 1 (línea base)
  comparacion.py     contraste Anexo vs. versión mejorada
  propuestas.py      índice de criticidad y acciones operativas
  validacion.py      métricas contra el conjunto manual
  visualizacion.py   gráficos
  main.py            pipeline de extremo a extremo
tests/               pytest + hypothesis
data/                muestra simulada y conjunto de validación manual
```

## Limitaciones conocidas

* El análisis de sentimiento por léxico no detecta el sarcasmo.
* Solo se reconocen corredores troncales; las rutas zonales del SITP no están en el catálogo.
* El conjunto de validación (53 comentarios) lo escribió el autor y se usó también para
  revisar errores del léxico: sus métricas son optimistas. Use un conjunto independiente.
