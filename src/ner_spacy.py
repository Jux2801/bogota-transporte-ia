"""Extracción de lugares con spaCy (reconocimiento de entidades nombradas, NER).

Complementa al catálogo de corredores de ``analisis.extraer_rutas``: spaCy
detecta lugares que el catálogo no conoce, y el catálogo corrige los nombres
locales (Autonorte, Portal 80, NQS) que el modelo genérico no siempre entiende.

spaCy es una dependencia opcional: si el modelo no está instalado, las funciones
lanzan un error claro con la instrucción de instalación.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

import pandas as pd

from .analisis import extraer_rutas
from .preprocesamiento import limpiar_texto

MODELO = "es_core_news_sm"
_RE_ARTICULO = re.compile(r"^(?:el|la|los|las|del|al)\s+", re.IGNORECASE)
# Lugares tan genéricos que no sirven para decidir rutas.
LUGARES_GENERICOS = frozenset({"bogota", "colombia", "bogota d c", "ciudad"})


@lru_cache(maxsize=1)
def cargar_modelo(nombre: str = MODELO):
    """Carga (una sola vez) el modelo de spaCy en español."""
    try:
        import spacy
        return spacy.load(nombre)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{nombre}'. Instálelo con:\n"
            f"  pip install spacy && python -m spacy download {nombre}"
        ) from exc


def normalizar_lugar(texto: str) -> str:
    """'la Autonorte' -> 'autonorte' (sin artículo, en minúscula, sin signos)."""
    limpio = limpiar_texto(texto)
    return _RE_ARTICULO.sub("", limpio).strip()


def contar_lugares(textos: Iterable[str], nlp=None) -> Counter:
    """Cuenta las entidades de tipo LOC (lugares) en una colección de textos."""
    nlp = nlp or cargar_modelo()
    conteo: Counter = Counter()
    for texto in textos:
        if not isinstance(texto, str) or not texto.strip():
            continue
        for ent in nlp(texto).ents:
            if ent.label_ == "LOC":
                lugar = normalizar_lugar(ent.text)
                if lugar:
                    conteo[lugar] += 1
    return conteo


def comparar_con_catalogo(textos: list[str], nlp=None) -> pd.DataFrame:
    """Tabla de lugares detectados por spaCy y su relación con el catálogo.

    Columnas: lugar, menciones, corredor_catalogo (o None), es_generico.
    """
    filas = []
    for lugar, n in contar_lugares(textos, nlp).most_common():
        rutas = extraer_rutas(lugar)
        filas.append({
            "lugar": lugar, "menciones": n,
            "corredor_catalogo": rutas[0] if rutas else None,
            "es_generico": lugar in LUGARES_GENERICOS,
        })
    return pd.DataFrame(filas, columns=["lugar", "menciones", "corredor_catalogo", "es_generico"])


def cobertura_catalogo(textos: list[str], nlp=None) -> dict[str, float]:
    """Mide qué tan bien spaCy encuentra los corredores que el catálogo reconoce.

    Para cada comentario con algún corredor del catálogo, revisa si spaCy marcó
    como LOC algún texto que el catálogo reconozca como ese mismo corredor.
    """
    nlp = nlp or cargar_modelo()
    con_ruta = acertados = 0
    for texto in textos:
        esperadas = set(extraer_rutas(texto))
        if not esperadas:
            continue
        con_ruta += 1
        detectadas = set()
        for ent in nlp(texto).ents:
            if ent.label_ == "LOC":
                detectadas.update(extraer_rutas(ent.text))
        if esperadas & detectadas:
            acertados += 1
    return {
        "comentarios_con_corredor": con_ruta,
        "detectados_por_spacy": acertados,
        "cobertura": acertados / con_ruta if con_ruta else 0.0,
    }
