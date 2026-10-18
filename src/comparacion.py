"""Compara la propuesta del Anexo 1 (línea base) con la versión mejorada."""
from __future__ import annotations

import pandas as pd

from .analisis import extraer_rutas
from .anexo_referencia import analyze_data, propose_optimizations
from .ner_spacy import cobertura_catalogo
from .propuestas import generar_propuestas


def comparar_con_anexo(textos: list[str], analizado: pd.DataFrame, nlp=None) -> dict:
    """Mide los problemas de la línea base y los contrasta con la versión mejorada.

    Métricas devueltas:
      lugares_anexo          lugares distintos que cuenta el Anexo
      propuestas_anexo       propuestas del Anexo (umbral fijo > 5 menciones)
      sin_corredor           propuestas sobre algo que no es un corredor del catálogo
      duplicadas             propuestas repetidas para un mismo corredor (alias distintos)
      corredores_unicos      corredores distintos realmente cubiertos
      propuestas_mejoradas   propuestas de la versión mejorada
      cobertura_spacy        proporción de comentarios con corredor que spaCy detecta
    """
    issues = analyze_data(textos, nlp)
    propuestas = propose_optimizations(issues)
    lugares = [p.removeprefix("Aumentar la frecuencia de buses en ") for p in propuestas]
    mapeo = [(extraer_rutas(l) or [None])[0] for l in lugares]
    con_corredor = [m for m in mapeo if m is not None]
    return {
        "lugares_anexo": len(issues),
        "propuestas_anexo": len(propuestas),
        "sin_corredor": sum(1 for m in mapeo if m is None),
        "duplicadas": len(con_corredor) - len(set(con_corredor)),
        "corredores_unicos": len(set(con_corredor)),
        "propuestas_mejoradas": len(generar_propuestas(analizado, top=5)),
        "cobertura_spacy": cobertura_catalogo(textos, nlp)["cobertura"],
    }
