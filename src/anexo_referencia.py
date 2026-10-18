"""Réplica de la lógica del código base del Anexo 1, para poder analizarla.

Se conserva el comportamiento original (conteo de entidades LOC y umbral fijo
de 5 menciones). Cambios mínimos y deliberados respecto al Anexo, todos para
poder ejecutarla y probarla:
  * recibe ``nlp`` por parámetro en vez de cargarlo al importar el módulo;
  * los nombres de funciones se dejan igual (analyze_data, propose_optimizations).
Sirve de línea base: ``main --comparar-anexo`` la contrasta con la versión mejorada.
"""
from __future__ import annotations

from .ner_spacy import cargar_modelo


def analyze_data(data: list[str], nlp=None) -> dict[str, int]:
    """Cuenta cuántas veces aparece cada entidad LOC (igual que el Anexo)."""
    nlp = nlp or cargar_modelo()
    issues: dict[str, int] = {}
    for text in data:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "LOC":
                if ent.text not in issues:
                    issues[ent.text] = 0
                issues[ent.text] += 1
    return issues


def propose_optimizations(issues: dict[str, int]) -> list[str]:
    """Propone aumentar la frecuencia donde haya más de 5 menciones (igual que el Anexo)."""
    optimizations = []
    for location, count in issues.items():
        if count > 5:  # umbral para considerar un problema significativo
            optimizations.append(f"Aumentar la frecuencia de buses en {location}")
    return optimizations
