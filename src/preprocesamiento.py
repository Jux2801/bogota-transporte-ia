"""Limpieza y normalización de texto en español (etapa de preprocesamiento NLP)."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .config import STOPWORDS

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENCION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#(\w+)")
# Todo lo que no sea letra latina básica/acentuada, ñ, ü, dígito o espacio.
_RE_RUIDO = re.compile(r"[^0-9a-záéíóúüñ\s]")

COLUMNAS_OBLIGATORIAS = ("id", "fuente", "fecha_hora", "texto")


def quitar_tildes(texto: str) -> str:
    """Elimina diacríticos ('américas' -> 'americas')."""
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def limpiar_texto(texto: object) -> str:
    """Devuelve el texto en minúsculas, sin URLs, menciones, emojis ni signos.

    Conserva tildes y la 'ñ'. Es idempotente: limpiar(limpiar(x)) == limpiar(x).
    Si la entrada no es texto (NaN, None, números) devuelve cadena vacía.
    """
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_MENCION.sub(" ", t)
    t = _RE_HASHTAG.sub(r" \1 ", t)
    t = _RE_RUIDO.sub(" ", t)
    return " ".join(t.split())


_STOP_NORMALIZADAS = frozenset(quitar_tildes(s) for s in STOPWORDS)


def tokenizar(texto: object, quitar_stop: bool = True) -> list[str]:
    """Convierte el texto en una lista de tokens limpios.

    Con ``quitar_stop=True`` se descartan palabras vacías y tokens de un solo
    carácter no numéricos. El análisis de sentimiento llama con
    ``quitar_stop=False`` porque 'no', 'sin' o 'muy' cambian el significado.
    """
    tokens = limpiar_texto(texto).split()
    if not quitar_stop:
        return tokens
    return [
        t for t in tokens
        if quitar_tildes(t) not in _STOP_NORMALIZADAS and (len(t) > 1 or t.isdigit())
    ]


def validar_columnas(df: pd.DataFrame) -> None:
    """Lanza ValueError si faltan columnas obligatorias."""
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")


def preprocesar(df: pd.DataFrame, longitud_minima: int = 3) -> pd.DataFrame:
    """Aplica el pipeline de limpieza a un DataFrame de comentarios crudos.

    Pasos: validar columnas -> limpiar -> descartar vacíos/muy cortos ->
    eliminar duplicados exactos del texto limpio -> parsear fechas.
    """
    validar_columnas(df)
    out = df.copy()
    out["texto_limpio"] = out["texto"].map(limpiar_texto)
    out = out[out["texto_limpio"].str.len() >= longitud_minima]
    out = out.drop_duplicates(subset="texto_limpio", keep="first")
    out["fecha_hora"] = pd.to_datetime(out["fecha_hora"], errors="coerce")
    return out.reset_index(drop=True)
