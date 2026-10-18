"""Técnicas de IA/NLP: sentimiento, clasificación temática, extracción y clustering."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .config import (
    FRANJAS, INTENSIFICADORES, LEXICO_SENTIMIENTO, MAX_ESPERA_MIN, NEGADORES,
    PRIORIDAD_TEMAS, RUTAS, STOPWORDS, TEMAS, UMBRAL_SENTIMIENTO,
)
from .preprocesamiento import limpiar_texto, quitar_tildes, tokenizar


# ---------------------------------------------------------------------------
# 1) Análisis de sentimiento basado en léxico, con negación e intensificadores
# ---------------------------------------------------------------------------
def puntaje_sentimiento(texto: object) -> float:
    """Devuelve un puntaje compuesto en el intervalo [-1, 1].

    Se suman los pesos del léxico; un negador ('no', 'nunca', 'sin'...) invierte
    la polaridad de la palabra que aparece en las dos posiciones siguientes y un
    intensificador ('muy', 'super'...) la multiplica por 1.5. El total se
    normaliza como en VADER: x / sqrt(x² + 15).
    """
    tokens = [quitar_tildes(t) for t in tokenizar(texto, quitar_stop=False)]
    total = 0.0
    for i, tok in enumerate(tokens):
        peso = LEXICO_SENTIMIENTO.get(tok)
        if peso is None:
            continue
        valor = float(peso)
        ventana = tokens[max(0, i - 2): i]
        if any(v in INTENSIFICADORES for v in ventana):
            valor *= 1.5
        if any(v in NEGADORES for v in ventana):
            valor *= -0.74  # la negación invierte y atenúa
        total += valor
    return total / math.sqrt(total * total + 15.0)


def etiqueta_sentimiento(puntaje: float) -> str:
    """Convierte el puntaje en 'positivo', 'negativo' o 'neutro'."""
    if puntaje > UMBRAL_SENTIMIENTO:
        return "positivo"
    if puntaje < -UMBRAL_SENTIMIENTO:
        return "negativo"
    return "neutro"


# ---------------------------------------------------------------------------
# 2) Clasificación temática por reglas (palabras clave ponderadas)
# ---------------------------------------------------------------------------
def _compilar(entrada: str) -> re.Pattern[str]:
    if entrada.endswith("*"):
        return re.compile(r"\b" + re.escape(entrada[:-1]))
    return re.compile(r"\b" + re.escape(entrada) + r"\b")


@lru_cache(maxsize=1)
def _patrones_temas() -> dict[str, list[re.Pattern[str]]]:
    return {t: [_compilar(e) for e in palabras] for t, palabras in TEMAS.items()}


def clasificar_tema(texto: object) -> str:
    """Asigna el tema con más coincidencias; 'otros' si no hay ninguna.

    Los empates se resuelven con ``PRIORIDAD_TEMAS``.
    """
    t = quitar_tildes(limpiar_texto(texto))
    mejor, mejor_n = "otros", 0
    for tema in PRIORIDAD_TEMAS:
        n = sum(1 for p in _patrones_temas()[tema] if p.search(t))
        if n > mejor_n:
            mejor, mejor_n = tema, n
    return mejor


# ---------------------------------------------------------------------------
# 3) Extracción de entidades: corredores, franja horaria y tiempos de espera
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _patrones_rutas() -> list[tuple[str, re.Pattern[str]]]:
    pares = []
    for canon, alias in RUTAS.items():
        for a in alias:
            pares.append((canon, re.compile(r"\b" + re.escape(a) + r"\b")))
    return pares


def extraer_rutas(texto: object) -> list[str]:
    """Devuelve los corredores mencionados (nombre canónico, sin repetir)."""
    t = quitar_tildes(limpiar_texto(texto))
    vistos: list[str] = []
    for canon, patron in _patrones_rutas():
        if canon not in vistos and patron.search(t):
            vistos.append(canon)
    return vistos


def franja_horaria(fecha: object) -> str:
    """Clasifica una marca de tiempo en su franja; 'desconocida' si es nula."""
    if fecha is None or pd.isna(fecha):
        return "desconocida"
    hora = pd.Timestamp(fecha).hour
    for nombre, (ini, fin) in FRANJAS.items():
        if ini <= hora < fin:
            return nombre
    return "desconocida"  # inalcanzable: las franjas cubren 0-23


_NUM_PALABRAS = {
    "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "diez": 10,
    "quince": 15, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
}
_NUM = r"(\d{1,3}|" + "|".join(_NUM_PALABRAS) + r")"
_RE_MIN = re.compile(_NUM + r"\s*(?:min|mins|minutos)\b")
_RE_HORAS = re.compile(_NUM + r"\s*horas?\b")


def _a_entero(s: str) -> int:
    return int(s) if s.isdigit() else _NUM_PALABRAS[s]


def espera_minutos(texto: object) -> int | None:
    """Extrae el tiempo de espera mencionado, en minutos (None si no hay).

    Reconoce '40 minutos', '30min', 'dos horas', 'media hora' y 'hora y media'.
    Valores fuera de 1..MAX_ESPERA_MIN se descartan por implausibles.
    """
    t = quitar_tildes(limpiar_texto(texto))
    candidatos: list[int] = []
    candidatos += [_a_entero(m) for m in _RE_MIN.findall(t)]
    candidatos += [_a_entero(m) * 60 for m in _RE_HORAS.findall(t)]
    if "media hora" in t:
        candidatos.append(30)
    if "hora y media" in t:
        candidatos.append(90)
    validos = [c for c in candidatos if 1 <= c <= MAX_ESPERA_MIN]
    return max(validos) if validos else None


# ---------------------------------------------------------------------------
# 4) Pipeline sobre DataFrame
# ---------------------------------------------------------------------------
def analizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de análisis a un DataFrame ya preprocesado."""
    out = df.copy()
    out["sentimiento_puntaje"] = out["texto"].map(puntaje_sentimiento)
    out["sentimiento"] = out["sentimiento_puntaje"].map(etiqueta_sentimiento)
    out["tema"] = out["texto"].map(clasificar_tema)
    out["rutas"] = out["texto"].map(extraer_rutas)
    out["franja"] = out["fecha_hora"].map(franja_horaria)
    out["espera_min"] = out["texto"].map(espera_minutos)
    return out


# ---------------------------------------------------------------------------
# 5) Aprendizaje no supervisado: TF-IDF + K-means para descubrir temas
# ---------------------------------------------------------------------------
@dataclass
class ResultadoClusters:
    etiquetas: list[int]
    terminos: dict[int, list[str]]
    silueta: float | None


def descubrir_temas(textos: list[str], k: int = 6, semilla: int = 42,
                    n_terminos: int = 6) -> ResultadoClusters:
    """Agrupa comentarios con TF-IDF y K-means y devuelve los términos top.

    Si hay menos documentos que clusters se reduce k automáticamente.
    """
    if not textos:
        raise ValueError("Se requiere al menos un texto")
    vec = TfidfVectorizer(
        preprocessor=lambda s: quitar_tildes(limpiar_texto(s)),
        stop_words=sorted(quitar_tildes(s) for s in STOPWORDS),
        ngram_range=(1, 2), min_df=2 if len(textos) >= 20 else 1,
    )
    try:
        X = vec.fit_transform(textos)
    except ValueError as exc:  # vocabulario vacío (solo palabras vacías)
        raise ValueError("Los textos no contienen términos informativos") from exc
    k_real = max(1, min(k, X.shape[0]))
    km = KMeans(n_clusters=k_real, n_init=10, random_state=semilla).fit(X)
    vocab = vec.get_feature_names_out()
    terminos = {}
    for c in range(k_real):
        idx = km.cluster_centers_[c].argsort()[::-1][:n_terminos]
        terminos[c] = [str(vocab[i]) for i in idx]
    sil = None
    if 1 < k_real < X.shape[0]:
        sil = float(silhouette_score(X, km.labels_, random_state=semilla))
    return ResultadoClusters([int(x) for x in km.labels_], terminos, sil)
