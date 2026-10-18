"""Pruebas basadas en propiedades con Hypothesis.

En vez de fijar ejemplos, se declaran invariantes que deben cumplirse para
CUALQUIER entrada; Hypothesis genera cientos de casos (incluyendo extremos
como cadenas vacías, emojis o caracteres Unicode raros) y reduce los fallos
al ejemplo mínimo.
"""
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.analisis import (
    clasificar_tema, espera_minutos, etiqueta_sentimiento, extraer_rutas,
    franja_horaria, puntaje_sentimiento,
)
from src.config import MAX_ESPERA_MIN, ORDEN_FRANJAS, PRIORIDAD_TEMAS, RUTAS
from src.preprocesamiento import limpiar_texto, quitar_tildes, tokenizar, preprocesar
from src.propuestas import indice_criticidad, normalizar_min_max

texto_libre = st.text(max_size=300)


# ------------------------- preprocesamiento ------------------------------
@given(texto_libre)
def test_limpiar_texto_es_idempotente(t):
    assert limpiar_texto(limpiar_texto(t)) == limpiar_texto(t)


@given(texto_libre)
def test_limpiar_texto_solo_deja_caracteres_permitidos(t):
    salida = limpiar_texto(t)
    assert salida == salida.lower()
    assert "  " not in salida and salida == salida.strip()
    assert all(c.isalnum() or c == " " for c in salida)


@given(texto_libre, st.from_regex(r"https?://[a-z0-9./?=_-]{1,30}", fullmatch=True))
def test_limpiar_texto_elimina_urls(t, url):
    # agregar una URL al final no debe alterar el texto limpio
    assert limpiar_texto(f"{t} {url}") == limpiar_texto(t)


@given(texto_libre)
def test_tokenizar_sin_stopwords_ni_espacios(t):
    from src.preprocesamiento import _STOP_NORMALIZADAS
    for tok in tokenizar(t):
        assert tok and " " not in tok
        assert quitar_tildes(tok) not in _STOP_NORMALIZADAS


@given(texto_libre)
def test_quitar_tildes_es_idempotente(t):
    salida = quitar_tildes(t)
    assert quitar_tildes(salida) == salida  # idempotente


# ------------------------------ análisis ---------------------------------
@given(texto_libre)
def test_sentimiento_siempre_en_rango_y_con_etiqueta_valida(t):
    p = puntaje_sentimiento(t)
    assert -1.0 < p < 1.0
    assert etiqueta_sentimiento(p) in {"positivo", "negativo", "neutro"}


@given(texto_libre)
def test_tema_siempre_es_valido(t):
    assert clasificar_tema(t) in set(PRIORIDAD_TEMAS) | {"otros"}


@given(texto_libre)
def test_rutas_extraidas_son_canonicas_y_sin_repeticion(t):
    r = extraer_rutas(t)
    assert len(r) == len(set(r))
    assert set(r) <= set(RUTAS)


@given(st.sampled_from(sorted(RUTAS)), st.data())
def test_toda_ruta_se_detecta_con_cualquiera_de_sus_alias(ruta, data):
    alias = data.draw(st.sampled_from(RUTAS[ruta]))
    relleno = data.draw(st.sampled_from(["", "hoy en ", "bus de la ", "¡"]))
    assert ruta in extraer_rutas(f"{relleno}{alias.upper()} pésimo")


@given(texto_libre)
def test_espera_minutos_none_o_dentro_de_rango(t):
    e = espera_minutos(t)
    assert e is None or 1 <= e <= MAX_ESPERA_MIN


@given(st.integers(min_value=1, max_value=MAX_ESPERA_MIN))
def test_espera_minutos_recupera_el_numero_escrito(n):
    assert espera_minutos(f"esperé {n} minutos") == n


@given(st.datetimes(min_value=pd.Timestamp("2020-01-01").to_pydatetime(),
                    max_value=pd.Timestamp("2035-01-01").to_pydatetime()))
def test_toda_fecha_cae_en_una_franja_valida(fecha):
    assert franja_horaria(fecha) in ORDEN_FRANJAS


@given(st.sampled_from(["es lento", "es malo", "es peligroso", "es caro", "es sucio"]))
def test_negar_una_palabra_negativa_sube_el_puntaje(frase):
    assert puntaje_sentimiento("no " + frase) > puntaje_sentimiento(frase)


# ------------------------------ propuestas -------------------------------
@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=50))
def test_min_max_siempre_en_cero_uno(valores):
    s = normalizar_min_max(pd.Series(valores))
    assert s.between(0, 1).all()


fila_comentario = st.fixed_dictionaries({
    "rutas": st.lists(st.sampled_from(sorted(RUTAS)), max_size=3, unique=True),
    "sentimiento": st.sampled_from(["positivo", "negativo", "neutro"]),
    "sentimiento_puntaje": st.floats(min_value=-1, max_value=1, allow_nan=False),
    "tema": st.sampled_from(PRIORIDAD_TEMAS + ["otros"]),
    "franja": st.sampled_from(ORDEN_FRANJAS + ["desconocida"]),
    "espera_min": st.one_of(st.none(), st.floats(min_value=1, max_value=MAX_ESPERA_MIN)),
})


@settings(max_examples=75, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.lists(fila_comentario, min_size=1, max_size=60))
def test_criticidad_en_rango_y_ordenada(filas):
    df = pd.DataFrame(filas)
    assume(df["rutas"].map(len).sum() > 0)
    r = indice_criticidad(df)
    assert r["criticidad"].between(0, 1).all()
    assert r["criticidad"].is_monotonic_decreasing
    assert r["ruta"].is_unique
    assert r["menciones"].sum() == df["rutas"].map(len).sum()


@settings(max_examples=50, deadline=None)
@given(st.lists(texto_libre, min_size=1, max_size=15))
def test_preprocesar_nunca_deja_duplicados_ni_vacios(textos):
    df = pd.DataFrame({"id": range(len(textos)), "fuente": "x",
                       "fecha_hora": "2026-09-01 07:00:00", "texto": textos})
    out = preprocesar(df)
    assert out["texto_limpio"].is_unique
    assert (out["texto_limpio"].str.len() >= 3).all()
    assert len(out) <= len(df)
