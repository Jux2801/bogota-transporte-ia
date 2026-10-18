"""Pruebas de la parte con spaCy y de la línea base del Anexo.

La mayoría usa un ``nlp`` falso (rápido y sin descargar el modelo). Las pruebas
de integración con el modelo real se omiten si no está instalado.
"""
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.anexo_referencia import analyze_data, propose_optimizations
from src.comparacion import comparar_con_anexo
from src.analisis import analizar_dataframe
from src.ner_spacy import (
    cobertura_catalogo, comparar_con_catalogo, contar_lugares, normalizar_lugar,
)
from src.preprocesamiento import preprocesar

LUGARES = ["la Autonorte", "Suba", "Portal 80", "Bogotá", "TransMilenio"]


class Ent:
    def __init__(self, text, label_):
        self.text, self.label_ = text, label_


class Doc:
    def __init__(self, ents):
        self.ents = ents


class NlpFalso:
    """Marca como LOC los lugares conocidos que aparezcan en el texto."""
    def __call__(self, texto):
        ents = [Ent(l, "LOC") for l in LUGARES if l in texto]
        if "Increíble" in texto:
            ents.append(Ent("Increíble", "MISC"))
        return Doc(ents)


NLP = NlpFalso()


# ------------------------------- unitarias -------------------------------
@pytest.mark.parametrize("entrada,esperado", [
    ("la Autonorte", "autonorte"), ("El Portal 80", "portal 80"),
    ("Suba", "suba"), ("los Héroes", "héroes"), ("", ""),
])
def test_normalizar_lugar(entrada, esperado):
    assert normalizar_lugar(entrada) == esperado


def test_contar_lugares_agrupa_y_solo_cuenta_loc():
    textos = ["Demora en Suba", "Suba y la Autonorte", "Increíble servicio", "", None]
    c = contar_lugares(textos, NLP)
    assert c["suba"] == 2 and c["autonorte"] == 1
    assert "increíble" not in c


def test_comparar_con_catalogo_relaciona_alias_con_corredor():
    tabla = comparar_con_catalogo(["Suba lleno", "la Autonorte tarda", "Bogotá hoy"], NLP)
    por_lugar = tabla.set_index("lugar")
    assert por_lugar.loc["suba", "corredor_catalogo"] == "Suba"
    assert por_lugar.loc["autonorte", "corredor_catalogo"] == "Autopista Norte"
    assert bool(por_lugar.loc["bogotá", "es_generico"]) is False  # 'bogotá' con tilde no está en la lista
    assert pd.isna(por_lugar.loc["bogotá", "corredor_catalogo"])


def test_cobertura_catalogo():
    textos = ["Suba lleno", "Demora en la Caracas", "sin lugares"]  # Caracas: spaCy falso no lo ve
    r = cobertura_catalogo(textos, NLP)
    assert r["comentarios_con_corredor"] == 2
    assert r["detectados_por_spacy"] == 1
    assert r["cobertura"] == 0.5


def test_cobertura_sin_comentarios_con_corredor():
    assert cobertura_catalogo(["hola"], NLP)["cobertura"] == 0.0


# ----------------------- línea base del Anexo -----------------------------
def test_anexo_analyze_data_cuenta_entidades_loc():
    assert analyze_data(["Suba lleno", "Suba tarde", "Portal 80"], NLP) == {"Suba": 2, "Portal 80": 1}


def test_anexo_umbral_es_estrictamente_mayor_que_cinco():
    assert propose_optimizations({"A": 5, "B": 6}) == ["Aumentar la frecuencia de buses en B"]


def test_anexo_limitacion_ignora_el_sentimiento():
    """Documenta la debilidad: elogios y quejas generan la misma propuesta."""
    quejas = ["Pésimo servicio en Suba"] * 6
    elogios = ["Excelente servicio en Suba"] * 6
    assert (propose_optimizations(analyze_data(quejas, NLP))
            == propose_optimizations(analyze_data(elogios, NLP)))


def test_anexo_limitacion_fragmenta_alias_de_un_mismo_corredor():
    class NlpAlias:
        def __call__(self, t):
            return Doc([Ent(a, "LOC") for a in ("Autonorte", "Portal Norte") if a in t])
    textos = ["Autonorte mal", "Portal Norte mal"] * 6
    props = propose_optimizations(analyze_data(textos, NlpAlias()))
    assert len(props) == 2  # dos propuestas para el mismo corredor


# --------------------- propiedades (Hypothesis) ---------------------------
@given(st.lists(st.text(max_size=80), max_size=20))
def test_prop_analyze_data_devuelve_dict_de_enteros_positivos(textos):
    issues = analyze_data(textos, NLP)
    assert isinstance(issues, dict)
    assert all(isinstance(k, str) and isinstance(v, int) and v > 0 for k, v in issues.items())


@given(st.lists(st.sampled_from(["Suba", "Portal 80", "Bogotá", "otra cosa", ""]), max_size=30))
def test_prop_total_de_conteos_igual_a_entidades_loc(textos):
    esperado = sum(len(NLP(t).ents) for t in textos)
    assert sum(analyze_data(textos, NLP).values()) == esperado


@given(st.dictionaries(st.text(max_size=20), st.integers(min_value=-5, max_value=50)))
def test_prop_propuestas_solo_para_lugares_sobre_el_umbral(issues):
    props = propose_optimizations(issues)
    assert isinstance(props, list)
    assert len(props) == sum(1 for v in issues.values() if v > 5)


# --------------------------- comparación ----------------------------------
def test_comparar_con_anexo_detecta_ruido_y_duplicados():
    class NlpMixto:
        def __call__(self, t):
            return Doc([Ent(a, "LOC") for a in ("Autonorte", "Portal Norte", "TransMilenio") if a in t])
    filas = []
    for i in range(6):
        filas += [
            dict(id=f"a{i}", fuente="x", fecha_hora="2026-09-01 07:00:00", texto=f"Autonorte mal número {i}"),
            dict(id=f"b{i}", fuente="x", fecha_hora="2026-09-01 07:00:00", texto=f"Portal Norte mal número {i}"),
            dict(id=f"c{i}", fuente="x", fecha_hora="2026-09-01 07:00:00", texto=f"TransMilenio mal número {i}"),
        ]
    an = analizar_dataframe(preprocesar(pd.DataFrame(filas)))
    r = comparar_con_anexo(an["texto"].tolist(), an, NlpMixto())
    assert r["propuestas_anexo"] == 3
    assert r["sin_corredor"] == 1          # 'TransMilenio' no es un corredor
    assert r["duplicadas"] == 1            # Autonorte y Portal Norte = mismo corredor
    assert r["corredores_unicos"] == 1


# --------------- integración con el modelo real (opcional) ----------------
@pytest.fixture(scope="module")
def nlp_real():
    pytest.importorskip("spacy")
    from src.ner_spacy import cargar_modelo
    try:
        return cargar_modelo()
    except RuntimeError:
        pytest.skip("modelo es_core_news_sm no instalado")


def test_modelo_real_detecta_lugares_conocidos(nlp_real):
    c = contar_lugares(["El bus de Suba llega tarde a Bogotá"], nlp_real)
    assert c["suba"] == 1 and c["bogotá"] == 1


@settings(max_examples=15, deadline=None)
@given(st.lists(st.text(max_size=60), max_size=5))
def test_modelo_real_analyze_data_siempre_devuelve_dict(nlp_real, textos):
    assert isinstance(analyze_data(textos, nlp_real), dict)
