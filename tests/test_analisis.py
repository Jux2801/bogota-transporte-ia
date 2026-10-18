import pandas as pd
import pytest

from src.analisis import (
    analizar_dataframe, clasificar_tema, descubrir_temas, espera_minutos,
    etiqueta_sentimiento, extraer_rutas, franja_horaria, puntaje_sentimiento,
)


# ---------------------------- sentimiento --------------------------------
@pytest.mark.parametrize("texto,etiqueta", [
    ("Qué desastre, servicio pésimo", "negativo"),
    ("Excelente servicio, bus limpio y puntual", "positivo"),
    ("¿A qué hora abre el portal?", "neutro"),
    ("", "neutro"),
])
def test_etiqueta_sentimiento(texto, etiqueta):
    assert etiqueta_sentimiento(puntaje_sentimiento(texto)) == etiqueta


def test_negacion_invierte_polaridad():
    assert puntaje_sentimiento("es lento") < 0
    assert puntaje_sentimiento("no es lento") > 0


def test_intensificador_aumenta_magnitud():
    assert abs(puntaje_sentimiento("muy malo")) > abs(puntaje_sentimiento("malo"))


def test_tildes_no_afectan_el_sentimiento():
    assert puntaje_sentimiento("pésimo") == puntaje_sentimiento("pesimo")


# ------------------------------- temas -----------------------------------
@pytest.mark.parametrize("texto,tema", [
    ("Llevo 40 minutos esperando el bus", "tiempos_espera"),
    ("El bus va abarrotado, no cabe nadie", "hacinamiento"),
    ("Me atracaron en la estación", "inseguridad"),
    ("Bus varado bloqueó toda la troncal", "fallas_operativas"),
    ("No puedo recargar la Tu Llave", "tarifa_pago"),
    ("Estación sucia y sin información", "estaciones_servicio"),
    ("Feliz domingo a todos", "otros"),
    ("", "otros"),
])
def test_clasificar_tema(texto, tema):
    assert clasificar_tema(texto) == tema


def test_tarde_no_coincide_con_tardes():
    # 'buenas tardes' no debe contarse como queja de espera
    assert clasificar_tema("Buenas tardes a todos") == "otros"


# ------------------------------- rutas -----------------------------------
def test_extraer_rutas_alias_y_sin_repetir():
    r = extraer_rutas("Entre el Portal 80 y la Calle 80 y también Suba")
    assert r == ["Calle 80", "Suba"]


def test_extraer_rutas_ignora_tildes_y_mayusculas():
    assert extraer_rutas("Por la AMÉRICAS no pasa nada") == ["Avenida Américas"]


def test_extraer_rutas_sin_coincidencias():
    assert extraer_rutas("nada que ver") == []


# ------------------------------ franjas ----------------------------------
@pytest.mark.parametrize("hora,franja", [
    (0, "madrugada"), (4, "madrugada"), (5, "pico_am"), (8, "pico_am"),
    (9, "valle"), (15, "valle"), (16, "pico_pm"), (19, "pico_pm"),
    (20, "noche"), (23, "noche"),
])
def test_franja_horaria_limites(hora, franja):
    assert franja_horaria(pd.Timestamp(2026, 9, 1, hora, 30)) == franja


@pytest.mark.parametrize("nulo", [None, pd.NaT, float("nan")])
def test_franja_horaria_nula(nulo):
    assert franja_horaria(nulo) == "desconocida"


# ----------------------------- esperas -----------------------------------
@pytest.mark.parametrize("texto,minutos", [
    ("esperé 40 minutos", 40),
    ("llevo 30min parado", 30),
    ("dos horas de espera", 120),
    ("media hora esperando", 30),
    ("hora y media en la fila", 90),
    ("sin datos de tiempo", None),
    ("esperé 900 minutos", None),   # implausible
    ("0 minutos", None),
])
def test_espera_minutos(texto, minutos):
    assert espera_minutos(texto) == minutos


# ----------------------------- pipeline ----------------------------------
def test_analizar_dataframe_agrega_columnas(analizado):
    for col in ("sentimiento_puntaje", "sentimiento", "tema", "rutas", "franja", "espera_min"):
        assert col in analizado.columns
    fila = analizado[analizado["texto"].str.contains("40 minutos")].iloc[0]
    assert fila["sentimiento"] == "negativo"
    assert fila["tema"] == "tiempos_espera"
    assert fila["rutas"] == ["Suba"]
    assert fila["franja"] == "pico_am"
    assert fila["espera_min"] == 40


# ---------------------------- clustering ---------------------------------
def test_descubrir_temas_es_reproducible():
    textos = (["bus tarda mucho espera larga estacion"] * 6
              + ["robo celular inseguridad ladrones estacion"] * 6
              + ["bus lleno apretado sardinas gente"] * 6)
    a = descubrir_temas(textos, k=3, semilla=1)
    b = descubrir_temas(textos, k=3, semilla=1)
    assert a.etiquetas == b.etiquetas
    assert len(set(a.etiquetas)) == 3


def test_descubrir_temas_reduce_k_si_hay_pocos_textos():
    r = descubrir_temas(["bus lento", "estacion sucia"], k=6)
    assert len(r.terminos) == 2


def test_descubrir_temas_sin_textos():
    with pytest.raises(ValueError):
        descubrir_temas([])


def test_descubrir_temas_solo_palabras_vacias():
    with pytest.raises(ValueError, match="informativos"):
        descubrir_temas(["uno dos", "el la los"])
