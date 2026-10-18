import pandas as pd
import pytest

from src.preprocesamiento import (
    limpiar_texto, preprocesar, quitar_tildes, tokenizar, validar_columnas,
)


@pytest.mark.parametrize("entrada,esperado", [
    ("Américas", "americas"),
    ("NIÑO", "nino"),
    ("sin tildes", "sin tildes"),
    ("", ""),
])
def test_quitar_tildes(entrada, esperado):
    assert quitar_tildes(entrada.lower()) == esperado


@pytest.mark.parametrize("entrada,esperado", [
    ("Visita https://t.co/abc123 ya", "visita ya"),
    ("@TransMilenio responde!!!", "responde"),
    ("#Bogotá es caótica 😡", "bogotá es caótica"),
    ("  Mucho   espacio \n\t aquí ", "mucho espacio aquí"),
    ("Pasaje: $3.550", "pasaje 3 550"),
])
def test_limpiar_texto_casos(entrada, esperado):
    assert limpiar_texto(entrada) == esperado


@pytest.mark.parametrize("entrada", [None, float("nan"), 123, ["lista"]])
def test_limpiar_texto_entradas_no_texto(entrada):
    assert limpiar_texto(entrada) == ""


def test_tokenizar_quita_stopwords_y_conserva_negaciones_si_se_pide():
    texto = "No es la peor estación de la ciudad"
    con_stop = tokenizar(texto, quitar_stop=True)
    sin_filtrar = tokenizar(texto, quitar_stop=False)
    assert "no" not in con_stop and "la" not in con_stop
    assert "peor" in con_stop and "estación" in con_stop
    assert sin_filtrar[0] == "no"


def test_preprocesar_elimina_vacios_duplicados_y_parsea_fechas(crudo_pequeno):
    out = preprocesar(crudo_pequeno)
    # 8 filas: 1 solo-URL, 1 vacío y 1 duplicado exacto => 5 quedan
    assert len(out) == 5
    assert out["texto_limpio"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(out["fecha_hora"])
    assert out["fecha_hora"].isna().sum() == 1  # 'fecha-invalida'


def test_preprocesar_no_modifica_el_original(crudo_pequeno):
    copia = crudo_pequeno.copy(deep=True)
    preprocesar(crudo_pequeno)
    pd.testing.assert_frame_equal(crudo_pequeno, copia)


def test_validar_columnas_detecta_faltantes():
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_columnas(pd.DataFrame({"texto": ["x"]}))
