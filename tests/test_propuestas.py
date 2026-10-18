import pandas as pd
import pytest

from src.propuestas import (
    ACCIONES, franja_critica, generar_propuestas, indice_criticidad,
    normalizar_min_max, tema_dominante,
)


def _df_sintetico() -> pd.DataFrame:
    """Dos corredores: A muy crítico (picos, esperas largas) y B tranquilo."""
    filas = []
    for i in range(10):
        filas.append(dict(rutas=["A"], sentimiento="negativo", sentimiento_puntaje=-0.6,
                          tema="tiempos_espera", franja="pico_am", espera_min=40.0))
    for i in range(2):
        filas.append(dict(rutas=["B"], sentimiento="positivo", sentimiento_puntaje=0.4,
                          tema="otros", franja="valle", espera_min=None))
    filas.append(dict(rutas=[], sentimiento="negativo", sentimiento_puntaje=-0.5,
                      tema="inseguridad", franja="noche", espera_min=None))
    return pd.DataFrame(filas)


def test_normalizar_min_max_rango_y_constante():
    s = normalizar_min_max(pd.Series([2, 4, 6]))
    assert s.min() == 0 and s.max() == 1
    assert (normalizar_min_max(pd.Series([5, 5, 5])) == 0).all()


def test_indice_criticidad_ordena_y_acota():
    r = indice_criticidad(_df_sintetico())
    assert list(r["ruta"]) == ["A", "B"]
    assert r["criticidad"].between(0, 1).all()
    assert r.loc[0, "prop_negativa"] == 1.0


def test_indice_criticidad_ignora_comentarios_sin_ruta():
    r = indice_criticidad(_df_sintetico())
    assert r["menciones"].sum() == 12


def test_pesos_invalidos():
    with pytest.raises(ValueError):
        indice_criticidad(_df_sintetico(), pesos=(0.5, 0.5, 0.5))


def test_franja_y_tema_dominante():
    df = _df_sintetico()
    assert franja_critica(df, "A") == "pico_am"
    assert tema_dominante(df, "A") == "tiempos_espera"
    assert franja_critica(df, "B") == "desconocida"
    assert tema_dominante(df, "B") == "otros"


def test_generar_propuestas_estructura():
    p = generar_propuestas(_df_sintetico(), top=1)
    assert len(p) == 1
    fila = p.iloc[0]
    assert fila["ruta"] == "A" and fila["prioridad"] == 1
    assert fila["accion_propuesta"] == ACCIONES["tiempos_espera"]
    assert bool(fila["requiere_obra_civil"]) is False


def test_generar_propuestas_top_invalido():
    with pytest.raises(ValueError):
        generar_propuestas(_df_sintetico(), top=0)


def test_dataframe_sin_rutas_da_ranking_vacio():
    df = pd.DataFrame(dict(rutas=[[]], sentimiento=["neutro"], sentimiento_puntaje=[0.0],
                           tema=["otros"], franja=["valle"], espera_min=[None]))
    assert indice_criticidad(df).empty
    assert generar_propuestas(df).empty


def test_todas_las_acciones_son_operativas():
    # ninguna acción propuesta debe pedir obra civil
    palabras_obra = ("construir", "ampliar estación", "nueva troncal", "obra civil")
    for texto in ACCIONES.values():
        assert not any(p in texto.lower() for p in palabras_obra)
