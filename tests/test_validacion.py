import pandas as pd
import pytest

from src.main import RUTA_ORO
from src.validacion import evaluar


def test_metricas_del_conjunto_manual_superan_el_piso():
    """Piso de regresión: si el léxico empeora, esta prueba falla."""
    m = evaluar(pd.read_csv(RUTA_ORO))
    assert m.n >= 50
    assert m.exactitud_sentimiento >= 0.70
    assert m.exactitud_tema >= 0.85


def test_evaluar_exige_columnas():
    with pytest.raises(ValueError):
        evaluar(pd.DataFrame({"texto": ["x"]}))


def test_evaluar_rechaza_vacio():
    with pytest.raises(ValueError):
        evaluar(pd.DataFrame(columns=["texto", "tema", "sentimiento"]))


def test_prediccion_perfecta_da_metricas_uno():
    df = pd.DataFrame({"texto": ["Excelente servicio", "Qué desastre, pésimo"],
                       "tema": ["otros", "otros"],
                       "sentimiento": ["positivo", "negativo"]})
    m = evaluar(df)
    assert m.exactitud_sentimiento == 1.0 and m.exactitud_tema == 1.0
    assert m.errores.empty
