"""Fixtures compartidas por las pruebas."""
import pandas as pd
import pytest

from src.analisis import analizar_dataframe
from src.preprocesamiento import preprocesar


@pytest.fixture
def crudo_pequeno() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "fuente": ["foro"] * 8,
        "fecha_hora": [
            "2026-09-02 07:15:00", "2026-09-02 07:40:00", "2026-09-02 18:05:00",
            "2026-09-03 12:00:00", "2026-09-03 22:30:00", "2026-09-04 07:10:00",
            "2026-09-04 07:10:00", "fecha-invalida",
        ],
        "texto": [
            "Esperé 40 minutos en Suba, qué desastre",
            "Suba llena, no cabe nadie, insoportable",
            "Me robaron el celular en la Caracas, inseguro",
            "Hoy la Calle 80 estuvo rápida y puntual, gracias",
            "Me robaron el celular en la Caracas, inseguro",   # duplicado
            "https://t.co/xyz",                               # solo URL
            "",                                               # vacío
            "Demoras en Usme otra vez, 30 minutos esperando",
        ],
    })


@pytest.fixture
def analizado(crudo_pequeno) -> pd.DataFrame:
    return analizar_dataframe(preprocesar(crudo_pequeno))
