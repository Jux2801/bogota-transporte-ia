"""Validación de los clasificadores contra un conjunto etiquetado manualmente."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .analisis import clasificar_tema, etiqueta_sentimiento, puntaje_sentimiento


@dataclass
class MetricasValidacion:
    n: int
    exactitud_sentimiento: float
    f1_sentimiento: float
    exactitud_tema: float
    f1_tema: float
    matriz_sentimiento: pd.DataFrame
    errores: pd.DataFrame


def evaluar(df_oro: pd.DataFrame) -> MetricasValidacion:
    """Compara las predicciones con las etiquetas de ``df_oro``.

    ``df_oro`` debe tener las columnas texto, tema y sentimiento.
    """
    faltan = {"texto", "tema", "sentimiento"} - set(df_oro.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    if df_oro.empty:
        raise ValueError("El conjunto de validación está vacío")
    pred = df_oro.copy()
    pred["sent_pred"] = pred["texto"].map(lambda t: etiqueta_sentimiento(puntaje_sentimiento(t)))
    pred["tema_pred"] = pred["texto"].map(clasificar_tema)
    errores = pred[(pred["sent_pred"] != pred["sentimiento"])
                   | (pred["tema_pred"] != pred["tema"])]
    return MetricasValidacion(
        n=len(pred),
        exactitud_sentimiento=float(accuracy_score(pred["sentimiento"], pred["sent_pred"])),
        f1_sentimiento=float(f1_score(pred["sentimiento"], pred["sent_pred"],
                                      average="macro", zero_division=0)),
        exactitud_tema=float(accuracy_score(pred["tema"], pred["tema_pred"])),
        f1_tema=float(f1_score(pred["tema"], pred["tema_pred"],
                               average="macro", zero_division=0)),
        matriz_sentimiento=pd.crosstab(pred["sentimiento"], pred["sent_pred"],
                                       rownames=["real"], colnames=["predicho"]),
        errores=errores.reset_index(drop=True),
    )
