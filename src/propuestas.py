"""Índice de criticidad por corredor y generación de propuestas operativas."""
from __future__ import annotations

import pandas as pd

# Acciones operativas (no requieren obra civil) según el problema dominante.
ACCIONES: dict[str, str] = {
    "tiempos_espera": (
        "Rebalancear la flota: trasladar buses de la franja valle a la franja crítica "
        "y acortar los intervalos de despacho en el corredor."
    ),
    "hacinamiento": (
        "Programar servicios expresos o de salida intercalada en la franja crítica "
        "y reforzar el control de ingreso para repartir la carga entre estaciones."
    ),
    "inseguridad": (
        "Reforzar la presencia de personal de seguridad y gestores de convivencia "
        "en las estaciones y horarios donde se concentran los reportes."
    ),
    "fallas_operativas": (
        "Priorizar mantenimiento preventivo y activar alertas tempranas de "
        "desvíos/cierres por los canales digitales del sistema."
    ),
    "tarifa_pago": (
        "Revisar la disponibilidad de puntos de recarga y comunicar con claridad "
        "las reglas de transbordo en las estaciones afectadas."
    ),
    "estaciones_servicio": (
        "Mejorar información al usuario (pantallas, aplicación) y reforzar "
        "protocolos de aseo y trato en las estaciones señaladas."
    ),
    "otros": "Monitorear: no hay un problema dominante identificable.",
}


def normalizar_min_max(serie: pd.Series) -> pd.Series:
    """Escala una serie al intervalo [0, 1]; si es constante devuelve ceros."""
    s = pd.to_numeric(serie, errors="coerce").astype(float)
    rango = s.max() - s.min()
    if pd.isna(rango) or rango == 0:
        return pd.Series(0.0, index=serie.index)
    return (s - s.min()) / rango


def explotar_rutas(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (comentario, corredor); descarta comentarios sin corredor."""
    out = df[df["rutas"].map(len) > 0].explode("rutas")
    return out.rename(columns={"rutas": "ruta"}).reset_index(drop=True)


def indice_criticidad(df: pd.DataFrame, pesos=(0.4, 0.4, 0.2)) -> pd.DataFrame:
    """Calcula el índice de criticidad por corredor, en [0, 1].

    criticidad = w1·volumen_norm + w2·proporción_negativa + w3·espera_norm
    donde volumen_norm y espera_norm se escalan min-max entre corredores.
    """
    if abs(sum(pesos) - 1.0) > 1e-9:
        raise ValueError("Los pesos deben sumar 1")
    base = explotar_rutas(df)
    if base.empty:
        return pd.DataFrame(columns=[
            "ruta", "menciones", "prop_negativa", "sentimiento_medio",
            "espera_media_min", "criticidad",
        ])
    g = base.groupby("ruta")
    res = pd.DataFrame({
        "menciones": g.size(),
        "prop_negativa": g["sentimiento"].apply(lambda s: (s == "negativo").mean()),
        "sentimiento_medio": g["sentimiento_puntaje"].mean(),
        "espera_media_min": g["espera_min"].mean(),
    }).reset_index()
    w1, w2, w3 = pesos
    res["criticidad"] = (
        w1 * normalizar_min_max(res["menciones"])
        + w2 * res["prop_negativa"]
        + w3 * normalizar_min_max(res["espera_media_min"].fillna(0))
    )
    return res.sort_values("criticidad", ascending=False).reset_index(drop=True)


def franja_critica(df: pd.DataFrame, ruta: str) -> str:
    """Franja con más comentarios negativos para un corredor ('desconocida' si no hay)."""
    base = explotar_rutas(df)
    sel = base[(base["ruta"] == ruta) & (base["sentimiento"] == "negativo")
               & (base["franja"] != "desconocida")]
    if sel.empty:
        return "desconocida"
    return str(sel["franja"].value_counts().idxmax())


def tema_dominante(df: pd.DataFrame, ruta: str) -> str:
    """Tema más frecuente entre los comentarios negativos de un corredor."""
    base = explotar_rutas(df)
    sel = base[(base["ruta"] == ruta) & (base["sentimiento"] == "negativo")
               & (base["tema"] != "otros")]
    if sel.empty:
        return "otros"
    return str(sel["tema"].value_counts().idxmax())


def generar_propuestas(df: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Genera propuestas para los ``top`` corredores más críticos."""
    if top < 1:
        raise ValueError("top debe ser >= 1")
    ranking = indice_criticidad(df).head(top)
    filas = []
    for pos, fila in enumerate(ranking.itertuples(index=False), start=1):
        tema = tema_dominante(df, fila.ruta)
        filas.append({
            "prioridad": pos,
            "ruta": fila.ruta,
            "criticidad": round(float(fila.criticidad), 3),
            "franja_critica": franja_critica(df, fila.ruta),
            "problema_dominante": tema,
            "accion_propuesta": ACCIONES[tema],
            "requiere_obra_civil": False,
        })
    return pd.DataFrame(filas)
