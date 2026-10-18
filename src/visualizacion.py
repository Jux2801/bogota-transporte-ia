"""Gráficos de resultados (matplotlib, backend sin pantalla)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .config import ORDEN_FRANJAS  # noqa: E402
from .propuestas import explotar_rutas  # noqa: E402

AZUL, AMARILLO, GRIS, ROJO, VERDE = "#0A3B8C", "#F2B705", "#8A94A6", "#C0392B", "#2E8B57"
ETIQ_TEMAS = {
    "tiempos_espera": "Tiempos de espera", "hacinamiento": "Hacinamiento",
    "inseguridad": "Inseguridad", "fallas_operativas": "Fallas operativas",
    "tarifa_pago": "Tarifa y pago", "estaciones_servicio": "Estaciones/servicio",
    "otros": "Otros",
}
ETIQ_FRANJAS = {"madrugada": "Madrugada\n(0-5 h)", "pico_am": "Pico AM\n(5-9 h)",
                "valle": "Valle\n(9-16 h)", "pico_pm": "Pico PM\n(16-20 h)",
                "noche": "Noche\n(20-24 h)"}


def _guardar(fig, ruta: str) -> str:
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=170)
    plt.close(fig)
    return ruta


def grafico_temas(df: pd.DataFrame, ruta: str) -> str:
    neg = df[df["sentimiento"] == "negativo"]["tema"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([ETIQ_TEMAS.get(t, t) for t in neg.index], neg.values, color=AZUL)
    for y, v in enumerate(neg.values):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("Comentarios negativos")
    ax.set_title("Temas de queja", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _guardar(fig, ruta)


def grafico_sentimiento_franja(df: pd.DataFrame, ruta: str) -> str:
    ct = pd.crosstab(df["franja"], df["sentimiento"]).reindex(ORDEN_FRANJAS).fillna(0)
    for col in ("negativo", "neutro", "positivo"):
        if col not in ct:
            ct[col] = 0
    ct = ct[["negativo", "neutro", "positivo"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    base = pd.Series(0.0, index=ct.index)
    for col, color in zip(ct.columns, (ROJO, GRIS, VERDE)):
        ax.bar([ETIQ_FRANJAS[i] for i in ct.index], ct[col], bottom=base,
               color=color, label=col.capitalize())
        base += ct[col]
    ax.set_ylabel("Comentarios")
    ax.set_title("Sentimiento por franja horaria", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _guardar(fig, ruta)


def grafico_ranking(ranking: pd.DataFrame, ruta: str) -> str:
    r = ranking.sort_values("criticidad")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(r["ruta"], r["criticidad"], color=AMARILLO, edgecolor=AZUL)
    for y, v in enumerate(r["criticidad"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Índice de criticidad (0-1)")
    ax.set_title("Ranking de corredores", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _guardar(fig, ruta)


def mapa_calor(df: pd.DataFrame, ruta: str, top: int = 8) -> str:
    base = explotar_rutas(df)
    neg = base[(base["sentimiento"] == "negativo") & (base["franja"] != "desconocida")]
    tabla = pd.crosstab(neg["ruta"], neg["franja"]).reindex(columns=ORDEN_FRANJAS).fillna(0)
    tabla = tabla.loc[tabla.sum(axis=1).sort_values(ascending=False).index[:top]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(tabla.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(tabla.columns)),
                  [ETIQ_FRANJAS[c] for c in tabla.columns], fontsize=8)
    ax.set_yticks(range(len(tabla.index)), tabla.index, fontsize=8)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            ax.text(j, i, int(tabla.values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Quejas por corredor y franja", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _guardar(fig, ruta)
