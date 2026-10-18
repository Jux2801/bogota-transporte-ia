"""Pipeline completo: datos -> preprocesamiento -> análisis -> propuestas.

Uso:
    python -m src.main --generar-muestra            # crea data/muestra_comentarios.csv
    python -m src.main                              # analiza la muestra
    python -m src.main --entrada data/mis_datos.csv # analiza datos propios
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from . import visualizacion as viz
from .analisis import analizar_dataframe, descubrir_temas
from .muestra import generar_muestra
from .preprocesamiento import preprocesar
from .propuestas import generar_propuestas, indice_criticidad
from .recoleccion import cargar_csv, guardar_csv
from .validacion import evaluar

RUTA_MUESTRA = "data/muestra_comentarios.csv"
RUTA_ORO = "data/validacion_manual.csv"


def ejecutar(entrada: str, salida: str, semilla: int = 42) -> dict:
    """Ejecuta el pipeline y devuelve un diccionario con los resultados."""
    crudo = cargar_csv(entrada)
    df = preprocesar(crudo)
    analizado = analizar_dataframe(df)
    ranking = indice_criticidad(analizado)
    propuestas = generar_propuestas(analizado, top=5)
    clusters = descubrir_temas(analizado["texto_limpio"].tolist(), k=6, semilla=semilla)

    os.makedirs(salida, exist_ok=True)
    fig = os.path.join(salida, "figuras")
    viz.grafico_temas(analizado, os.path.join(fig, "temas.png"))
    viz.grafico_sentimiento_franja(analizado, os.path.join(fig, "sentimiento_franja.png"))
    viz.grafico_ranking(ranking, os.path.join(fig, "ranking.png"))
    viz.mapa_calor(analizado, os.path.join(fig, "mapa_calor.png"))

    salida_df = analizado.assign(rutas=analizado["rutas"].map(" | ".join))
    salida_df.to_csv(os.path.join(salida, "comentarios_analizados.csv"), index=False)
    ranking.to_csv(os.path.join(salida, "ranking_rutas.csv"), index=False)
    propuestas.to_csv(os.path.join(salida, "propuestas.csv"), index=False)
    return {"crudo": crudo, "analizado": analizado, "ranking": ranking,
            "propuestas": propuestas, "clusters": clusters}


def imprimir_resumen(r: dict) -> None:
    a = r["analizado"]
    print("=" * 62)
    print("ANÁLISIS DE COMENTARIOS - TRANSPORTE PÚBLICO DE BOGOTÁ")
    print("=" * 62)
    print(f"Comentarios crudos:        {len(r['crudo'])}")
    print(f"Comentarios tras limpieza: {len(a)}  (duplicados/vacíos eliminados: "
          f"{len(r['crudo']) - len(a)})")
    print("\nSentimiento:")
    print((a["sentimiento"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nTemas en comentarios negativos:")
    print(a[a["sentimiento"] == "negativo"]["tema"].value_counts().to_string())
    esp = a["espera_min"].dropna()
    print(f"\nEsperas mencionadas: {len(esp)} (media {esp.mean():.1f} min)")
    print("\nRanking de corredores (top 5):")
    print(r["ranking"].head(5)[["ruta", "menciones", "prop_negativa", "criticidad"]]
          .round(2).to_string(index=False))
    print("\nPropuestas:")
    for f in r["propuestas"].itertuples():
        print(f" {f.prioridad}. {f.ruta} [{f.franja_critica}] -> {f.problema_dominante}")
    c = r["clusters"]
    print(f"\nK-means (k={len(c.terminos)}), silueta: "
          f"{c.silueta:.2f}" if c.silueta is not None else "\nK-means: sin silueta")
    for k, t in c.terminos.items():
        print(f" cluster {k}: {', '.join(t[:4])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", default=RUTA_MUESTRA)
    p.add_argument("--salida", default="resultados")
    p.add_argument("--generar-muestra", action="store_true")
    p.add_argument("--comparar-anexo", action="store_true",
                   help="contrasta la línea base del Anexo (spaCy) con esta versión")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    if args.generar_muestra:
        guardar_csv(generar_muestra(args.n), RUTA_MUESTRA)
        print(f"Muestra simulada guardada en {RUTA_MUESTRA}")
    resultados = ejecutar(args.entrada, args.salida, args.semilla)
    imprimir_resumen(resultados)
    if args.comparar_anexo:
        from .comparacion import comparar_con_anexo
        c = comparar_con_anexo(resultados["analizado"]["texto"].tolist(),
                               resultados["analizado"])
        print("\nComparación con el código base del Anexo (spaCy):")
        print(f" lugares distintos contados:      {c['lugares_anexo']}")
        print(f" propuestas del Anexo (>5):       {c['propuestas_anexo']}")
        print(f"   - sobre algo que no es corredor: {c['sin_corredor']}")
        print(f"   - repetidas para un mismo corredor: {c['duplicadas']}")
        print(f"   - corredores realmente cubiertos: {c['corredores_unicos']}")
        print(f" propuestas de la versión mejorada: {c['propuestas_mejoradas']}")
        print(f" cobertura de spaCy sobre corredores: {c['cobertura_spacy']:.0%}")
    if os.path.exists(RUTA_ORO):
        m = evaluar(pd.read_csv(RUTA_ORO))
        print(f"\nValidación manual (n={m.n}): exactitud sentimiento "
              f"{m.exactitud_sentimiento:.2f} | exactitud tema {m.exactitud_tema:.2f}")


if __name__ == "__main__":
    main()
