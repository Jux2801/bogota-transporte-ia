"""Generador de una muestra SIMULADA de comentarios sobre el transporte de Bogotá.

IMPORTANTE: los textos de este módulo son sintéticos. Sirven para ejecutar y
probar el pipeline sin depender de redes sociales (que exigen inicio de sesión
y restringen la extracción automática). Los resultados obtenidos con esta
muestra NO describen la opinión real de los usuarios: para eso hay que
reemplazarla por datos recolectados con ``recoleccion.py``.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

from .config import FRANJAS, RUTAS

# Probabilidad de que un comentario sobre cada corredor sea una queja.
P_QUEJA = {
    "Autopista Norte": 0.80, "Calle 80": 0.75, "Avenida Américas": 0.72,
    "Suba": 0.80, "Usme": 0.78, "Autopista Sur": 0.70, "Troncal Caracas": 0.65,
    "NQS": 0.55, "Tunal": 0.50, "Calle 26 / El Dorado": 0.45,
}
PESO_RUTA = {
    "Autopista Norte": 17, "Calle 80": 14, "Avenida Américas": 13, "Suba": 14,
    "Usme": 10, "Autopista Sur": 10, "Troncal Caracas": 12, "NQS": 5, "Tunal": 3,
    "Calle 26 / El Dorado": 2,
}
# Temas de queja más probables por corredor (peso por tema).
TEMA_POR_RUTA = {
    "Suba": {"tiempos_espera": 5, "hacinamiento": 4, "fallas_operativas": 1},
    "Usme": {"tiempos_espera": 5, "hacinamiento": 3, "inseguridad": 2},
    "Calle 80": {"hacinamiento": 5, "fallas_operativas": 3, "tiempos_espera": 2},
    "Troncal Caracas": {"inseguridad": 4, "hacinamiento": 4, "estaciones_servicio": 2},
    "Autopista Norte": {"tiempos_espera": 5, "fallas_operativas": 3, "hacinamiento": 2},
}
TEMA_DEFECTO = {"tiempos_espera": 3, "hacinamiento": 3, "fallas_operativas": 2,
                "inseguridad": 2, "tarifa_pago": 2, "estaciones_servicio": 2}
FRANJA_POR_TEMA = {
    "tiempos_espera": {"pico_am": 40, "pico_pm": 40, "valle": 10, "noche": 8, "madrugada": 2},
    "hacinamiento": {"pico_am": 45, "pico_pm": 45, "valle": 6, "noche": 3, "madrugada": 1},
    "inseguridad": {"noche": 30, "pico_pm": 30, "valle": 20, "pico_am": 15, "madrugada": 5},
    "fallas_operativas": {"pico_am": 25, "pico_pm": 25, "valle": 30, "noche": 15, "madrugada": 5},
    "tarifa_pago": {"pico_am": 30, "pico_pm": 25, "valle": 35, "noche": 8, "madrugada": 2},
    "estaciones_servicio": {"pico_am": 20, "pico_pm": 25, "valle": 40, "noche": 12, "madrugada": 3},
}

PLANTILLAS_QUEJA = {
    "tiempos_espera": [
        "Llevo {m} minutos esperando el bus en {r} y no pasa nada",
        "La frecuencia en {r} es pésima, tocó esperar {m} minutos",
        "Otra vez demoras en {r}, nunca llega el articulado",
        "Saliendo de la oficina y en {r} la espera es eterna, {m} min parado",
        "No entiendo por qué en {r} pasan tan pocos buses en hora pico",
        "Hoy en {r} esperé {m} minutos para poder subirme, qué desastre",
    ],
    "hacinamiento": [
        "{r} full, no cabe ni un alfiler, viajamos como sardinas",
        "Imposible subirse en {r}, el bus llega lleno y empujan",
        "Aglomeración total en las estaciones de {r}, esto es insoportable",
        "Los articulados de {r} van abarrotados, uno no respira",
        "Tuve que dejar pasar tres buses llenos en {r}",
        "Estación de {r} colapsada otra vez, gente apretada por todos lados",
    ],
    "inseguridad": [
        "Hoy me robaron el celular en {r}, cada vez más inseguro",
        "Cuidado con los raponeros en las estaciones de {r}",
        "Acoso en el bus por {r} y nadie hace nada",
        "Atracaron a una señora en {r}, no hay vigilancia",
        "Qué peligro bajarse de noche en {r}",
        "Los cosquilleros en {r} están peor que nunca",
    ],
    "fallas_operativas": [
        "Bus varado en {r} y toda la troncal parada",
        "Cerraron la estación de {r} sin avisar, qué desorden",
        "Los ascensores de {r} dañados otra vez",
        "Manifestación bloqueó {r}, sin servicio por una hora",
        "Falla eléctrica en {r}, estación apagada y sin aviso",
        "Accidente en {r} y desvío sin información, terrible",
    ],
    "tarifa_pago": [
        "Subieron el pasaje a $3.550 y el servicio en {r} sigue igual de malo",
        "La tarjeta Tu Llave no recarga en {r}, perdí el transbordo",
        "Cada vez más colados en {r} y uno pagando pasaje caro",
        "Pasaje caro para el servicio que dan en {r}",
    ],
    "estaciones_servicio": [
        "Estaciones sucias en {r}, da pena",
        "Conductor grosero en {r} hoy, maltrata a los usuarios",
        "Sin información de tiempos en {r}, la aplicación no sirve",
        "Sin rampas en {r} para personas con movilidad reducida",
    ],
}
PLANTILLAS_POSITIVAS = [
    "Hoy {r} estuvo rápido y puntual, gracias",
    "Buen servicio en {r}, bus limpio y cómodo",
    "Los buses eléctricos de {r} son silenciosos y agradables",
    "Excelente la frecuencia en {r} esta mañana",
    "Por fin llegó rápido el bus en {r}, todo fluido",
    "Muy buena la atención del personal en {r}, felicitaciones",
]
PLANTILLAS_NEUTRAS = [
    "¿Alguien sabe cuánto dura el recorrido por {r} hasta el centro?",
    "Mañana hay cambio de ruta en {r}, revisen la información oficial",
    "Pregunta: ¿cuál es el horario de {r} los domingos?",
    "Hoy pasé por {r} a las ocho y estaba normal",
]
RUIDO_SIN_RUTA = [
    "Feliz domingo a todos", "Vendo bicicleta, interesados al DM",
    "Qué buen partido el de anoche", "Buscando apartamento por el centro",
]
SUFIJOS = ["", "", "", " 😡", " 😤", " 🙄", " #TransMilenio", " #Bogotá",
           " @TransMilenio", " https://t.co/abc123", " !!!", " 🚌"]

PREFIJOS = ["", "", "", "Hoy ", "Ayer ", "Otra vez: ", "Urgente: ", "Increíble, ",
            "Qué pereza: ", "Reporte: ", "Ojo: ", "Buenas tardes, "]
FUENTES = ["red_social", "foro"]
INICIO = datetime(2026, 9, 1)
DIAS = 40


def _elegir(rng: random.Random, pesos: dict[str, float]) -> str:
    return rng.choices(list(pesos), weights=list(pesos.values()), k=1)[0]


def _fecha(rng: random.Random, franja: str) -> datetime:
    ini, fin = FRANJAS[franja]
    dia = rng.randrange(DIAS)
    return INICIO + timedelta(days=dia, hours=rng.randrange(ini, fin),
                              minutes=rng.randrange(60), seconds=rng.randrange(60))


def generar_muestra(n: int = 600, semilla: int = 2026) -> pd.DataFrame:
    """Genera ``n`` comentarios simulados de forma reproducible (misma semilla)."""
    if n < 1:
        raise ValueError("n debe ser >= 1")
    rng = random.Random(semilla)
    filas = []
    for i in range(n):
        if rng.random() < 0.04:  # comentarios sin relación con el transporte
            texto, franja = rng.choice(RUIDO_SIN_RUTA), _elegir(rng, FRANJA_POR_TEMA["tarifa_pago"])
        else:
            ruta = _elegir(rng, PESO_RUTA)
            alias = rng.choice(RUTAS[ruta])
            nombre = alias.title() if alias != "nqs" else "NQS"
            m = rng.randint(8, 90)
            u = rng.random()
            if u < P_QUEJA[ruta]:
                tema = _elegir(rng, TEMA_POR_RUTA.get(ruta, TEMA_DEFECTO))
                plantilla = rng.choice(PLANTILLAS_QUEJA[tema])
                franja = _elegir(rng, FRANJA_POR_TEMA[tema])
            elif u < P_QUEJA[ruta] + (1 - P_QUEJA[ruta]) * 0.55:
                plantilla = rng.choice(PLANTILLAS_POSITIVAS)
                franja = _elegir(rng, FRANJA_POR_TEMA["estaciones_servicio"])
            else:
                plantilla = rng.choice(PLANTILLAS_NEUTRAS)
                franja = _elegir(rng, FRANJA_POR_TEMA["tarifa_pago"])
            texto = plantilla.format(r=nombre, m=m)
            pref = rng.choice(PREFIJOS)
            if pref:
                texto = pref + texto[0].lower() + texto[1:]
        texto += rng.choice(SUFIJOS)
        filas.append({
            "id": f"sim-{i:05d}",
            "fuente": rng.choice(FUENTES),
            "fecha_hora": _fecha(rng, franja).strftime("%Y-%m-%d %H:%M:%S"),
            "texto": texto,
        })
    # Repetidos exactos (retuits/copias), ~3 %, para ejercitar la deduplicación.
    for _ in range(max(1, n // 33)):
        copia = dict(rng.choice(filas))
        copia["id"] = f"sim-dup-{len(filas):05d}"
        filas.append(copia)
    return pd.DataFrame(filas)
