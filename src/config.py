"""Configuración y recursos léxicos del proyecto.

Todo el conocimiento de dominio (corredores de TransMilenio, temas de queja,
léxico de sentimiento) vive aquí para poder ajustarlo sin tocar la lógica.
"""
from __future__ import annotations

import unicodedata


def _sin_tildes(texto: str) -> str:
    """Quita tildes (se usa aquí para evitar importar otros módulos)."""
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------
# Franjas horarias: (hora_inicio, hora_fin_exclusiva)
# --------------------------------------------------------------------------
FRANJAS: dict[str, tuple[int, int]] = {
    "madrugada": (0, 5),
    "pico_am": (5, 9),
    "valle": (9, 16),
    "pico_pm": (16, 20),
    "noche": (20, 24),
}
ORDEN_FRANJAS = ["madrugada", "pico_am", "valle", "pico_pm", "noche"]

# --------------------------------------------------------------------------
# Corredores: nombre canónico -> alias (sin tildes, en minúscula)
# --------------------------------------------------------------------------
RUTAS: dict[str, list[str]] = {
    "Troncal Caracas": ["caracas", "troncal caracas"],
    "Autopista Norte": ["autonorte", "autopista norte", "portal norte"],
    "Calle 80": ["calle 80", "portal 80"],
    "Avenida Américas": ["americas", "avenida americas", "portal americas"],
    "NQS": ["nqs"],
    "Suba": ["suba", "portal suba"],
    "Usme": ["usme", "portal usme"],
    "Tunal": ["tunal", "portal tunal"],
    "Calle 26 / El Dorado": ["calle 26", "el dorado", "portal eldorado"],
    "Autopista Sur": ["autosur", "autopista sur", "portal sur", "soacha"],
}

# --------------------------------------------------------------------------
# Temas de queja. Entradas terminadas en '*' son prefijos; las demás, palabras
# o frases exactas. Todo sin tildes.
# --------------------------------------------------------------------------
TEMAS: dict[str, list[str]] = {
    "tiempos_espera": [
        "espera*", "esper*", "demora*", "demor*", "tarda*", "tarde", "frecuencia",
        "no pasa", "no llega*", "nunca llega*", "minutos", "min", "pocos buses",
        "eterna", "rapido", "puntual*",
    ],
    "hacinamiento": [
        "lleno", "llena", "llenos", "llenas", "atesta*", "abarrot*", "apret*",
        "sardinas", "no cabe*", "aglomera*", "hacina*", "empuj*", "full",
    ],
    "inseguridad": [
        "rob*", "atrac*", "ladron*", "acoso", "acosa*", "raponer*", "cosquill*",
        "inseguridad", "inseguro*", "pelea*", "hurto*", "peligro*", "vigilancia",
        "miedo*", "iluminad*",
    ],
    "fallas_operativas": [
        "vara*", "dana*", "dano*", "averi*", "falla*", "apag*", "cerrad*", "cerraron",
        "cierre*", "desvio*", "bloque*", "manifesta*", "accidente*",
        "sin servicio", "suspend*", "no funciona*", "ascensor*",
    ],
    "tarifa_pago": [
        "tarifa*", "pasaje*", "tu llave", "recarga*", "recargar", "cobr*",
        "precio*", "caro*", "colados", "colarse", "saldo", "tarjeta*",
        "transbordo*",
    ],
    "estaciones_servicio": [
        "sucia*", "sucio*", "aseo", "limpi*", "senal*", "informacion", "pantalla*",
        "aviso*", "app", "aplicacion", "conductor*", "grosero*", "maltrat*",
        "amabilidad", "rampa*", "accesib*", "atencion", "personal",
    ],
}
# Orden de desempate (el primero gana).
PRIORIDAD_TEMAS = list(TEMAS.keys())

# --------------------------------------------------------------------------
# Léxico de sentimiento (sin tildes). Peso: +1 / -1; extremos ±2.
# --------------------------------------------------------------------------
_NEGATIVAS = {
    "demora": -1, "demoras": -1, "demorado": -1, "demoro": -1, "lento": -1, "lenta": -1,
    "horrible": -2, "pesimo": -2, "pesima": -2, "malo": -1, "mala": -1, "terrible": -2,
    "caos": -2, "caotico": -2, "caotica": -2, "lleno": -1, "llena": -1, "llenos": -1,
    "atestado": -1, "abarrotado": -1, "abarrotados": -1, "imposible": -1, "peligroso": -1,
    "peligro": -1, "robo": -2, "robos": -2, "robaron": -2, "atraco": -2, "atracaron": -2,
    "ladrones": -2, "acoso": -2, "sucio": -1, "sucias": -1, "sucia": -1, "danado": -1,
    "danados": -1, "danadas": -1, "varado": -1, "varados": -1, "colapso": -2,
    "colapsado": -2, "cansado": -1, "harto": -2, "harta": -2, "inseguridad": -2,
    "inseguro": -1, "colados": -1, "verguenza": -1, "deficiente": -1, "ineficiente": -1,
    "desastre": -2, "insoportable": -2, "incomodo": -1, "eterna": -1, "eterno": -1,
    "grosero": -1, "caro": -1, "caros": -1, "pena": -1, "tragedia": -2, "peor": -2,
    "abandono": -1, "desorden": -1, "falla": -1, "fallas": -1, "sardinas": -1,
    "empujan": -1, "esperando": -1, "espere": -1, "tarde": -1, "tocaron": 0,
    "queja": -1, "inaceptable": -2, "estres": -1, "estresante": -1,
    "miedo": -1, "abuso": -2, "fatal": -2, "mal": -1, "lata": -1, "botado": -1,
    "botados": -1, "maltrata": -1, "maltrato": -1, "bloqueo": -1, "odio": -2,
}
_POSITIVAS = {
    "rapido": 1, "rapida": 1, "puntual": 1, "bueno": 1, "buena": 1, "excelente": 2,
    "genial": 2, "limpio": 1, "limpia": 1, "comodo": 1, "comoda": 1, "seguro": 1,
    "segura": 1, "eficiente": 1, "mejoro": 1, "mejor": 1, "fluido": 1, "agradable": 1,
    "gracias": 1, "bien": 1, "util": 1, "agil": 1, "tranquilo": 1, "silenciosos": 1,
    "silencioso": 1, "amable": 1, "felicitaciones": 2, "perfecto": 2, "increible": 1,
    "ordenado": 1, "ordenada": 1,
}
LEXICO_SENTIMIENTO: dict[str, int] = {
    **{_sin_tildes(k): v for k, v in _NEGATIVAS.items() if v != 0},
    **{_sin_tildes(k): v for k, v in _POSITIVAS.items()},
}
NEGADORES = {"no", "nunca", "jamas", "ni", "tampoco", "sin"}
INTENSIFICADORES = {"muy", "super", "demasiado", "tan", "bastante", "realmente",
                    "totalmente", "tremendamente"}

UMBRAL_SENTIMIENTO = 0.05
MAX_ESPERA_MIN = 240

# --------------------------------------------------------------------------
# Palabras vacías del español (lista compacta, sin dependencias externas)
# --------------------------------------------------------------------------
_STOP = """
a al algo algun alguna algunas alguno algunos ante antes asi aun aunque cada
casi como con contra cual cuales cuando de del desde donde dos el ella ellas
ello ellos en entre era eran es esa esas ese eso esos esta estaba estaban
estan estar este esto estos fue fueron ha han hasta hay la las le les lo los
mas me mi mis mucho muchos muy nada ni no nos nuestra nuestro o os otra otras
otro otros para pero poco por porque que quien quienes se sea segun ser si sin
sobre su sus tal tambien tan tanto te tengo tiene tienen todo todos tu tus un
una uno unos usted ustedes va van voy y ya yo
"""
STOPWORDS: frozenset[str] = frozenset(_STOP.split())
