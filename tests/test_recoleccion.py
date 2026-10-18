"""Pruebas del recolector con un navegador falso (sin red ni Chrome)."""
import pandas as pd
import pytest
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from src.recoleccion import (
    ConfigFuente, cargar_csv, guardar_csv, recolectar_fuente, robots_permite,
)


class Elemento:
    def __init__(self, texto="", hijos=None, attrs=None):
        self.text, self._hijos, self._attrs = texto, hijos or {}, attrs or {}

    def find_element(self, by, selector):
        if selector not in self._hijos:
            raise NoSuchElementException(selector)
        return self._hijos[selector]

    def get_attribute(self, nombre):
        return self._attrs.get(nombre)


class DriverFalso:
    def __init__(self, paginas):
        self.paginas, self.visitadas, self._actual = paginas, [], []

    def get(self, url):
        self.visitadas.append(url)
        pag = int(url.rsplit("=", 1)[1])
        if self.paginas.get(pag) == "error":
            raise TimeoutException("simulado")
        self._actual = self.paginas.get(pag, [])

    def find_elements(self, by, selector):
        return self._actual


def comentario(texto, fecha):
    return Elemento(hijos={
        ".txt": Elemento(texto),
        ".fecha": Elemento("", attrs={"datetime": fecha}),
    })


def _cfg(**kw):
    base = dict(nombre="foro_x", url_base="https://foro.test/hilo?pagina={pagina}",
                selector_item=".com", selector_texto=".txt", selector_fecha=".fecha",
                max_paginas=3, pausa_segundos=0.5)
    base.update(kw)
    return ConfigFuente(**base)


def _sin_espera(driver, cfg):
    return None


def test_recolecta_varias_paginas_y_respeta_pausa():
    d = DriverFalso({
        1: [comentario("Demora en Suba", "2026-09-01T07:00:00"), comentario("Bus lleno", "2026-09-01T08:00:00")],
        2: [comentario("Robo en la Caracas", "2026-09-02T19:00:00")],
        3: [],
    })
    pausas = []
    df = recolectar_fuente(d, _cfg(), verificar_robots=False, esperar=_sin_espera,
                           dormir=pausas.append)
    assert list(df.columns) == ["id", "fuente", "fecha_hora", "texto"]
    assert len(df) == 3 and df["id"].is_unique
    assert df.loc[0, "fecha_hora"] == "2026-09-01T07:00:00"
    assert pausas == [0.5, 0.5]       # pausa tras la pág. 1 y la 2; la 3 está vacía y corta
    assert len(d.visitadas) == 3


def test_omite_items_sin_texto():
    d = DriverFalso({1: [Elemento(hijos={}), comentario("Hola Suba", "2026-09-01T07:00:00")]})
    df = recolectar_fuente(d, _cfg(max_paginas=1), verificar_robots=False, esperar=_sin_espera)
    assert len(df) == 1


def test_error_de_pagina_corta_sin_perder_lo_recolectado():
    d = DriverFalso({1: [comentario("Uno", "2026-09-01T07:00:00")], 2: "error"})
    df = recolectar_fuente(d, _cfg(), verificar_robots=False, esperar=_sin_espera,
                           dormir=lambda s: None)
    assert len(df) == 1


def test_max_paginas_invalido():
    with pytest.raises(ValueError):
        recolectar_fuente(DriverFalso({}), _cfg(max_paginas=0), verificar_robots=False)


class LectorRobots:
    def __init__(self, permite):
        self.permite = permite

    def can_fetch(self, agente, url):
        return self.permite


class LectorRoto:
    def can_fetch(self, agente, url):
        raise OSError("sin red")


def test_robots_permite_y_bloquea():
    assert robots_permite("https://foro.test/a", lector=LectorRobots(True)) is True
    assert robots_permite("https://foro.test/a", lector=LectorRobots(False)) is False


def test_robots_ante_error_no_recolecta():
    assert robots_permite("https://foro.test/a", lector=LectorRoto()) is False


@pytest.mark.parametrize("url", ["ftp://x.test/a", "no-es-url", ""])
def test_robots_rechaza_urls_invalidas(url):
    assert robots_permite(url, lector=LectorRobots(True)) is False


def test_fuente_bloqueada_por_robots_devuelve_vacio(monkeypatch):
    monkeypatch.setattr("src.recoleccion.robots_permite", lambda *a, **k: False)
    d = DriverFalso({1: [comentario("x", "2026-09-01T07:00:00")]})
    df = recolectar_fuente(d, _cfg(), esperar=_sin_espera)
    assert df.empty and d.visitadas == []


def test_csv_conserva_ids_numericos_como_texto(tmp_path):
    df = pd.DataFrame({"id": ["007"], "fuente": ["foro"], "fecha_hora": ["2026-09-01 07:00:00"],
                       "texto": ["hola Suba"]})
    ruta = str(tmp_path / "ids.csv")
    guardar_csv(df, ruta)
    assert cargar_csv(ruta).loc[0, "id"] == "007"


def test_csv_ida_y_vuelta(tmp_path):
    df = pd.DataFrame({"id": ["1"], "fuente": ["foro"], "fecha_hora": ["2026-09-01 07:00:00"],
                       "texto": ["Ñandú en el Portal Américas, ¡qué caos!"]})
    ruta = str(tmp_path / "datos.csv")
    guardar_csv(df, ruta)
    pd.testing.assert_frame_equal(cargar_csv(ruta), df)


def test_cargar_csv_sin_columnas(tmp_path):
    ruta = tmp_path / "malo.csv"
    ruta.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_csv(str(ruta))
