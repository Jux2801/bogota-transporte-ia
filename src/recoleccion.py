"""Recolección automatizada de comentarios con Selenium.

Principios de diseño:
* Solo se recolecta contenido público y se respeta ``robots.txt``.
* Pausa entre páginas para no sobrecargar el sitio.
* El navegador se inyecta (``driver``) para poder probar sin red ni Chrome.
* No se guardan nombres de usuario: solo texto, fecha y fuente (minimización
  de datos, Ley 1581 de 2012 de Colombia).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, WebDriverException,
)
from selenium.webdriver.common.by import By

from .preprocesamiento import COLUMNAS_OBLIGATORIAS, validar_columnas

LOG = logging.getLogger(__name__)
USER_AGENT = "UNIMINUTO-EstudioCaso-IA/1.0 (uso academico)"


@dataclass
class ConfigFuente:
    """Describe cómo extraer comentarios de un sitio concreto."""
    nombre: str
    url_base: str                      # admite el marcador {pagina}
    selector_item: str                 # CSS de cada comentario
    selector_texto: str | None = None  # CSS relativo; None = texto del item
    selector_fecha: str | None = None  # CSS relativo; None = sin fecha
    atributo_fecha: str = "datetime"
    max_paginas: int = 3
    pausa_segundos: float = 2.0
    espera_max: int = 10


def robots_permite(url: str, agente: str = USER_AGENT,
                   lector: robotparser.RobotFileParser | None = None) -> bool:
    """True si robots.txt permite a ``agente`` leer ``url``.

    Ante cualquier error al leer robots.txt se devuelve False (criterio
    conservador: ante la duda, no se recolecta).
    """
    partes = urlparse(url)
    if partes.scheme not in ("http", "https") or not partes.netloc:
        return False
    try:
        if lector is None:
            lector = robotparser.RobotFileParser()
            lector.set_url(f"{partes.scheme}://{partes.netloc}/robots.txt")
            lector.read()
        return bool(lector.can_fetch(agente, url))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo => no recolectar
        LOG.warning("No se pudo evaluar robots.txt (%s): %s", url, exc)
        return False


def construir_driver(headless: bool = True):
    """Crea un Chrome controlado por Selenium (Selenium Manager descarga el driver)."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opciones = Options()
    if headless:
        opciones.add_argument("--headless=new")
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-dev-shm-usage")
    opciones.add_argument(f"--user-agent={USER_AGENT}")
    return webdriver.Chrome(options=opciones)


def _esperar_items(driver, cfg: ConfigFuente) -> None:
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    WebDriverWait(driver, cfg.espera_max).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, cfg.selector_item))
    )


def _extraer_item(item, cfg: ConfigFuente) -> tuple[str, str | None]:
    """Devuelve (texto, fecha) de un elemento web; texto vacío si falla."""
    try:
        elem_txt = (item.find_element(By.CSS_SELECTOR, cfg.selector_texto)
                    if cfg.selector_texto else item)
        texto = (elem_txt.text or "").strip()
    except NoSuchElementException:
        return "", None
    fecha = None
    if cfg.selector_fecha:
        try:
            elem_f = item.find_element(By.CSS_SELECTOR, cfg.selector_fecha)
            fecha = elem_f.get_attribute(cfg.atributo_fecha) or elem_f.text or None
        except NoSuchElementException:
            fecha = None
    return texto, fecha


def recolectar_fuente(driver, cfg: ConfigFuente, verificar_robots: bool = True,
                      esperar: Callable = _esperar_items,
                      dormir: Callable[[float], None] = time.sleep) -> pd.DataFrame:
    """Recorre las páginas de una fuente y devuelve un DataFrame de comentarios."""
    if cfg.max_paginas < 1:
        raise ValueError("max_paginas debe ser >= 1")
    if verificar_robots and not robots_permite(cfg.url_base.format(pagina=1)):
        LOG.warning("robots.txt no permite recolectar %s; se omite", cfg.nombre)
        return pd.DataFrame(columns=list(COLUMNAS_OBLIGATORIAS))

    filas: list[dict] = []
    for pagina in range(1, cfg.max_paginas + 1):
        url = cfg.url_base.format(pagina=pagina)
        try:
            driver.get(url)
            esperar(driver, cfg)
        except (TimeoutException, WebDriverException) as exc:
            LOG.warning("Página %s sin contenido o con error: %s", url, exc)
            break
        items = driver.find_elements(By.CSS_SELECTOR, cfg.selector_item)
        if not items:
            break
        for i, item in enumerate(items):
            texto, fecha = _extraer_item(item, cfg)
            if texto:
                filas.append({
                    "id": f"{cfg.nombre}-{pagina}-{i}",
                    "fuente": cfg.nombre,
                    "fecha_hora": fecha,
                    "texto": texto,
                })
        if pagina < cfg.max_paginas:
            dormir(cfg.pausa_segundos)
    return pd.DataFrame(filas, columns=list(COLUMNAS_OBLIGATORIAS))


def recolectar(configs: list[ConfigFuente], driver=None, **kwargs) -> pd.DataFrame:
    """Recolecta varias fuentes y concatena los resultados.

    Si no se entrega ``driver`` se crea uno y se cierra al terminar.
    """
    propio = driver is None
    if propio:
        driver = construir_driver()
    try:
        partes = [recolectar_fuente(driver, c, **kwargs) for c in configs]
    finally:
        if propio:
            driver.quit()
    partes = [p for p in partes if not p.empty]
    if not partes:
        return pd.DataFrame(columns=list(COLUMNAS_OBLIGATORIAS))
    return pd.concat(partes, ignore_index=True)


def guardar_csv(df: pd.DataFrame, ruta: str) -> None:
    """Guarda comentarios en CSV UTF-8 validando las columnas."""
    validar_columnas(df)
    df.to_csv(ruta, index=False, encoding="utf-8")


def cargar_csv(ruta: str) -> pd.DataFrame:
    """Lee un CSV de comentarios y valida las columnas obligatorias."""
    df = pd.read_csv(ruta, encoding="utf-8", dtype=str)
    validar_columnas(df)
    return df
