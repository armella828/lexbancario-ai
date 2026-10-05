"""Descarga concurrente de URLs y limpieza de HTML (fase I/O-intensiva).

Usa asyncio + httpx porque la fase de descarga es I/O-bound: el proceso
libera el GIL mientras espera la red, de modo que una sola hebra puede
atender decenas de peticiones simultaneas. Un ThreadPoolExecutor tambien
funcionaria aqui, pero asyncio evita el coste de un hilo del sistema
operativo por peticion y hace explicito el limite de concurrencia.
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

TIMEOUT_SEGUNDOS = 25.0
MAX_REDIRECCIONES = 5
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

ETIQUETAS_NO_TEXTUALES = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "iframe", "svg", "button", "select",
]

ESPACIOS = re.compile(r"[ \t\r\f\v]+")
SALTOS = re.compile(r"\n{3,}")


@dataclass
class ResultadoDescarga:
    """Estado de una URL individual. Un fallo aqui no aborta el lote."""

    url: str
    ok: bool
    titulo: str = ""
    texto: str = ""
    codigo_http: Optional[int] = None
    bytes_html: int = 0
    error: Optional[str] = None
    segundos: float = 0.0
    enlaces: List[str] = field(default_factory=list)


def limpiar_html(html: str, url: str) -> tuple:
    """Extrae titulo y texto legible, descartando cromo de navegacion.

    Se eliminan las etiquetas de contenido no textual y ademas los bloques
    nav/header/footer/aside, que suelen aportar la mayor parte del ruido
    en paginas institucionales y que emborronarian los embeddings.
    """
    sopa = BeautifulSoup(html, "lxml")

    titulo = (sopa.title.get_text(strip=True) if sopa.title else "") or url

    for etiqueta in ETIQUETAS_NO_TEXTUALES:
        for nodo in sopa.find_all(etiqueta):
            nodo.decompose()

    texto = sopa.get_text(separator="\n")
    texto = ESPACIOS.sub(" ", texto)
    texto = SALTOS.sub("\n\n", texto)
    texto = texto.strip()

    enlaces = [
        a.get("href")
        for a in sopa.find_all("a", href=True)
        if a["href"].lower().split("?")[0].endswith(".pdf")
    ]

    return titulo.strip(), texto, enlaces


async def _descargar_una(
    cliente: httpx.AsyncClient,
    url: str,
    loop: asyncio.AbstractEventLoop,
) -> ResultadoDescarga:
    """Descarga y limpia una URL, capturando cualquier excepcion."""
    inicio = loop.time()
    try:
        respuesta = await cliente.get(url)
        if respuesta.status_code != 200:
            return ResultadoDescarga(
                url=url,
                ok=False,
                codigo_http=respuesta.status_code,
                error=f"HTTP {respuesta.status_code}",
                segundos=loop.time() - inicio,
            )

        html = respuesta.text
        titulo, texto, enlaces = limpiar_html(html, url)

        if not texto:
            return ResultadoDescarga(
                url=url,
                ok=False,
                codigo_http=respiente.status_code,
                bytes_html=len(html),
                error="HTML sin texto extraible",
                segundos=loop.time() - inicio,
            )

        return ResultadoDescarga(
            url=url,
            ok=True,
            titulo=titulo,
            texto=texto,
            codigo_http=respuesta.status_code,
            bytes_html=len(html),
            segundos=loop.time() - inicio,
            enlaces=enlaces,
        )
    except Exception as exc:
        # Aislamiento de fallos: una URL rota, lenta o con TLS invalido
        # devuelve un ResultadoDescarga con error en vez de propagar la
        # excepcion y abortar las demas del lote.
        return ResultadoDescarga(
            url=url,
            ok=False,
            error=f"{type(exc).__name__}: {exc}"[:200],
            segundos=loop.time() - inicio,
        )


async def descargar_lote(
    urls: List[str],
    concurrencia: int = 4,
    timeout: float = TIMEOUT_SEGUNDOS,
) -> List[ResultadoDescarga]:
    """Descarga `urls` con como maximo `concurrencia` peticiones en vuelo.

    El Semaphore es el unico regulador de peticiones simultaneas: sin el,
    asyncio crearia una tarea por URL y abriria tantas conexiones como URLs
    haya, reproducirse la limitacion del servidor remoto y disparar 429.
    """
    limite = asyncio.Semaphore(max(1, concurrencia))
    limites = httpx.Limits(
        max_connections=max(1, concurrencia),
        max_keepalive_connections=max(1, concurrencia),
    )
    tiempos = httpx.Timeout(timeout, connect=min(timeout, 10.0))

    async with httpx.AsyncClient(
        follow_redirects=True,
        max_redirects=MAX_REDIRECCIONES,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        limits=limites,
        timeout=tiempos,
    ) as cliente:

        async def con_semaforo(url: str) -> ResultadoDescarga:
            async with limite:
                return await _descargar_una(cliente, url, asyncio.get_running_loop())

        tareas = [con_semaforo(url) for url in urls]
        # gather con return_exceptions=False porque _descargar_una ya
        # captura sus propias excepciones; el resultado se correlaciona con
        # su URL por indice, nunca por orden de finalizacion.
        return await asyncio.gather(*tareas)