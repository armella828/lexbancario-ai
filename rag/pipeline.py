"""Orquestacion del pipeline de ingesta concurrente.

Encadena las cuatro fases y cronometra cada una de forma independiente. La
separacion en fases es lo que permite luego atribuir el speedup: sin medir
por separado descarga, chunking, embeddings e insercion no se puede saber
que parte del trabajo se paraleliza y cual sigue siendo serial.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .chunking import Trozo, trocear_lote
from .clients import ENTORNO
from .embeddings import LimitadorCadencia, generar_embeddings as _generar_embeddings
from .scraper import descargar_lote

# Tamano de lote de embeddings: lo resuelve `rag.embeddings` desde el
# entorno, asi que no se duplica la constante aqui.
# Filas por llamada de insercion. Supabase acepta lotes grandes, pero un
# INSERT unico con cientos de filas y vectores de 768 dimensiones excede
# el limite util de la peticion y complica reintentar un fallo parcial.
INSERCION_LOTE = 200


@dataclass
class MetricasFase:
    segundos: float = 0.0

    def __float__(self) -> float:
        return self.segundos


@dataclass
class ResultadoIngesta:
    coleccion_id: str
    descargadas_ok: int = 0
    descargadas_fallidas: int = 0
    fragmentos: int = 0
    insertados: int = 0
    errores: List[str] = field(default_factory=list)
    tiempos: dict = field(default_factory=dict)
    por_url: List[dict] = field(default_factory=list)

    @property
    def segundos_totales(self) -> float:
        return sum(self.tiempos.values())

    def resumen(self) -> dict:
        total = self.segundos_totales
        return {
            "coleccion_id": self.coleccion_id,
            "urls_ok": self.descargadas_ok,
            "urls_fallidas": self.descargadas_fallidas,
            "fragmentos": self.fragmentos,
            "insertados": self.insertados,
            "segundos_totales": round(total, 3),
            "segundos_por_fase": {k: round(v, 3) for k, v in self.tiempos.items()},
            "porcentaje_por_fase": {
                k: (round(100.0 * v / total, 1) if total > 0 else 0.0)
                for k, v in self.tiempos.items()
            },
            "errores": self.errores,
        }


def _embedir_lote(textos: List[str]) -> List[list]:
    """Compatibilidad: un unico lote sin control de cuota."""
    return _generar_embeddings(textos, tamano_lote=len(textos) or 1)


def generar_embeddings(
    trozos: List[Trozo],
    coleccion_id: str,
    documento_por_url: dict,
    tamano_lote: Optional[int] = None,
    limitador: Optional[LimitadorCadencia] = None,
) -> List[dict]:
    """Genera embeddings en lotes y devuelve filas listas para insertar."""
    filas = []
    # Un unico limitador para toda la ingesta: la cadencia se aplica sobre
    # el total de requests emitidos, no de forma independiente por lote.
    vectores = _generar_embeddings(
        [t.contenido for t in trozos],
        tamano_lote=tamano_lote,
        limitador=limitador or LimitadorCadencia(),
    )
    for trozo, vector in zip(trozos, vectores):
        filas.append({
            "documento_origen": documento_por_url.get(trozo.url, trozo.url),
            "organismo": documento_por_url.get(f"_org::{trozo.url}", "Web"),
            "tipo_norma": documento_por_url.get(f"_tipo::{trozo.url}", "Documento web"),
            "jerarquia": "Web",
            "articulo_ref": f"Fragmento {trozo.indice + 1}",
            "contenido": trozo.contenido,
            "coleccion_id": coleccion_id,
            "embedding": vector,
        })
    return filas


def _organizar_por_url(resultados) -> dict:
    """Mapea url -> metadatos, mas las filas de organisation."""
    mapa = {}
    for r in resultados:
        if not r.ok:
            continue
        mapa[r.url] = r.titulo or r.url
        mapa[f"_org::{r.url}"] = r.titulo or r.url
        mapa[f"_tipo::{r.url}"] = "Documento web"
    return mapa


def insertar_filas(filas: List[dict]) -> int:
    """Inserta las filas por lotes y devuelve cuantas se enviaron.

    Se trocea en lugar de mandar un unico INSERT porque el payload crece
    con el numero de filas y cada vector aporta 768 flotantes: un solo
    INSERT de cientos de filas se acerca al limite de la peticion y, si
    falla a medias, obliga a reintentar el lote entero. Con lotes de 200
    un fallo descarta mucho menos trabajo.
    """
    total = 0
    for inicio in range(0, len(filas), INSERCION_LOTE):
        lote = filas[inicio:inicio + INSERCION_LOTE]
        ENTORNO.supabase.table("normativa_bancaria").insert(lote).execute()
        total += len(lote)
    return total


def ingestar(
    urls: List[str],
    coleccion_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    concurrencia: int = 4,
    procesos_chunking: Optional[int] = None,
    tamano_lote: Optional[int] = None,
    inserting: bool = True,
) -> ResultadoIngesta:
    """Ejecuta el pipeline completo y devuelve metricas por fase.

    `tamano_lote=None` delega en la configuracion del entorno
    (`EMBEDDINGS_TAMANO_LOTE`), de modo que el ajuste se hace en un solo
    sitio y no queda duplicado entre el pipeline y el modulo de embeddings.
    """
    resultado = ResultadoIngesta(coleccion_id=coleccion_id)

    # --- Fase 1: descarga y limpieza (I/O) -----------------------------
    t0 = time.perf_counter()
    descargas = asyncio.run(
        descargar_lote(urls, concurrencia=concurrencia)
    )
    resultado.tiempos["descarga"] = time.perf_counter() - t0

    ok = [d for d in descargas if d.ok]
    resultado.descargadas_ok = len(ok)
    resultado.descargadas_fallidas = len(descargas) - len(ok)
    resultado.errores.extend(
        f"{d.url}: {d.error}" for d in descargas if not d.ok
    )
    resultado.por_url = [
        {
            "url": d.url,
            "ok": d.ok,
            "codigo_http": d.codigo_http,
            "bytes_html": d.bytes_html,
            "caracteres": len(d.texto),
            "error": d.error,
            "segundos": round(d.segundos, 3),
        }
        for d in descargas
    ]

    if not ok:
        resultado.tiempos["chunking"] = 0.0
        resultado.tiempos["embeddings"] = 0.0
        resultado.tiempos["insercion"] = 0.0
        return resultado

    # --- Fase 2: chunking (CPU) ----------------------------------------
    t0 = time.perf_counter()
    paquetes = [
        {
            "texto": d.texto,
            "url": d.url,
            "documento": d.titulo or d.url,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        for d in ok
    ]
    trozos = trocear_lote(paquetes, procesos=procesos_chunking)
    resultado.tiempos["chunking"] = time.perf_counter() - t0
    resultado.fragmentos = len(trozos)

    # --- Fase 3: embeddings (I/O contra API de terceros) ---------------
    t0 = time.perf_counter()
    filas = generar_embeddings(
        trozos, coleccion_id, _organizar_por_url(descargas), tamano_lote
    )
    resultado.tiempos["embeddings"] = time.perf_counter() - t0

    # --- Fase 4: insercion en Supabase ---------------------------------
    t0 = time.perf_counter()
    if inserting and filas:
        resultado.insertados = insertar_filas(filas)
    resultado.tiempos["insercion"] = time.perf_counter() - t0

    return resultado