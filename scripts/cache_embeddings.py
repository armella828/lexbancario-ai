"""Calienta la cache de embeddings de un corpus, sin repetir trabajo ya hecho.

Motivo: el nivel gratuito de Google AI Studio da 1000 requests de
embeddings al dia. Repetir un mismo corpus en varias replicas de benchmark
agotaria la cuota en la primera y las siguientes medirian esperas de backoff
en lugar de concurrencia. Calentar la cache una vez deja el resto de
ejecuciones sin coste de API.

Ejemplos:

    # Calentar con los fragmentos que ya hay en la base
    python scripts/cache_embeddings.py --desde-db

    # Calentar con un corpus de texto local
    python scripts/cache_embeddings.py --texto "capital minimo" --texto "reserva legal"

    # Ver estado sin gastar cuota
    python scripts/cache_embeddings.py --estado
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from rag.clients import DIMENSIONES_EMBEDDING
from rag.embeddings import CacheEmbeddings, CuotaAgotada, generar_embeddings

load_dotenv()


def desde_db(limite=0, cliente=None):
    """Fragmentos ya insertados, para no pagar de nuevo lo que esta guardado."""
    from rag.clients import ENTORNO

    supabase = cliente or ENTORNO.supabase
    consulta = supabase.table("normativa_bancaria").select("contenido")
    if limite:
        consulta = consulta.limit(limite)
    filas = consulta.execute().data or []
    textos, vistos = [], set()
    for fila in filas:
        contenido = (fila.get("contenido") or "").strip()
        # La clave es el contenido: si dos fragmentos coinciden, la cache
        # tambien guarda solo uno.
        if contenido and contenido not in vistos:
            vistos.add(contenido)
            textos.append(contenido)
    return textos


def desde_archivos(rutas):
    """Texto de archivos locales, para corpus de benchmark."""
    import re as _re

    textos = []
    for ruta in rutas:
        with open(ruta, encoding="utf-8", errors="replace") as f:
            crudo = f.read()
        for trozo in _re.split(r"\n\s*\n", crudo):
            limpio = trozo.strip()
            if len(limpio) > 40:
                textos.append(limpio)
    return textos


def mostrar_estado(cache):
    info = cache.resumen()
    print(f"[i] Ruta de cache : {cache.ruta}")
    print(f"[i] Vectores      : {info['vectores']}")
    print(f"[i] Modelo        : {cache.modelo}")
    print(f"[i] Dimension     : {cache.dimension}")
    print(f"[i] Bytes en disco: {os.path.getsize(cache.ruta) if os.path.exists(cache.ruta) else 0}")
    return info


def main():
    p = argparse.ArgumentParser(description="Cache de embeddings")
    p.add_argument("--ruta", default=None, help="Ruta del archivo de cache")
    p.add_argument("--estado", action="store_true", help="Solo mostrar estado")
    p.add_argument("--desde-db", action="store_true", help="Tomar fragmentos de la base")
    p.add_argument("--limite-db", type=int, default=0)
    p.add_argument("--texto", action="append", default=[], help="Texto a vectorizar")
    p.add_argument("--archivo", action="append", default=[], help="Archivo de texto")
    args = p.parse_args()

    cache = CacheEmbeddings(ruta=args.ruta).cargar()

    if args.estado or not (args.desde_db or args.texto or args.archivo):
        mostrar_estado(cache)
        return 0

    textos = list(args.texto)
    if args.archivo:
        textos.extend(desde_archivos(args.archivo))
    if args.desde_db:
        textos.extend(desde_db(args.limite_db))

    if not textos:
        print("[X] No hay textos que vectorizar.")
        return 1

    pendientes = cache.faltan(textos)
    print(f"[*] {len(textos)} textos, {len(pendientes)} sin cachear")

    if not pendientes:
        print("[OK] Todo estaba cacheado. 0 requests consumidos.")
        mostrar_estado(cache)
        return 0

    try:
        t0 = time.perf_counter()
        vectores = generar_embeddings(textos, cache=cache)
        elapsed = time.perf_counter() - t0
    except CuotaAgotada as exc:
        # Calentar la cache depende de cuota. Si falta, se dice cuando
        # reintentar en lugar de dejar un volcado de excepcion.
        print(f"[--] No se pudo calentar la cache: {exc}")
        mostrar_estado(cache)
        return 2

    if len(vectores) != len(textos):
        print(f"[X] Se esperaban {len(textos)} vectores y salieron {len(vectores)}")
        return 1

    for v in vectores:
        if len(v) != DIMENSIONES_EMBEDDING:
            print(f"[X] Vector con {len(v)} dimensiones, se esperaban {DIMENSIONES_EMBEDDING}")
            return 1

    print(f"[OK] {len(textos)} vectores en {elapsed:.1f}s "
          f"({len(pendientes)} textos nuevos)")
    mostrar_estado(cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())