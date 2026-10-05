import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import ingest_normativa as ing


def medir(fn, *args):
    t0 = time.time()
    resultado = fn(*args)
    return resultado, time.time() - t0


def bench_ocr(pdfs, workers):
    print("\n[i] Extraccion de texto/OCR (CPU-bound) - NO usa API")

    tabla = []
    t0 = time.time()
    for ruta in pdfs:
        t1 = time.time()
        ing.extraer_texto_pdf(ruta)
        tabla.append((os.path.basename(ruta), t1, time.time() - t1))
    t_seq = time.time() - t0

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(ing.extraer_texto_pdf, pdfs))
    t_par = time.time() - t0

    return tabla, t_seq, t_par


def bench_embed(chunks, workers, tam_lote):
    print("\n[i] Embeddings de fragmentos (I/O-bound contra Gemini API)")

    textos = [c["contenido"] for c in chunks[:60]]
    lotes = [textos[i:i + tam_lote] for i in range(0, len(textos), tam_lote)]

    total = 0
    t0 = time.time()
    for l in lotes:
        ing.generar_embeddings_con_reintento(l)
    t_seq = time.time() - t0
    total += len(textos)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(ing.generar_embeddings_con_reintento, lotes))
    t_par = time.time() - t0

    return len(textos), t_seq, t_par


def main():
    ap = argparse.ArgumentParser(description="Benchmark OCR y embeddings (sin escribir en Supabase)")
    ap.add_argument("--pdfs", type=int, default=4, help="Cantidad de PDFs a usar (default 4)")
    ap.add_argument("--ocr-workers", type=int, default=4)
    ap.add_argument("--embed-workers", type=int, default=4)
    ap.add_argument("--lote", type=int, default=10)
    ap.add_argument("--modo", choices=["todo", "ocr", "embed"], default="todo")
    args = ap.parse_args()

    directorio = "documentos_pdf"
    archivos = sorted(
        f for f in os.listdir(directorio)
        if f.lower().endswith(".pdf") and f.lower() != "circulares.pdf"
    )[:args.pdfs]
    if not archivos:
        print("[!] No hay PDFs.")
        sys.exit(1)

    rutas = [os.path.join(directorio, a) for a in archivos]
    print(f"[*] PDFs seleccionados ({len(rutas)}): {', '.join(archivos)}")

    chunks_totales = []
    if args.modo in ("todo", "ocr"):
        tabla_ocr, tot_seq, tot_par = bench_ocr(rutas, args.ocr_workers)
        print(f"\n{'Documento':<28}{'t_seq_pdf(s)':>14}")
        for nombre, _t1, t_sec in tabla_ocr:
            print(f"{nombre:<28}{t_sec:>14.2f}")
        print(f"\n{'Indicador':<28}{'valor':>14}")
        print(f"{'Total secuencial (todos)':<28}{tot_seq:>14.2f}")
        print(f"{'Total paralelo (workers=%d)' % args.ocr_workers:<28}{tot_par:>14.2f}")
        print(f"{'Speedup':<28}{tot_seq/tot_par:>14.2f}")

        if args.modo == "ocr":
            return

        for ruta in rutas:
            paginas = ing.extraer_texto_pdf(ruta)
            chunks_totales += ing.parsear_articulos(paginas, os.path.basename(ruta))

    if args.modo in ("todo", "embed"):
        if not chunks_totales:
            for ruta in rutas:
                paginas = ing.extraer_texto_pdf(ruta)
                chunks_totales += ing.parsear_articulos(paginas, os.path.basename(ruta))
        n, t_seq, t_par = bench_embed(chunks_totales, args.embed_workers, args.lote)
        print(f"\n{'Fragmentos':>10}{'t_seq(s)':>12}{'t_par(s)':>12}{'Speedup':>9}")
        print(f"{n:>10}{t_seq:>12.2f}{t_par:>12.2f}{t_seq / t_par:>9.2f}")


if __name__ == "__main__":
    main()