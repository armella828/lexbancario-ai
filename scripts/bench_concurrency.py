import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL_DEFAULT = "http://localhost:8000/api/consultar"
PAYLOAD_DEFAULT = {"pregunta": "¿Qué es la ASFI y cuáles son sus facultades?", "top_k": 2}


def medir(guardar, url, payload, n, nivel):
    """Ejecuta n consultas concurrentes y guarda resultados [(t_seg, http_code)]."""
    timeout = 130

    def una_consulta(_):
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            return (time.time() - t0, r.status_code)
        except requests.exceptions.RequestException as e:
            return (time.time() - t0, 0)

    with ThreadPoolExecutor(max_workers=nivel) as ex:
        for res in ex.map(una_consulta, range(n)):
            guardar.append(res)


def imprimir_tabla(nivel, latencias, codes):
    if not latencias:
        print(f"{nivel:>6}{'0':>10}{'0.0':>10}{'0.0':>10}{'0.0':>10}{'0.00':>8}{str(codes):>14}")
        return
    avg = statistics.mean(latencias)
    orde = sorted(latencias)
    p50 = orde[len(orde) // 2]
    p95 = orde[int(len(orde) * 0.95) - 1]
    pared = time.monotonic()  # no usado; rps desde latencias paralelas
    ok = sum(1 for c in codes if c == 200)
    # rps estimado: solicitudes exitosas / ventana (aprox. max latencia de la tanda si es 1 tanda)
    rps = ok / max(orde) if orde else 0
    distint = {c: codes.count(c) for c in sorted(set(codes))}
    print(f"{nivel:>6}{len(latencias):>10}{avg:>10.2f}{p50:>10.2f}{p95:>10.2f}{rps:>8.2f}{str(distint):>16}")


def main():
    ap = argparse.ArgumentParser(description="Prueba de carga simple contra /api/consultar")
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--pregunta", default=PAYLOAD_DEFAULT["pregunta"])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--n", type=int, default=5, help="Solicitudes por nivel (default 5)")
    args = ap.parse_args()

    payload = {"pregunta": args.pregunta, "top_k": 2}
    print(f"[*] Endpoint: {args.url}")
    print(f"[*] Solicitudes por nivel: {args.n} | Niveles: {args.levels}\n")

    print(f"{'Concurrencia':>12}{'Reqs':>10}{'avg(s)':>10}{'p50(s)':>10}{'p95(s)':>10}{'rps':>8}{'códigos_http':>18}")
    for nivel in args.levels:
        resultado = []
        medir(resultado, args.url, payload, args.n, nivel)
        latencias = [r[0] for r in resultado]
        codes = [r[1] for r in resultado]
        imprimir_tabla(nivel, latencias, codes)

    print("\n[*] Notas: rps = solicitudes 200 / max(latencia); 0 en http_code = timeout/error de conexión.")


if __name__ == "__main__":
    main()