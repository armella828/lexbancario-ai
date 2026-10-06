"""Benchmark del pipeline: Serie A (sin embeddings) y Serie B (con cache).

Division del trabajo, y por que es esta:

  Serie A  Descarga + troceado + insercion, SIN llamar al proveedor de
           embeddings. Es la parte del trabajo que este repositorio puede
           paralelizar por su cuenta. No depende de la cuota externa, asi
           que se puede medir en cualquier momento.

  Serie B  Lo mismo mas la generacion de embeddings, servidos desde la
           cache local. Es la parte que la cuota externa hace lenta y no
           paralelizable por nosotros: por eso va aparte.

Que S_p y E_p signifiquen exactamente lo que dicen:

  S_p  Speedup del pipeline web (descarga + chunking + insercion) al
       comparar concurrencia alta contra concurrencia 1. Mide el
       paralelismo propio del codigo.

  E_p  Speedup del embedding al comparar lotes de 10 contra una peticion
       por texto. Mide el ahorro de peticiones, no el paralelismo.

Se reportan por separado y no se multiplican entre si: mezclarlas
produciria un numero que no corresponde a ninguna ejecucion real.

Uso:
    python scripts/benchmark.py --serie a
    python scripts/benchmark.py --serie b --replicas 5
    python scripts/benchmark.py --serie todas
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from rag.chunking import trocear_lote
from rag.clients import ENTORNO
from rag.embeddings import CacheEmbeddings, generar_embeddings
from rag.pipeline import INSERCION_LOTE
from rag.scraper import descargar_lote

load_dotenv()

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.environ.get("BENCHMARK_BASE_URL", "http://127.0.0.1:8099")
COLECCION = os.environ.get("BENCHMARK_COLECCION", "bench_serie_a")

# Fases del pipeline que Serie A mide. La de embeddings se excluye a
# proposito: es la unica que depende de un servicio externo con cuota.
FASES_SERIE_A = ("descarga", "chunking", "insercion")

# Fases que el pipeline ejecuta en secuencia y que por tanto fijan el
# techo de Amdahl. Se declaran en vez de calcularse para que el motivo
# quede en el codigo: la insercion manda los lotes de 200 uno tras otro,
# mientras descarga (concurrencia asyncio) y chunking (ProcessPoolExecutor)
# si se paralelizan.
FASES_SERIALES = ("insercion",)


def urls_fixture(cantidad):
    """URLs de documentos validos.

    El servidor expone /doc/1..N; /doc/0 responde 404. Empezar en 0 daria
    URLs rotas que la descarga descartaria, y el benchmark mediria menos
    trabajo del que cree, en vez de detectar el problema.
    """
    base = FIXTURES.rstrip("/")
    return [f"{base}/doc/{i}" for i in range(1, cantidad + 1)]


def esperar_fixture(intentos=25, espera=0.4):
    """Comprueba que el servidor local responde antes de medir nada.

    Medir contra un servidor caido produce tiempos rapidos y falsos, que es
    la forma mas facil de publicar un speedup que no existe.
    """
    import httpx

    for _ in range(intentos):
        try:
            r = httpx.get(f"{FIXTURES.rstrip('/')}/doc/1", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(espera)
    return False


def _con_db(fn, intentos=4, espera_base=1.0):
    """Reintenta una operacion contra Supabase ante un error transitorio.

    Un `Connection reset by peer` puntual no invalida la medicion: abortar
    la corrida entera obligaria a repetir cinco minutos de trabajo por un
    tropiezo de red que no tiene nada que ver con el paralelismo que se
    mide. Se reintenta con espera creciente, y si persiste se propaga el
    error original para no enmascarar un fallo real de la base.
    """
    ultimo = None
    for i in range(intentos):
        try:
            return fn()
        except Exception as e:
            ultimo = e
            if i == intentos - 1:
                break
            time.sleep(espera_base * (i + 1))
    raise ultimo


def _insertar(filas, coleccion):
    total = 0
    for inicio in range(0, len(filas), INSERCION_LOTE):
        lote = filas[inicio:inicio + INSERCION_LOTE]
        _con_db(lambda lote=lote: ENTORNO.supabase.table("normativa_bancaria")
                .insert(lote).execute())
        total += len(lote)
    return total


def _borrar(coleccion):
    _con_db(lambda: ENTORNO.supabase.table("normativa_bancaria").delete()
            .eq("coleccion_id", coleccion).execute())


def serie_a_una_pasada(urls, concurrencia, procesos, coleccion, insertar=True):
    """Una ejecucion del pipeline sin embeddings. Devuelve tiempos por fase."""
    t0 = time.perf_counter()
    descargas = asyncio.run(descargar_lote(urls, concurrencia=concurrencia))
    t_descarga = time.perf_counter() - t0

    ok = [d for d in descargas if d.ok]
    if not ok:
        return {"descarga": t_descarga, "chunking": 0.0, "insercion": 0.0,
                "ok": 0, "fragmentos": 0}

    paquetes = [{"texto": d.texto, "url": d.url, "documento": d.url,
                 "chunk_size": 1000, "chunk_overlap": 200} for d in ok]

    t0 = time.perf_counter()
    trozos = trocear_lote(paquetes, procesos=procesos)
    t_chunk = time.perf_counter() - t0

    t_ins = 0.0
    if insertar and trozos:
        filas = [{
            "documento_origen": t.url, "organismo": "Fixture",
            "tipo_norma": "Documento web", "jerarquia": "Web",
            "articulo_ref": f"Fragmento {t.indice + 1}",
            "contenido": t.contenido, "coleccion_id": coleccion,
        } for t in trozos]
        # El borrado va en `finally`: si la insercion falla a mitad, las
        # filas que entraron quedarian en la tabla y la siguiente replica
        # las mediria como suyas.
        try:
            t0 = time.perf_counter()
            _insertar(filas, coleccion)
            t_ins = time.perf_counter() - t0
        finally:
            _borrar(coleccion)

    return {"descarga": t_descarga, "chunking": t_chunk,
            "insercion": t_ins, "ok": len(ok), "fragmentos": len(trozos)}


def serie_b_una_pasada(urls, concurrencia, procesos, coleccion, tamano_lote,
                       con_cache=True):
    """Pipeline completo: descarga, chunking, embeddings e insercion.

    Se miden las cuatro fases por separado porque la guia pide el reparto
    de tiempo con la fase de API incluida. Antes la insercion se saltaba
    (`insertar=False`, de ahi `insercion: 0.0`) y la descarga se hacia
    dos veces: el total no correspondia a ningun pipeline real.

    `con_cache=False` manda los textos contra la API real. Esa variacion
    no es un error de medicion: es la que muestra cuanto pesa la cuota de
    un tercero cuando no se ha construido ninguna mitigacion.
    """
    t0 = time.perf_counter()
    descargas = asyncio.run(descargar_lote(urls, concurrencia=concurrencia))
    t_descarga = time.perf_counter() - t0

    ok = [d for d in descargas if d.ok]
    if not ok:
        return {"descarga": t_descarga, "chunking": 0.0, "embeddings": 0.0,
                "insercion": 0.0, "ok": 0, "fragmentos": 0}

    paquetes = [{"texto": d.texto, "url": d.url, "documento": d.url,
                 "chunk_size": 1000, "chunk_overlap": 200} for d in ok]

    t0 = time.perf_counter()
    trozos = trocear_lote(paquetes, procesos=procesos)
    t_chunk = time.perf_counter() - t0

    cache = CacheEmbeddings().cargar() if con_cache else None
    t0 = time.perf_counter()
    vectores = generar_embeddings(
        [t.contenido for t in trozos], tamano_lote=tamano_lote, cache=cache
    )
    t_emb = time.perf_counter() - t0

    # Filas con embedding, como las que escribe el pipeline real: sin el
    # vector la insercion seria mas barata que en produccion.
    filas = [{
        "documento_origen": t.url, "organismo": "Fixture",
        "tipo_norma": "Documento web", "jerarquia": "Web",
        "articulo_ref": f"Fragmento {t.indice + 1}",
        "contenido": t.contenido, "coleccion_id": coleccion,
        "embedding": vector,
    } for t, vector in zip(trozos, vectores)]

    t_ins = 0.0
    if filas:
        try:
            t0 = time.perf_counter()
            _insertar(filas, coleccion)
            t_ins = time.perf_counter() - t0
        finally:
            _borrar(coleccion)

    return {"descarga": t_descarga, "chunking": t_chunk,
            "embeddings": t_emb, "insercion": t_ins,
            "ok": len(ok), "fragmentos": len(trozos),
            "cache_aciertos": cache.aciertos if cache else 0,
            "cache_fallidos": cache.fallos if cache else 0,
            "peticiones_ahorradas": cache.peticiones_evitadas if cache else 0}


def estadisticas(valores):
    if not valores:
        return None
    orden = sorted(valores)
    return {
        "media": round(statistics.mean(valores), 4),
        "mediana": round(statistics.median(valores), 4),
        "min": round(min(valores), 4),
        "max": round(max(valores), 4),
        "desviacion": round(statistics.pstdev(valores), 4),
        "n": len(valores),
    }


def amdahl(fraccion_serial, p):
    """Techo de speedup de Amdahl con p procesadores.

    Es un techo, no una prediccion: asume que la parte serial escala
   perfectamente y que la paralela se divide entre p sin coste alguno. El
    rendimiento real siempre queda por debajo, y la razon suele ser
    justamente el costo de coordinacion, que es lo que mas crece al
    pasar de 2 a 24 nucleos.
    """
    return 1.0 / (fraccion_serial + (1.0 - fraccion_serial) / p)


def ejecutar_serie_a(args):
    if not esperar_fixture():
        print("[X] El servidor de fixtures no responde en "
              f"{FIXTURES}. Arrancalo con:  python scripts/servidor_fixtures.py")
        return 1

    urls = urls_fixture(args.urls)
    print("=" * 74)
    print("SERIE A: pipeline web sin embeddings (no depende de cuota)")
    print("=" * 74)
    print(f"[*] {len(urls)} URLs, {args.replicas} replicas por configuracion")

    # Limpieza previa: una corrida abortada (SIGKILL, OOM, corte de red)
    # puede dejar filas sueltas, y esas filas inflarian el tiempo de
    # insercion de la primera replica sin que nadie lo note.
    _borrar(COLECCION)

    configuraciones = [(1, 1), (2, 2), (4, 4), (args.concurrencia, args.procesos)]
    configuraciones = sorted(set(configuraciones))

    resultados = []
    for concurrencia, procesos in configuraciones:
        muestras = []
        fases = {f: [] for f in FASES_SERIE_A}
        for _ in range(args.replicas):
            r = serie_a_una_pasada(urls, concurrencia, procesos, COLECCION)
            muestras.append(r["descarga"] + r["chunking"] + r["insercion"])
            for f in FASES_SERIE_A:
                fases[f].append(r[f])
        fila = {
            "concurrencia": concurrencia,
            "procesos": procesos,
            "total": estadisticas(muestras),
            "fases": {f: estadisticas(v) for f, v in fases.items()},
        }
        resultados.append(fila)
        t = fila["total"]
        print(f"  c={concurrencia:<3} p={procesos:<3} "
              f"media={t['media']:>7.2f}s  min={t['min']:>7.2f}s  "
              f"max={t['max']:>7.2f}s  desv={t['desviacion']:>6.2f}s")

    base = resultados[0]["total"]["media"]
    print()
    print("Speedup respecto a concurrencia 1:")
    for fila in resultados:
        sp = base / fila["total"]["media"]
        fila["speedup"] = round(sp, 3)
        # Eficiencia = S_p / p, la formula que pide la guia. NO es speedup
        # sobre el techo de Amdahl: eso daria otra magnitud, que ademas
        # hace quedar al paralelismo mejor de lo que es.
        fila["eficiencia"] = round(sp / fila["procesos"], 3)
        print(f"  c={fila['concurrencia']:<3} {sp:>6.2f}x  "
              f"eficiencia {fila['eficiencia']:.2f}")

    # Amdahl: la fraccion serial se mide del reparto de fases observado,
    # no se supone. Es lo que separa "no paralelizable" de "no medido".
    #
    # Dos precisiones que cambian el resultado:
    #  - Las fases seriales son las que el pipeline ejecuta en secuencia.
    #    La insercion manda lotes uno detras de otro y se mantiene plana
    #    al subir p; el chunking si usa ProcessPoolExecutor. Medir la
    #    fraccion con el chunking (0.5%) daria un techo de 7.7x y haria
    #    parecer mal al paralelismo cuando el limite real es la insercion.
    #  - La linea base de Amdahl es p=1. Con p=8 las fases seriales
    #    conservan su duracion pero las paralelas se encogen, y el cociente
    #    deja de ser la fraccion que busca la formula.
    ref = resultados[0]
    fases_tot = sum(f["media"] for f in ref["fases"].values())
    serial = sum(ref["fases"][f]["media"] for f in FASES_SERIALES)
    frac = serial / fases_tot if fases_tot else 0.0
    print()
    print(f"Amdahl sobre Serie A (fraccion serial medida = "
          f"{'+'.join(FASES_SERIALES)} {frac:.1%}):")
    for p in sorted({2, 4, 8, args.procesos}):
        techo = amdahl(frac, p)
        medido = next((f["speedup"] for f in resultados
                       if f["procesos"] == p), None)
        if medido is not None:
            print(f"  p={p:<3} techo {techo:>6.2f}x  medido {medido:>6.2f}x  "
                  f"eficiencia {medido / p:.2f}  "
                  f"del techo {medido / techo:.0%}")
        else:
            print(f"  p={p:<3} techo {techo:>6.2f}x  (sin medicion en este equipo)")

    return {
        "serie": "A",
        "descripcion": "descarga + chunking + insercion, sin embeddings",
        "urls": len(urls),
        "configuraciones": resultados,
        "fraccion_serial_medida": round(frac, 4),
    }


def ejecutar_serie_b(args):
    if not esperar_fixture():
        print("[X] El servidor de fixtures no responde en " + FIXTURES)
        return 1

    fria = getattr(args, "sin_cache", False)
    urls = urls_fixture(args.urls)
    print("=" * 74)
    if fria:
        print("SERIE B FRIA: pipeline completo contra la API de embeddings")
    else:
        print("SERIE B: Serie A + embeddings desde cache local")
    print("=" * 74)

    if not fria:
        cache = CacheEmbeddings().cargar()
        if cache.resumen()["vectores"] == 0:
            print("[X] La cache esta vacia. Calentala con:")
            print("      python scripts/cache_embeddings.py --desde-db")
            print("    Sin cache, esta serie mediria la cuota de la API, que es")
            print("    exactamente lo que no se quiere medir.")
            return 1
        print(f"[*] {cache.resumen()['vectores']} vectores en cache; "
              f"{len(urls)} URLs, {args.replicas} replicas")
    else:
        print(f"[*] sin cache: {len(urls)} URLs, {args.replicas} replicas")
        print("[i] Los embeddings salen contra la API real; paga cuota.")

    r = serie_b_una_pasada(urls, args.concurrencia, args.procesos,
                           COLECCION, args.tamano_lote, con_cache=not fria)
    total = r["descarga"] + r["chunking"] + r["embeddings"] + r["insercion"]
    print(f"  descarga    {r['descarga']:>7.2f}s")
    print(f"  chunking    {r['chunking']:>7.2f}s")
    print(f"  embeddings  {r['embeddings']:>7.2f}s  "
          f"(cache: {r.get('cache_aciertos', 0)} aciertos, "
          f"{r.get('peticiones_ahorradas', 0)} peticiones evitadas)")
    print(f"  insercion   {r['insercion']:>7.2f}s")
    print(f"  TOTAL       {total:>7.2f}s")

    # Si la mayoria de los textos no estaban en cache, esta corrida mide
    # la API real, no la cache: es una pasada en frio y se guarda con esa
    # etiqueta. Detectarlo por el resultado en vez de por una bandera evita
    # publicar como "pipeline rapido" una medicion que pago 920 requests.
    fallos = r.get("cache_fallidos", 0)
    fria = fria or (fallos > max(1, r.get("fragmentos", 0) * 0.5))

    clave = "serie_b_fria" if fria else "serie_b"
    descripcion = ("pipeline completo en cache fria: la fase de embeddings "
                   "llama a la API de terceros" if fria else
                   "Serie A + embeddings servidos desde cache local")
    return clave, {
        "serie": "B" + (" (fria)" if fria else ""),
        "descripcion": descripcion,
        "sin_cache": fria,
        "una_pasada": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items()},
        "total_segundos": round(total, 4),
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark Serie A y B")
    p.add_argument("--serie", choices=["a", "b", "todas"], default="a")
    p.add_argument("--urls", type=int, default=12)
    p.add_argument("--replicas", type=int, default=5)
    p.add_argument("--concurrencia", type=int, default=8)
    p.add_argument("--procesos", type=int, default=8)
    p.add_argument("--tamano-lote", type=int, default=10)
    p.add_argument("--sin-cache", action="store_true",
                   help="Serie B contra la API real (paga cuota). Sirve para "
                        "medir cuanto pesa la API de terceros sin mitigacion.")
    args = p.parse_args()

    salida = {}
    if args.serie in ("a", "todas"):
        salida["serie_a"] = ejecutar_serie_a(args)
    if args.serie in ("b", "todas"):
        # Cualquier excepcion aqui (cuota agotada, red caida) se convierte
        # en un fallo registrado. Sin esto, la Serie B aborta el proceso
        # y se tira a la basura la Serie A, que ya se habia medido y que
        # no depende de la API: son el descuento mas caro que puede haber.
        try:
            rta = ejecutar_serie_b(args)
        except Exception as e:
            print(f"[X] Serie B no pudo completarse: {type(e).__name__}: {e}")
            rta = 1
        if isinstance(rta, tuple):
            salida[rta[0]] = rta[1]
        else:
            salida["serie_b"] = rta

    # Se fusiona con lo anterior en lugar de sobrescribir: las dos series
    # no dependen la una de la otra, y correr solo una no debe borrar la
    # evidencia de la otra. Ademas, un codigo de retorno no es un
    # resultado: si una serie fallo, se guarda el error con su mensaje, no
    # un entero que un grafico interpretaria como dato.
    limpia = {k: v for k, v in salida.items() if not isinstance(v, int)}
    if not limpia:
        print("[X] Ninguna serie produjo resultados.")
        return 1

    ruta = os.path.join(RAIZ, "resultados", "benchmark.json")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    previo = {}
    if os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as fh:
                previo = json.load(fh)
        except (ValueError, OSError):
            previo = {}

    acumulado = dict(previo.get("resultados", {}))
    acumulado.update(limpia)
    fallidas = [k for k, v in salida.items() if isinstance(v, int)]

    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "resultados": acumulado},
                  fh, ensure_ascii=False, indent=2)

    print()
    print(f"[*] Series guardadas: {', '.join(sorted(limpia))}")
    if fallidas:
        print(f"[!] Sin datos: {', '.join(fallidas)} "
              "(se conserva lo anterior)")
    print(f"[*] Resultados en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())