"""Prueba de aceptacion del pipeline de ingesta concurrente (Requisito 2).

Verifica, contra el servidor de fixtures, que:
  1. se ingieren 10 URLs de tamano similar;
  2. una URL inexistente (404) y una lenta (timeout) NO abortan el lote;
  3. los fragmentos se insertan en la coleccion solicitada y en ninguna otra;
  4. los embeddings son consultables a traves de la RPC aislada;
  5. las filas de prueba se eliminan al terminar.

Uso:
  python scripts/servidor_fixtures.py &          # terminal 1
  python scripts/probar_ingesta.py               # terminal 2
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# El script vive en scripts/; se anade la raiz del proyecto al path para
# poder importar el paquete `rag` igual que lo hace main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from supabase import create_client

from rag.embeddings import CuotaAgotada
from rag.pipeline import ingestar

load_dotenv()

TABLA = "normativa_bancaria"
RPC = "match_normativa_coleccion"
COLECCION = "demo_ingesta_web"
PUERTO = 8099


def arrancar_servidor(puerto: int) -> subprocess.Popen:
    proceso = subprocess.Popen(
        [sys.executable, "scripts/servidor_fixtures.py", "--port", str(puerto)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import httpx

    for _ in range(40):
        time.sleep(0.5)
        try:
            r = httpx.get(f"http://127.0.0.1:{puerto}/salud", timeout=2)
            if r.status_code == 200:
                print(f"[i] Servidor de fixtures listo en el puerto {puerto}")
                return proceso
        except Exception:
            continue
    proceso.terminate()
    raise RuntimeError("El servidor de fixtures no respondio a tiempo")


def limpiar(cliente, coleccion: str):
    cliente.table(TABLA).delete().eq("coleccion_id", coleccion).execute()
    total = (
        cliente.table(TABLA)
        .select("id", count="exact")
        .eq("coleccion_id", coleccion)
        .execute()
    )
    print(f"[i] Filas de '{coleccion}' tras limpiar: {getattr(total, 'count', 0)}")


def purgar(cliente, coleccion: str):
    """Deja la coleccion vacia ANTES de empezar.

    Necesario porque una ejecucion interrumpida puede dejar filas a medias:
    sin esta purga, el conteo final compararia contra residuos de corridas
    previas y la prueba fallaria sin que el pipeline tenga culpa.
    """
    previas = (
        cliente.table(TABLA)
        .select("id", count="exact")
        .eq("coleccion_id", coleccion)
        .execute()
    )
    n = getattr(previas, "count", 0)
    if n:
        limpiar(cliente, coleccion)
        print(f"[i] Purgadas {n} filas residuales de una corrida previa")
    else:
        print(f"[i] Coleccion '{coleccion}' vacia al inicio")


def main():
    ap = argparse.ArgumentParser(description="Prueba del pipeline de ingesta concurrente")
    ap.add_argument("--puerto", type=int, default=PUERTO)
    ap.add_argument("--concurrencia", type=int, default=4)
    ap.add_argument("--externo", action="store_true",
                    help="Asume que el servidor ya esta corriendo")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.puerto}"
    proceso = None
    if not args.externo:
        proceso = arrancar_servidor(args.puerto)

    cliente = create_client(
        __import__("os").getenv("SUPABASE_URL"),
        __import__("os").getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )

    # 10 documentos validos mas dos casos de fallo deliberados.
    purgar(cliente, COLECCION)
    urls_ok = [f"{base}/doc/{n}" for n in range(1, 11)]
    urls = urls_ok + [f"{base}/roto", f"{base}/lento?segundos=40"]
    print(f"[*] Ingestando {len(urls)} URLs en la coleccion '{COLECCION}' "
          f"(concurrencia={args.concurrencia})")

    try:
        try:
            resultado = ingestar(
                urls=urls,
                coleccion_id=COLECCION,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                concurrencia=args.concurrencia,
                procesos_chunking=args.concurrencia,
            )
        except CuotaAgotada as exc:
            # Sin cuota de embeddings no se puede completar la ingesta. Es
            # una condicion del entorno, no un fallo del pipeline.
            print("\n" + "=" * 72)
            print("[--] PRUEBA OMITIDA: cuota de embeddings agotada.")
            print("=" * 72)
            print(f"     {exc}")
            print("\n     Reejecutar cuando la cuota se renueve.")
            return 0

        resumen = resultado.resumen()
        print("\n" + "=" * 72)
        print("PRUEBA DE INGESTA CONCURRENTE")
        print("=" * 72)
        print(f"URLs recibidas        : {len(urls)}")
        print(f"URLs correctas        : {resultado.descargadas_ok}")
        print(f"URLs con error        : {resultado.descargadas_fallidas}")
        print(f"Fragmentos generados  : {resultado.fragmentos}")
        print(f"Fragmentos insertados : {resultado.insertados}")
        print(f"Tiempo total          : {resumen['segundos_totales']} s")
        print("\nDesglose por fase:")
        for fase, pct in resumen["porcentaje_por_fase"].items():
            print(f"  {fase:<14} {resumen['segundos_por_fase'][fase]:>8.3f} s  {pct:>6.1f}%")
        print("\nErrores aislados por URL:")
        for err in resultado.errores:
            print(f"  {err[:110]}")
        print("=" * 72 + "\n")

        # --- Aserciones ---------------------------------------------
        fallos = []

        if resultado.descargadas_ok != 10:
            fallos.append(
                f"Se esperaban 10 URLs correctas, llegaron {resultado.descargadas_ok}."
            )
        if resultado.descargadas_fallidas != 2:
            fallos.append(
                f"Se esperaban 2 URLs en fallo, llegaron {resultado.descargadas_fallidas}."
            )
        if resultado.fragmentos <= 0:
            fallos.append("No se genero ningun fragmento.")
        if resultado.insertados != resultado.fragmentos:
            fallos.append(
                f"Se insertaron {resultado.insertados} de {resultado.fragmentos} fragmentos."
            )

        # Aislamiento: nada debe haber aterrizado fuera de la coleccion.
        propias = (
            cliente.table(TABLA)
            .select("id", count="exact")
            .eq("coleccion_id", COLECCION)
            .execute()
        )
        n_propias = getattr(propias, "count", 0)
        if n_propias != resultado.insertados:
            fallos.append(
                f"La coleccion '{COLECCION}' tiene {n_propias} filas pero se "
                f"insertaron {resultado.insertados}."
            )

        # Consultabilidad: la RPC aislada debe devolver algo.
        vector = _embedir_pregunta()
        consulta = cliente.rpc(RPC, {
            "query_embedding": vector,
            "match_threshold": 0.2,
            "match_count": 3,
            "p_coleccion_id": COLECCION,
        }).execute().data or []
        if not consulta:
            fallos.append("La RPC no devolvio fragmentos de la coleccion ingestsada.")

        ajenos = [f for f in consulta if f.get("coleccion_id") != COLECCION]

        print("Consulta de verificacion sobre la coleccion ingestsada:")
        for f in consulta:
            sim = f.get("similarity")
            print(f"  [{f.get('coleccion_id')}] sim={sim:.4f}  "
                  f"{str(f.get('contenido'))[:52]}")
        print()

        if ajenos:
            fallos.append(f"{len(ajenos)} fragmentos ajenos aparecieron en la consulta.")

        if fallos:
            print("[X] FAIL")
            for f in fallos:
                print(f"    - {f}")
            return 1

        print("[OK] PASS: ingesta concurrente con aislamiento de fallos por URL.")
        print(f"    - {resultado.descargadas_ok} URLs ingestadas, "
              f"{resultado.descargadas_fallidas} fallidas sin abortar el lote")
        print(f"    - {resultado.insertados} fragmentos en '{COLECCION}', "
              f"0 fuera de la coleccion")
        print(f"    - RPC filtrada devuelve {len(consulta)} fragmentos, "
              f"todos de '{COLECCION}'")
        return 0

    finally:
        limpiar(cliente, COLECCION)
        if proceso is not None:
            proceso.terminate()
            try:
                proceso.wait(timeout=10)
            except Exception:
                proceso.kill()


def _embedir_pregunta():
    """Vector de la pregunta, pasando por el control de cuota."""
    from rag.embeddings import generar_embeddings

    vectores = generar_embeddings(["capital minimo y reserva"], tamano_lote=1)
    return vectores[0]


if __name__ == "__main__":
    sys.exit(main())