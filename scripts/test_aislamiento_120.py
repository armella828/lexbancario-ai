"""Prueba de aislamiento multi-tenant a escala de consulta.

Por que 120 consultas y no 6: con pocas consultas, un filtro ausente
puede pasar por casualidad, porque el corpus es pequeno y los fragmentos
de una coleccion pueden no aparecer nunca en el top-k. Repetir sobre
distintos vectores de consulta y dosidos umbrales convierte el aislamiento
en algo comprobable en lugar de accidental.

Protocolo:
  * 40 consultas por cada coleccion sembrada
  * match_threshold = -1.0, para no descartar resultados por similitud
  * match_count = 20
  * ademas, consultas a colecciones inexistentes, que deben dar 0 filas

Requiere cuota de embeddings para las preguntas. Si esta agotada, la
prueba se omite con un mensaje claro: es una condicion del entorno, no un
fallo del aislamiento.

Uso:  python scripts/test_aislamiento_120.py
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

TABLA = "normativa_bancaria"
RPC = "match_normativa_coleccion"
# RPC original, sin filtro de coleccion. Solo se usa como control negativo:
# sirve para comprobar que la prueba tiene sensibilidad real.
RPC_SIN_FILTRO = "match_normativa"
COLECCIONES_FANTASMA = ["no_existe_1", "no_existe_2", "coleccion_vacia"]
MATCH_COUNT = 20
THRESHOLD = -1.0

PREGUNTAS_POR_DOMINIO = {
    "bancaria": [
        "capital minimo de las entidades de credito",
        "reserva legal exigible a los bancos",
        "provision para cartera de creditos",
        "tasa de interes de los depositos",
        "encaje legal del banco central",
        "solvencia y adequacy de capital",
        "garantias exigidas en prestamos",
        "utilidades y patrimonio computable",
        "requisitos de liquidez del banco",
        "credito de consumo y sus limites",
    ],
    "regulatorio": [
        "supervision de las entidades financieras",
        "sanciones por infracciones a la normativa",
        "procedimiento de control de la superintendencia",
        "autorizacion para funcionar como banco",
        "inspeccion y fiscalizacion bancaria",
        "normativa de supervision financiera",
        "preventiva y medidas correctivas",
        "multas por incumplimientos",
        "infracciones administrativas",
        "reglamento de supervision",
    ],
    "tributario": [
        "impuesto sobre la renta de sociedades",
        "declaracion jurada de impuestos",
        "contribuyentes del regimen tributario",
        "iva y sus obligaciones",
        "codigo tributario del pais",
        "ganancias y utilidades gravadas",
        "fiscalizacion tributaria",
        "litigios y sanciones tributarias",
        "patrimonio neto imponible",
        "obligaciones del sujeto pasivo",
    ],
}


def obtener_colecciones():
    from rag.clients import ENTORNO

    res = ENTORNO.supabase.rpc("listar_colecciones").execute()
    disponibles = {f["coleccion_id"]: f["total_fragmentos"]
                   for f in (res.data or [])}
    return disponibles


def consultas_de(colecciones, por_dominio, n_por_dominio):
    """Construye el juego de consultas, rotando las preguntas disponibles."""
    plan = []
    for coleccion in colecciones:
        preguntas = por_dominio.get(coleccion)
        if not preguntas:
            # Sin guion propio, se usan preguntas tematicas genericas.
            preguntas = PREGUNTAS_POR_DOMINIO["bancaria"] + \
                PREGUNTAS_POR_DOMINIO["regulatorio"] + \
                PREGUNTAS_POR_DOMINIO["tributario"]
        for i in range(n_por_dominio):
            plan.append((coleccion, preguntas[i % len(preguntas)]))
    return plan


def control_negativo(suabase, coleccion, vector, match_count):
    """Comprueba que la RPC SIN filtro de coleccion SIARIA devolviendo mezcla.

    Sin esto, la prueba de aislamiento no demuestra nada: si la RPC
    devolviera cero filas siempre, pasaria igual. Al ejecutar la version
    sin `p_coleccion_id` y contar cuantas filas vienen de otras colecciones,
    se mide la sensibilidad real de la prueba. Si aqui no hay
    contaminacion, el corpus no sirve para detectar un fallo de filtrado y
    hay que decirlo, no publicarlo como un PASS.
    """
    res = suabase.rpc(RPC_SIN_FILTRO, {
        "query_embedding": vector,
        "match_threshold": THRESHOLD,
        "match_count": match_count,
    }).execute()
    filas = res.data or []
    ajenas = [f for f in filas if f.get("coleccion_id") != coleccion]
    return len(filas), ajenas


def main():
    p = argparse.ArgumentParser(description="Aislamiento multi-tenant, 120 consultas")
    p.add_argument("--por-dominio", type=int, default=40)
    p.add_argument("--match-count", type=int, default=MATCH_COUNT)
    p.add_argument("--umbral", type=float, default=THRESHOLD)
    args = p.parse_args()

    from rag.clients import ENTORNO

    disponibles = obtener_colecciones()
    objetivo = [c for c in disponibles if c in PREGUNTAS_POR_DOMINIO]
    if not objetivo:
        print("[X] Ninguna coleccion sembrada de las 3 esperadas.")
        print(f"    Hay en la base: {sorted(disponibles)}")
        print("    Ejecuta primero:  python scripts/sembrar_dominios.py")
        return 1

    print("=" * 74)
    print("AISLAMIENTO MULTI-TENANT A ESCALA")
    print("=" * 74)
    print(f"[*] Colecciones: {objetivo}")
    print(f"[*] Fragmentos por coleccion: "
          f"{ {c: disponibles[c] for c in objetivo} }")
    total_consultas = len(objetivo) * args.por_dominio
    print(f"[*] {total_consultas} consultas, match_count={args.match_count}, "
          f"match_threshold={args.umbral}")
    print()

    plan = consultas_de(objetivo, PREGUNTAS_POR_DOMINIO, args.por_dominio)

    # Los embeddings de las preguntas se cachean: las 120 consultas
    # comparten few textos, y ademas la prueba se repite al recalibrar.
    cache = CacheEmbeddings().cargar()
    try:
        textos = [pregunta for _, pregunta in plan]
        vectores = generar_embeddings(textos, cache=cache)
    except CuotaAgotada as exc:
        print("[--] PRUEBA OMITIDA: cuota de embeddings agotada.")
        print(f"     {exc}")
        print("\n     Reejecutar cuando la cuota se renueve.")
        return 0

    if len(vectores) != len(plan):
        print(f"[X] {len(plan)} consultas pero {len(vectores)} vectores")
        return 1

    suabase = ENTORNO.supabase
    intrusiones = 0
    filas_totales = 0
    resultados_vacios = 0
    por_coleccion = {c: 0 for c in objetivo}
    t0 = time.perf_counter()

    for i, ((coleccion, _), vector) in enumerate(zip(plan, vectores), start=1):
        res = suabase.rpc(RPC, {
            "query_embedding": vector,
            "match_threshold": args.umbral,
            "match_count": args.match_count,
            "p_coleccion_id": coleccion,
        }).execute()
        filas = res.data or []
        filas_totales += len(filas)
        por_coleccion[coleccion] += len(filas)
        if not filas:
            resultados_vacios += 1
        for fila in filas:
            # La comprobacion real: ninguna fila puede venir de otra
            # coleccion. Esto es lo que un filtro ausente delata.
            if fila.get("coleccion_id") != coleccion:
                intrusiones += 1
        if i % 20 == 0:
            print(f"    {i}/{total_consultas} consultas, "
                  f"{intrusiones} intrusiones hasta ahora")

    elapsed = time.perf_counter() - t0

    # Consultas a colecciones inexistentes: deben devolver 0 filas.
    fantasma_vacias = 0
    for nombre in COLECCIONES_FANTASMA:
        res = suabase.rpc(RPC, {
            "query_embedding": vectores[0],
            "match_threshold": args.umbral,
            "match_count": args.match_count,
            "p_coleccion_id": nombre,
        }).execute()
        filas = res.data or []
        fantasma_vacias += len(filas)
        if filas:
            print(f"    [X] '{nombre}' devolvio {len(filas)} filas")

    print()
    print(f"Consultas realizadas        : {total_consultas}")
    print(f"Filas recuperadas           : {filas_totales}")
    print(f"Filas de otra coleccion     : {intrusiones}")
    print(f"Consultas con 0 resultados   : {resultados_vacios}")
    print(f"Filas en colecciones inexistentes: {fantasma_vacias}")
    print(f"Tiempo total                : {elapsed:.1f}s")
    print()
    for coleccion in objetivo:
        print(f"  {coleccion:<14} {por_coleccion[coleccion]:>5} filas "
              f"({disponibles[coleccion]} fragmentos sembrados)")

    # Control negativo: mide si esta prueba PODRIA detectar una fuga.
    print()
    print("Control negativo (RPC sin filtro de coleccion):")
    sensibilidad = 0
    for i, coleccion in enumerate(objetivo):
        total, ajenas = control_negativo(
            suabase, coleccion, vectores[i * args.por_dominio], args.match_count
        )
        sensibilidad += len(ajenas)
        print(f"  desde {coleccion:<12} -> {total:>2} filas sin filtrar, "
              f"{len(ajenas)} de otras colecciones")
    if sensibilidad == 0:
        print("  [!] ATENCION: la busqueda sin filtro NO devuelve filas de")
        print("      otras colecciones. Con este corpus, la prueba de")
        print("      aislamiento no puede detectar un fallo de filtrado:")
        print("      pasaria aunque el filtro no existiera.")

    fallos = []
    if intrusiones > 0:
        fallos.append(f"{intrusiones} filas de otra coleccion")
    if fantasma_vacias > 0:
        fallos.append(f"{fantasma_vacias} filas en colecciones inexistentes")
    if total_consultas != filas_totales and filas_totales == 0:
        fallos.append("Ninguna consulta devolvio filas; el aislamiento "
                      "no seria interpretable")

    print()
    if fallos:
        print("[X] FAIL: se detecto contaminacion cruzada")
        for f in fallos:
            print(f"    - {f}")
        return 1
    print(f"[OK] PASS: {total_consultas} consultas sin contaminacion cruzada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())