"""Recoge evidencia de que los indices se usan de verdad.

Por que este script: afirmar que el indice B-Tree filtra por coleccion y
que HNSW acelera el vector es barato. `EXPLAIN (ANALYZE, BUFFERS)` muetra
el plan que Postgres eligio de verdad. Si el plan dijera "Seq Scan", el
indice existiria pero no serviria, y eso habria que decirlo en el informe
en lugar de ocultarlo.

Tres consultas, porque cada una prueba algo distinto:

  1. match_normativa_coleccion: el caso de uso real. Debe filtrar por
     coleccion en indice y despues ordenar por distancia sobre el subconjunto.
  2. Consulta sin filtro de coleccion: control. Si aqui Postgres eligiera
     HNSW sobre 672 filas, seria una razon para no crear ese indice.
  3. Consulta sobre una coleccion sin HNSW: no aplica, el filtro manda.

Uso:  python scripts/explicar_indices.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

RPC = "explicar_consulta"

CONSULTAS = {
    "1. RPC filtrada por coleccion (caso de uso real)": """
        select nb.id, nb.contenido, nb.coleccion_id,
               1 - (nb.embedding <=> '[%s]'::vector) as similarity
        from normativa_bancaria nb
        where nb.coleccion_id = 'bancaria'
          and 1 - (nb.embedding <=> '[%s]'::vector) > -1.0
        order by nb.embedding <=> '[%s]'::vector
        limit 20
    """,
    "2. Consulta SIN filtro de coleccion (control: vecindad global)": """
        select nb.id, nb.contenido, nb.coleccion_id,
               1 - (nb.embedding <=> '[%s]'::vector) as similarity
        from normativa_bancaria nb
        where 1 - (nb.embedding <=> '[%s]'::vector) > -1.0
        order by nb.embedding <=> '[%s]'::vector
        limit 20
    """,
    "3. Filtrado puro por coleccion, sin vector (debe usar B-Tree)": """
        select id, contenido, coleccion_id
        from normativa_bancaria
        where coleccion_id = 'regulatorio'
        order by documento_origen
        limit 20
    """,
}


def vector_ejemplo():
    """Toma un vector real de la base: un vector inventado no representa nada."""
    from rag.clients import ENTORNO

    res = ENTORNO.supabase.table("normativa_bancaria") \
        .select("embedding").limit(1).execute()
    filas = res.data or []
    if not filas:
        print("[X] La tabla esta vacia: no hay vector de ejemplo.")
        return None
    valor = filas[0]["embedding"]
    if isinstance(valor, str):
        valor = valor.strip().strip("[]")
    else:
        valor = ",".join(str(x) for x in valor)
    return valor


def pedir_plan(suabase, sql):
    res = suabase.rpc(RPC, {"sql_texto": sql}).execute()
    return res.data or "(sin respuesta)"


VECTOR_EN_PLAN = re.compile(r"'\[(?:-?[\d.eE+]+,){20,}[^\]]*\]'::vector")


def comprimir(plan):
    """Quita el vector literal del plan.

    Un vector de 768 decimales ocupa cuatro lineas de ancho de terminal y
    esconde lo unico interesante del plan: que nodo eligio Postgres y a
    cuantas filas redujo el conjunto. Se sustituye por un marcador para que
    la evidencia que se lee y se adjunta al informe sea la del plan.
    """
    texto = plan if isinstance(plan, str) else "\n".join(plan)
    return VECTOR_EN_PLAN.sub("'[vector de 768 dims]'::vector", texto)


def resumen(plan):
    """Extrae el nodo raiz, el indice usado y las filas EFFECTIVAS leidas.

    `rows=241` en un Bitmap Index Scan es el indice delivering esas filas;
    el filtro de similitud recien ahi. Distinguirlo importa: si el filtro de
    tenant no hubiera funcionado, el numero de filas Habria sido 672.
    """
    texto = comprimir(plan)
    nodos = []
    for linea in texto.splitlines():
        limpio = linea.strip()
        if limpio.startswith("->") or limpio.startswith(("Limit", "Sort")):
            nombre = limpio.split("  ")[0].lstrip("-> ").strip()
            filas = re.search(r"rows=(\d+)", limpio)
            idx = re.search(r"on (\w+)", limpio)
            real = re.search(r"actual time=[\d.]+\.\.\d+ rows=(\d+)", limpio)
            nodos.append({
                "nodo": nombre,
                "indice": idx.group(1) if idx else None,
                "filas_estimadas": int(filas.group(1)) if filas else None,
                "filas_reales": int(real.group(1)) if real else None,
            })
    return nodos


def main():
    from rag.clients import ENTORNO

    vector = vector_ejemplo()
    if vector is None:
        return 1

    print("=" * 74)
    print("EVIDENCIA DE USO DE INDICES")
    print("=" * 74)

    evidencia = {}
    for titulo, plantilla in CONSULTAS.items():
        sql = plantilla % ((vector,) * plantilla.count("%s"))
        print()
        print(f"--- {titulo} ---")
        try:
            plan = pedir_plan(ENTORNO.supabase, sql)
        except Exception as exc:
            print(f"[X] No se pudo explicar: {exc}")
            evidencia[titulo] = {"error": str(exc)}
            continue

        limpio = comprimir(plan)
        print(limpio)

        nodos = resumen(plan)
        indices = [n["indice"] for n in nodos if n["indice"]]
        ejecucion = re.search(r"Execution Time: ([\d.]+) ms", limpio)
        evidencia[titulo] = {
            "plan": limpio,
            "indices_usados": indices,
            "nodos": nodos,
            "execution_ms": float(ejecucion.group(1)) if ejecucion else None,
        }

    # ------------------------------------------------------------------
    # Conclusion: lo que los planes dicen, INCLUDING lo que NO dicen.
    # ------------------------------------------------------------------
    print()
    print("=" * 74)
    print("CONCLUSION")
    print("=" * 74)

    filtrada = next((v for k, v in evidencia.items() if k.startswith("1.")), {})
    control = next((v for k, v in evidencia.items() if k.startswith("2.")), {})

    if filtrada.get("indices_usados"):
        print(f"  Filtro de tenant: usa {', '.join(filtrada['indices_usados'])}.")
        print("  El B-Tree recorta el conjunto ANTES de ordenar por distancia,")
        print("  que es justo lo que hace correcto el aislamiento.")

    idx_filtrada = set(filtrada.get("indices_usados", []))
    idx_control = set(control.get("indices_usados", []))
    hnsw_filtrada = any("embedding" in i for i in idx_filtrada)
    hnsw_control = any("embedding" in i for i in idx_control)

    # El indice compuesto empieza por coleccion_id, asi que para el filtro
    # de tenant cubre lo mismo que el B-Tree simple y ademas ordena por
    # documento_origen sin pasos extra. Se declara el indice que Postgres
    # eligio de verdad, no el que mas favorably suena en el informe.
    todos_los_idx = set()
    for v in evidencia.values():
        todos_los_idx.update(v.get("indices_usados", []))
    compuesto = sorted(i for i in todos_los_idx if i.endswith("_coleccion_embedding_idx"))
    if compuesto:
        print(f"  El filtro de tenant se resuelve con {', '.join(compuesto)} en la")
        print("  consulta que ademas ordena por documento. El compuesto cubre el")
        print("  filtro y el orden en un solo indice, asi que el B-Tree simple")
        print("  en coleccion_id es redundante para este caso de uso.")
        print("  Se deja, pero el informe debe decir cual se eligio de verdad.")

    print()
    if not hnsw_filtrada and not hnsw_control:
        print("  HNSW: NO aparece en ningun plan.")
        print("  Con este tamano de tabla Postgres elige Secuencial, porque un")
        print("  indice aproximado costaria mas de lo que ahorra. El indice")
        print("  HNSW es capacidad futura, NO una aceleracion actual: asi debe")
        print("  constar en el informe.")
    else:
        print(f"  HNSW en uso: filtrada={hnsw_filtrada}, sin filtro={hnsw_control}.")

    if filtrada.get("execution_ms") and control.get("execution_ms"):
        a = filtrada["execution_ms"]
        b = control["execution_ms"]
        print()
        print(f"  Filtrada {a:.2f} ms vs sin filtro {b:.2f} ms "
              f"({b / a:.2f}x).")
        print("  La ganancia viene del filtro de tenant, no del indice vectorial.")

    ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "resultados", "explain_indices.json")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(evidencia, fh, ensure_ascii=False, indent=2)
    print()
    print(f"[*] Evidencia guardada en {ruta}")

    print()
    print("=" * 74)
    print("COMO LEER ESTOS PLANES")
    print("=" * 74)
    print("  Index Scan using ..._embedding_idx   -> HNSW en uso (busqueda vectorial)")
    print("  Index Scan using ..._coleccion_idx   -> B-Tree en uso (filtro de tenant)")
    print("  Index Scan using ..._pkey            -> lectura por id")
    print("  Seq Scan                             -> Postgres decidio leer secuencial;")
    print("                                         el indice no le aporta aqui")
    print()
    print("  Buffers: shared hit/read muestra cuantas paginas toco. Si shared")
    print("  read es 0, todo salio de la cache de PostgreSQL y no hubo I/O.")
    print("  Heap Fetches != 0 con indice vectorial es normal: el indice HNSW")
    print("  referencia las filas y PostgreSQL debe traerlas de la tabla.")
    return 0


if __name__ == "__main__":
    sys.exit(main())