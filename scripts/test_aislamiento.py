"""Demostracion de cero contaminacion cruzada entre colecciones (Requisito 1).

Crea dos colecciones con lexico DELIBERADAMENTE solapado pero dominios
distintos, y comprueba que una busqueda vectorial sobre cada una nunca
devuelve fragmentos de la otra.

Colecciones de prueba:
  - `demo_tributaria`  : normativa tributaria (GAI, GUB, IUE)
  - `demo_bancaria`    : normativa bancaria (RNSF, capital,Provisioning)

Comparten terminologia casi identica ("capital minimo", "reserva",
"provisionamiento") para forzar la confusion semantica: si el filtro por
coleccion no funciona, estos son justo los fragmentos que se colarian.

Uso: python scripts/test_aislamiento.py
"""
import os
import sys
import uuid

from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TABLA = "normativa_bancaria"
RPC = "match_normativa_coleccion"
MODEL_EMBEDDING = "models/gemini-embedding-001"
DIMENSIONES = 768
COLECCION_A = "demo_tributaria"
COLECCION_B = "demo_bancaria"
COLECCION_INEXISTENTE = "demo_inexistente_xyz"

# Dominios opuestos con lexico casi identico.
FRAGMENTOS_A = [
    "El capital minimo del regimen tributario debe adicionarse a las reservas.",
    "La reserva minima tributaria se calculates sobre el capital exigido.",
    "El provisioning de cartera sebirn para fines de la reserva legal.",
]
FRAGMENTOS_B = [
    "El capital minimo exigible a la entidad financiera cubre el riesgo deProvisioning.",
    "La reserva minima del capital regularization se mide contra el capital minimo.",
    "El provisioning de la cartera crediticia alimenta la reserva de capital.",
]

CLIENTE_SUPABASE = None
CLIENTE_AI = None
TAG = None


def init_clientes():
    global CLIENTE_SUPABASE, CLIENTE_AI
    if CLIENTE_SUPABASE is None:
        CLIENTE_SUPABASE = create_client(SUPABASE_URL, SUPABASE_KEY)
    if CLIENTE_AI is None:
        CLIENTE_AI = genai.Client(api_key=GEMINI_API_KEY)
    return CLIENTE_SUPABASE, CLIENTE_AI


def embedir(textos):
    _, ai = init_clientes()
    res = ai.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=textos,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONES),
    )
    return [e.values for e in res.embeddings]


def sembrar():
    """Inserta los fragmentos de ambas colecciones y devuelve sus ids."""
    supa, _ = init_clientes()
    textos = FRAGMENTOS_A + FRAGMENTOS_B
    vectores = embedir(textos)

    filas = []
    for idx, texto in enumerate(textos):
        coleccion = COLECCION_A if idx < len(FRAGMENTOS_A) else COLECCION_B
        filas.append({
            "documento_origen": f"TEST_{TAG}.pdf",
            "organismo": "DEMO",
            "tipo_norma": "Circular / Resolucion",
            "jerarquia": "DEMO",
            "articulo_ref": f"Art. {idx + 1}",
            "contenido": texto,
            "coleccion_id": coleccion,
            "embedding": vectores[idx],
        })

    resp = supa.table(TABLA).insert(filas).execute()
    ids = [f["id"] for f in resp.data]
    print(f"[i] Sembradas {len(ids)} filas de prueba en '{COLECCION_A}' y '{COLECCION_B}'")
    return ids


def limpiar(ids):
    supa, _ = init_clientes()
    if not ids:
        return
    supa.table(TABLA).delete().in_("id", ids).execute()
    print(f"[i] Filas de prueba eliminadas ({len(ids)})")


def consultar(coleccion, vector, top_k=10, umbral=0.0):
    """Ejecuta la RPC aislada y devuelve las filas devueltas."""
    supa, _ = init_clientes()
    resp = supa.rpc(RPC, {
        "query_embedding": vector,
        "match_threshold": umbral,
        "match_count": top_k,
        "p_coleccion_id": coleccion,
    }).execute()
    return resp.data or []


def intrusos(filas, coleccion_esperada):
    """Filas cuya coleccion_id no es la solicitada: contaminacion."""
    return [f for f in filas if f.get("coleccion_id") != coleccion_esperada]


def main():
    global TAG
    TAG = uuid.uuid4().hex[:8]
    ids = None
    try:
        ids = sembrar()
        vector = embedir(["capital minimo y reserva"])[0]

        filas_a = consultar(COLECCION_A, vector)
        filas_b = consultar(COLECCION_B, vector)
        control = consultar(COLLECCION_INEXISTENTE, vector)

        intrusos_a = intrusos(filas_a, COLECCION_A)
        intrusos_b = intrusos(filas_b, COLECCION_B)
        intrusos_control = intrusos(control, COLECCION_INEXISTENTE)

        print("\n" + "=" * 68)
        print("PRUEBA DE AISLAMIENTO MULTI-TENANT")
        print("=" * 68)
        print(f"Coleccion solicitada : {COLECCION_A}")
        print(f"  fragmentos devueltos         : {len(filas_a)}")
        print(f"  intrusos de otra coleccion   : {len(intrusos_a)}")
        print(f"Coleccion solicitada : {COLECCION_B}")
        print(f"  fragmentos devueltos         : {len(filas_b)}")
        print(f"  intrusos de otra coleccion   : {len(intrusos_b)}")
        print("-" * 68)
        print(f"Control (coleccion inexistente): {len(control)} fragmentos")
        print("-" * 68)

        print("\nDetalle de similitudes devueltas:")
        for etiqueta, filas in ((COLECCION_A, filas_a), (COLECCION_B, filas_b)):
            for f in filas:
                sim = f.get("similarity")
                sim_txt = f"{sim:.4f}" if isinstance(sim, (int, float)) else "n/d"
                print(
                    f"  [{str(f.get('coleccion_id')):<16}] sim={sim_txt}  "
                    f"{str(f.get('contenido', ''))[:50]}"
                )

        print("\n" + "=" * 68)

        # --- Aserciones -----------------------------------------------
        fallos = []

        if not filas_a:
            fallos.append(f"La RPC no devolvio filas de '{COLECCION_A}'.")
        if not filas_b:
            fallos.append(f"La RPC no devolvio filas de '{COLECCION_B}'.")

        total_intrusos = len(intrusos_a) + len(intrusos_b)
        if total_intrusos:
            fallos.append(f"{total_intrusos} fragmentos de la coleccion ajena se colaron.")

        if control:
            fallos.append(
                f"Una coleccion inexistente devolvio {len(control)} fragmentos; "
                "el filtro no esta restringiendo el espacio vectorial."
            )

        if not ids_son_visibles(filas_a, filas_b, ids):
            fallos.append("Las filas sembradas no aparecen en los resultados.")

        if fallos:
            print("[X] FAIL")
            for f in fallos:
                print(f"    - {f}")
            return 1

        print("[✓] PASS: cero contaminacion cruzada entre colecciones.")
        print(f"    - '{COLECCION_A}': {len(filas_a)} fragmentos, 0 intrusos")
        print(f"    - '{COLECCION_B}': {len(filas_b)} fragmentos, 0 intrusos")
        print(f"    - Coleccion inexistente: 0 fragmentos (espacio vectorial restringido)")
        return 0

    finally:
        limpiar(ids)


def ids_son_visibles(filas_a, filas_b, ids):
    """Confirma que la RPC devolvio nuestras filas sembradas."""
    esperados = set(ids)
    obtenidos = {f["id"] for f in filas_a} | {f["id"] for f in filas_b}
    return bool(esperados & obtenidos)


if __name__ == "__main__":
    sys.exit(main())