"""Demostracion de cero contaminacion cruzada entre colecciones (Requisito 1).

Crea dos colecciones con lexico DELIBERADAMENTE solapado pero dominios
distintos, y comprueba que una busqueda vectorial sobre cada una nunca
devuelve fragmentos de la otra.

Colecciones de prueba:
  - demo_tributaria : normativa tributaria (capital minimo, reserva)
  - demo_bancaria   : normativa bancaria (provisioning, capital exigible)

Comparten terminologia casi identica para forzar la confusion semantica:
si el filtro por coleccion no funciona, estos son justo los fragmentos que
se colarian.

Uso: python scripts/test_aislamiento.py
Sale con codigo 0 si PASS, 1 si FAIL.
"""
import os
import sys
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

TABLA = "normativa_bancaria"
RPC = "match_normativa_coleccion"
MODEL_EMBEDDING = "models/gemini-embedding-001"
DIMENSIONES = 768

COLECCION_TRIBUTARIA = "demo_tributaria"
COLECCION_BANCARIA = "demo_bancaria"
COLECCION_FANTASMA = "demo_inexistente_xyz"

# Dominios opuestos con lexico casi identico.
FRAGMENTOS_TRIBUTARIOS = [
    "El capital mínimo del régimen tributario debe adicionarse a las reservas "
    "de la entidad.",
    "La reserva mínima tributaria se calcula sobre el capital exigido por la norma.",
    "El provisioning de cartera se computa para fines de la reserva legal del impuesto.",
]

FRAGMENTOS_BANCARIOS = [
    "El capital mínimo exigible a la entidad financiera cubre el riesgo de provisioning.",
    "La reserva mínima del capital regulatorio se mide contra el capital mínimo solicitado.",
    "El provisioning de la cartera crediticia alimenta la reserva de capital del banco.",
]


class Entorno:
    """Agrupa los clientes para no depender de variables globales mutables."""

    def __init__(self):
        self.supa = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        )
        self.ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def embedir(self, textos):
        res = self.ai.models.embed_content(
            model=MODEL_EMBEDDING,
            contents=textos,
            config=types.EmbedContentConfig(output_dimensionality=DIMENSIONES),
        )
        return [e.values for e in res.embeddings]

    def consultar(self, coleccion, vector, top_k=10, umbral=0.0):
        resp = self.supa.rpc(RPC, {
            "query_embedding": vector,
            "match_threshold": umbral,
            "match_count": top_k,
            "p_coleccion_id": coleccion,
        }).execute()
        return resp.data or []


def sembrar(env, tag):
    """Inserta los fragmentos de ambas colecciones y devuelve sus ids."""
    textos = FRAGMENTOS_TRIBUTARIOS + FRAGMENTOS_BANCARIOS
    vectores = env.embedir(textos)
    corte = len(FRAGMENTOS_TRIBUTARIOS)

    filas = []
    for idx, texto in enumerate(textos):
        coleccion = COLECCION_TRIBUTARIA if idx < corte else COLECCION_BANCARIA
        filas.append({
            "documento_origen": f"TEST_{tag}.pdf",
            "organismo": "DEMO",
            "tipo_norma": "Circular / Resolucion",
            "jerarquia": "DEMO",
            "articulo_ref": f"Art. {idx + 1}",
            "contenido": texto,
            "coleccion_id": coleccion,
            "embedding": vectores[idx],
        })

    resp = env.supa.table(TABLA).insert(filas).execute()
    ids = [f["id"] for f in resp.data]
    print(f"[i] Sembradas {len(ids)} filas de prueba "
          f"({corte} '{COLECCION_TRIBUTARIA}' + {len(textos) - corte} '{COLECCION_BANCARIA}')")
    return ids


def limpiar(env, ids):
    if not ids:
        return
    env.supa.table(TABLA).delete().in_("id", ids).execute()
    print(f"[i] Filas de prueba eliminadas ({len(ids)})")


def intrusos(filas, coleccion_esperada):
    """Filas cuya coleccion_id no es la solicitada: contaminacion cruzada."""
    return [f for f in filas if f.get("coleccion_id") != coleccion_esperada]


def detalle(filas):
    for f in filas:
        sim = f.get("similarity")
        sim_txt = f"{sim:.4f}" if isinstance(sim, (int, float)) else "n/d"
        print(f"  [{str(f.get('coleccion_id')):<18}] sim={sim_txt}  "
              f"{str(f.get('contenido', ''))[:48]}")


def main():
    env = Entorno()
    tag = uuid.uuid4().hex[:8]
    ids = None
    try:
        ids = sembrar(env, tag)
        vector = env.embedir(["capital minimo y reserva"])[0]

        filas_tributaria = env.consultar(COLECCION_TRIBUTARIA, vector)
        filas_bancaria = env.consultar(COLECCION_BANCARIA, vector)
        filas_fantasma = env.consultar(COLECCION_FANTASMA, vector)

        intrusos_t = intrusos(filas_tributaria, COLECCION_TRIBUTARIA)
        intrusos_b = intrusos(filas_bancaria, COLECCION_BANCARIA)

        print("\n" + "=" * 70)
        print("PRUEBA DE AISLAMIENTO MULTI-TENANT")
        print("=" * 70)
        print(f"Consulta sobre '{COLECCION_TRIBUTARIA}':")
        print(f"  fragmentos devueltos        : {len(filas_tributaria)}")
        print(f"  intrusos de otra coleccion  : {len(intrusos_t)}")
        detalle(filas_tributaria)
        print()
        print(f"Consulta sobre '{COLECCION_BANCARIA}':")
        print(f"  fragmentos devueltos        : {len(filas_bancaria)}")
        print(f"  intrusos de otra coleccion  : {len(intrusos_b)}")
        detalle(filas_bancaria)
        print()
        print(f"Control sobre coleccion inexistente '{COLECCION_FANTASMA}':")
        print(f"  fragmentos devueltos        : {len(filas_fantasma)}")
        print("=" * 70)

        # --- Aserciones ---------------------------------------------
        fallos = []

        if not filas_tributaria:
            fallos.append(f"La RPC no devolvio filas de '{COLECCION_TRIBUTARIA}'.")
        if not filas_bancaria:
            fallos.append(f"La RPC no devolvio filas de '{COLECCION_BANCARIA}'.")

        total_intrusos = len(intrusos_t) + len(intrusos_b)
        if total_intrusos:
            fallos.append(f"{total_intrusos} fragmentos de la coleccion ajena se colaron.")

        if filas_fantasma:
            fallos.append(
                f"Una coleccion inexistente devolvio {len(filas_fantasma)} fragmentos; "
                "el filtro no restringe el espacio vectorial."
            )

        esperados = set(ids)
        obtenidos = {f["id"] for f in filas_tributaria} | {f["id"] for f in filas_bancaria}
        if not (esperados & obtenidos):
            fallos.append("Las filas sembradas no aparecen en los resultados: el test no es sensible.")

        print()
        if fallos:
            print("[X] FAIL")
            for msg in fallos:
                print(f"    - {msg}")
            return 1

        print("[OK] PASS: cero contaminacion cruzada entre colecciones.")
        print(f"    - '{COLECCION_TRIBUTARIA}': {len(filas_tributaria)} fragmentos, 0 intrusos")
        print(f"    - '{COLECCION_BANCARIA}': {len(filas_bancaria)} fragmentos, 0 intrusos")
        print(f"    - Coleccion inexistente: 0 fragmentos (espacio vectorial restringido)")
        return 0

    finally:
        limpiar(env, ids)


if __name__ == "__main__":
    sys.exit(main())