"""Prueba de la insercion masiva por lotes (Requisito 3).

Comprueba dos cosas distintas:

1. La logica de troceado del INSERT, con una tabla falsa: que 450 filas se
   partan en 200/200/50 y no en un unico INSERT.
2. Que Supabase acepta de verdad esos lotes con vectores de 768
   dimensiones, y que las filas quedan todas recuperables.

La segunda parte si toca la base real, pero usa una coleccion desechable y
la borra al terminar, tambien si algo falla.

Ejecutar:  python scripts/probar_insercion.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.clients import DIMENSIONES_EMBEDDING, ENTORNO
from rag.pipeline import INSERCION_LOTE

COLECCION = "demo_insercion_masiva"
TABLA = "normativa_bancaria"

fallos = []
ok = 0


def comprobar(descripcion, condicion, detalle=""):
    global ok
    if condicion:
        ok += 1
        print(f"  [OK] {descripcion}")
    else:
        fallos.append(descripcion)
        print(f"  [X]  {descripcion} {detalle}")


class ConsultaFalsa:
    """Resultado encadenable de `insert()`."""

    def __init__(self, registro):
        self._registro = registro

    def execute(self):
        self._registro["inserts"] += 1
        return self


class TablaFalsa:
    def __init__(self, registro):
        self._registro = registro

    def insert(self, filas):
        # El registro va aqui: el pipeline llama `table().insert(filas).execute()`,
        # asi que `insert` se invoca sobre la tabla, no sobre la consulta.
        self._registro["lotes"].append(len(filas))
        self._registro["filas"].extend(filas)
        return ConsultaFalsa(self._registro)


class SupabaseFalso:
    def __init__(self, registro):
        self.registro = registro

    def table(self, nombre):
        return TablaFalsa(self.registro)


print("=" * 74)
print("PRUEBA DE INSERCION MASIVA")
print("=" * 74)

# ------------------------------------------------- 1. logica de troceado
print("\n[1] Troceado del INSERT (tabla falsa)")
from rag import pipeline as pl
from rag.pipeline import INSERCION_LOTE, insertar_filas

registro = {"lotes": [], "filas": [], "inserts": 0}
original = pl.ENTORNO._supabase
pl.ENTORNO._supabase = SupabaseFalso(registro)
try:
    insertar_filas([{"n": i} for i in range(450)])
    comprobar("450 filas se parten en 3 lotes", registro["inserts"] == 3, f"-> {registro['lotes']}")
    comprobar("Los lotes son 200/200/50", registro["lotes"] == [200, 200, 50], f"-> {registro['lotes']}")
    comprobar("Ningun lote excede el limite", all(n <= INSERCION_LOTE for n in registro["lotes"]))
    comprobar("No se pierde ni duplica ninguna fila", len(registro["filas"]) == 450)

    registro["lotes"].clear()
    registro["filas"].clear()
    registro["inserts"] = 0
    insertar_filas([{"n": i} for i in range(104)])
    comprobar("104 filas caben en un solo lote", registro["lotes"] == [104], f"-> {registro['lotes']}")

    registro["lotes"].clear()
    registro["inserts"] = 0
    comprobar("Una lista vacia no lanza nada", insertar_filas([]) == 0 and registro["inserts"] == 0)
finally:
    pl.ENTORNO._supabase = original

# ------------------------------------------------ 2. contra Supabase real
print("\n[2] Lotes reales en Supabase")
try:
    supabase = ENTORNO.supabase
except Exception as e:
    comprobar("Supabase accesible", False, f"-> {type(e).__name__}: {e}")
    supabase = None

if supabase is not None:
    supabase.table(TABLA).delete().eq("coleccion_id", COLECCION).execute()
    try:
        vector = [0.0125] * DIMENSIONES_EMBEDDING
        filas = [{
            "documento_origen": f"https://ejemplo.test/{i}",
            "organismo": "Prueba",
            "tipo_norma": "Documento web",
            "jerarquia": "Web",
            "articulo_ref": f"Fragmento {i}",
            "contenido": "texto de prueba para insercion masiva " * 5,
            "coleccion_id": COLECCION,
            "embedding": vector,
        } for i in range(450)]

        t0 = time.perf_counter()
        enviados = insertar_filas(filas)
        elapsed = time.perf_counter() - t0

        comprobar("450 filas insertadas por Supabase", enviados == 450, f"-> {enviados}")

        total = supabase.table(TABLA).select("id", count="exact") \
            .eq("coleccion_id", COLECCION).execute()
        comprobar("Las 450 filas se pueden contar", getattr(total, "count", 0) == 450,
                  f"-> {getattr(total, 'count', 0)}")

        muestra = supabase.table(TABLA).select("embedding") \
            .eq("coleccion_id", COLECCION).limit(1).execute()
        if muestra.data:
            crudo = muestra.data[0].get("embedding")
            # PostgREST devuelve la columna pgvector como texto
            # "[0.01,0.02,...]", no como lista, asi que hay que parsearla.
            if isinstance(crudo, str):
                dim = len(crudo.strip().strip("[]").split(","))
            else:
                dim = len(crudo or [])
            comprobar(f"Los vectores guardan {DIMENSIONES_EMBEDDING} dimensiones",
                      dim == DIMENSIONES_EMBEDDING, f"-> {dim}")
        else:
            comprobar("Se puede leer un vector de vuelta", False, "-> sin filas")

        print(f"      (450 filas en 3 lotes en {elapsed:.2f}s)")
    except Exception as e:
        comprobar("Insercion masiva sin errores", False, f"-> {type(e).__name__}: {e}")
    finally:
        supabase.table(TABLA).delete().eq("coleccion_id", COLECCION).execute()
        resto = supabase.table(TABLA).select("id", count="exact") \
            .eq("coleccion_id", COLECCION).execute()
        comprobar("Limpieza: 0 filas residuales", getattr(resto, "count", 0) == 0,
                  f"-> {getattr(resto, 'count', 0)}")

print("\n" + "=" * 74)
if fallos:
    print(f"[X] FAIL: {len(fallos)} de {ok + len(fallos)} comprobaciones fallidas")
    for f in fallos:
        print(f"    - {f}")
    sys.exit(1)
print(f"[OK] PASS: {ok} comprobaciones de insercion masiva.")
sys.exit(0)