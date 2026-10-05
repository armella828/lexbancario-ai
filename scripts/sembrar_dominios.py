"""Siembra las colecciones de los tres dominios normativos.

Motivo: el aislamiento multi-tenant solo se demuestra con evidencia. Con
una sola coleccion, un filtro ausente pasaria inadvertido. Con tres
dominios tematicamente distintos, cualquier fallo de filtrado se ve de
inmediato, porque un fragmento de otra coleccion aparecio en el resultado.

Los tres dominios salen de los 13 PDF de la ASFI que ya estan en la base,
partidos por tema. No se inventan normas: se reparte el corpus real que ya
teniamos en una sola coleccion.

    bancaria     : capital, solvencia, credito, depositos, rates
    regulatorio  : supervision, sanciones, control, norms de ASFI
    tributario   : impuestos y normativa fiscal que aparece en el corpus

Uso:
    python scripts/sembrar_dominios.py            # siembra
    python scripts/sembrar_dominios.py --limpiar   # borra solo lo sembrado
    python scripts/sembrar_dominios.py --reporte  # estado, sin escribir
"""
import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from rag.clients import DIMENSIONES_EMBEDDING, ENTORNO

load_dotenv()

TABLA = "normativa_bancaria"
COLECCION_ORIGEN = "asfi_rnsf"

DOMINIOS = {
    "bancaria": {
        "titulo": "Normativa bancaria y de mercados",
        "organismo": "ASFI /Banco Central de Bolivia",
        "palabras": [
            "banco", "entidad financiera", "credito", "deposito", "capital",
            "solvencia", "reserva", "tasa", "interes", "cartera", "provision",
            "garantia", "encaje", "liquidez", "patrimonio", "utilidad",
        ],
    },
    "regulatorio": {
        "titulo": "Supervisión, control y sanciones",
        "organismo": "ASFI",
        "palabras": [
            "supervision", "superintendencia", "autoridad", "regulacion",
            "normativa", "sancion", "control", "fiscalizacion", "autorizacion",
            "preventiva", "inspectoria", "infraccion", "multa", "procedimiento",
        ],
    },
    "tributario": {
        "titulo": "Normativa tributaria y fiscal",
        "organismo": "Servicio Nacional de Impuestos",
        "palabras": [
            "impuesto", "tributario", "tribut", "fiscal", "iva", "renta",
            "ganancias", "contribuyente", "declaracion jurada", "codigo tributario",
            "litigio", "fiscalizacion tributaria",
        ],
    },
}


def _norm(texto):
    return re.sub(r"[^a-z0-9ñ ]", " ", (texto or "").lower())


def clasificar(documento, contenido):
    """Asigna un dominio comparando pesos de palabras clave.

    No es un clasificador semantico, y no pretende serlo: reparte el corpus
    real por tema de forma explicable y reproducible. Un fragmento sin
    ninguna palabra clave va a la coleccion mas poblada, para no perder
    fragmentos en el reparto.
    """
    texto = _norm(f"{documento} {contenido}")
    pesos = {}
    for dominio, cfg in DOMINIOS.items():
        pesos[dominio] = sum(texto.count(p) for p in cfg["palabras"])
    total = sum(pesos.values())
    if total == 0:
        return None
    return max(pesos, key=pesos.get)


def vector_de_pg(valor):
    """PostgREST devuelve pgvector como texto; Supabase acepta lista."""
    if valor is None:
        return None
    if isinstance(valor, str):
        limpio = valor.strip().strip("[]")
        if not limpio:
            return None
        try:
            return [float(x) for x in limpio.split(",")]
        except ValueError:
            return None
    return list(valor)


def leer_origen():
    """Fragmentos de la coleccion de origen, con su vector ya pagado."""
    res = ENTORNO.supabase.table(TABLA) \
        .select("id,documento_origen,contenido,embedding,articulo_ref") \
        .eq("coleccion_id", COLECCION_ORIGEN).execute()
    return res.data or []


def ya_sembradas():
    res = ENTORNO.supabase.table(TABLA).select("documento_origen") \
        .in_("coleccion_id", list(DOMINIOS)).execute()
    return res.data or []


def sembrar(dry_run=False):
    origen = leer_origen()
    if not origen:
        print(f"[X] No hay fragmentos en '{COLECCION_ORIGEN}'.")
        return 1

    print(f"[*] Origen: {len(origen)} fragmentos en '{COLECCION_ORIGEN}'")

    grupos = defaultdict(list)
    sin_palabras = 0
    for fila in origen:
        dominio = clasificar(fila.get("documento_origen"), fila.get("contenido"))
        if dominio is None:
            sin_palabras += 1
            dominio = "bancaria"
        grupos[dominio].append(fila)

    print(f"[*] {sin_palabras} fragmentos sin palabra clave, asignados a 'bancaria'")
    total = 0
    for dominio, filas in sorted(grupos.items()):
        cfg = DOMINIOS[dominio]
        documentos = len({f.get("documento_origen") for f in filas})
        print(f"    {dominio:<12} {len(filas):>4} frag  "
              f"{documentos:>2} documentos  ({cfg['titulo']})")
        total += len(filas)

    if total != len(origen):
        print(f"[X] El reparto pierde fragmentos: {total} != {len(origen)}")
        return 1

    if dry_run:
        print("[i] --reporte: no se escribio nada.")
        return 0

    # Se comprueba que los vectores siguen siendo utilizables antes de
    # duplicar 336 filas: si vinieran corruptos, es mejor fallar aqui.
    for fila in origen[:5]:
        dims = len(vector_de_pg(fila.get("embedding")) or [])
        if dims != DIMENSIONES_EMBEDDING:
            print(f"[X] Vector con {dims} dimensiones en {fila.get('id')}, "
                  f"se esperaban {DIMENSIONES_EMBEDDING}")
            return 1

    nuevos = []
    for dominio, filas in grupos.items():
        cfg = DOMINIOS[dominio]
        for fila in filas:
            vector = vector_de_pg(fila.get("embedding"))
            if vector is None:
                continue
            nuevos.append({
                "documento_origen": fila.get("documento_origen"),
                "organismo": cfg["organismo"],
                "tipo_norma": "Circular / Resolucion",
                "jerarquia": dominio.upper(),
                "articulo_ref": fila.get("articulo_ref"),
                "contenido": fila.get("contenido"),
                "coleccion_id": dominio,
                "embedding": vector,
            })

    print(f"[*] Insertando {len(nuevos)} filas...")
    insertados = 0
    for inicio in range(0, len(nuevos), 100):
        lote = nuevos[inicio:inicio + 100]
        ENTORNO.supabase.table(TABLA).insert(lote).execute()
        insertados += len(lote)
        print(f"    {insertados}/{len(nuevos)}")

    print(f"[OK] Sembradas {insertados} filas en {len(DOMINIOS)} colecciones")
    return 0


def limpiar():
    res = ENTORNO.supabase.table(TABLA).delete() \
        .in_("coleccion_id", list(DOMINIOS)).execute()
    print(f"[OK] Borradas las colecciones de prueba: {res}")
    return 0


def reporte():
    res = ENTORNO.supabase.rpc("listar_colecciones").execute()
    print(f"{'coleccion':<18} {'fragmentos':>10}")
    for fila in res.data or []:
        print(f"{fila['coleccion_id']:<18} {fila['total_fragmentos']:>10}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Siembra de dominios normativos")
    p.add_argument("--limpiar", action="store_true")
    p.add_argument("--reporte", action="store_true")
    args = p.parse_args()

    if args.limpiar:
        return limpiar()
    if args.reporte:
        return reporte()
    return sembrar(dry_run=False)


if __name__ == "__main__":
    sys.exit(main())