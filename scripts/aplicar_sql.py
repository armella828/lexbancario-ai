"""Ejecuta un archivo .sql contra Supabase.

Uso: python scripts/aplicar_sql.py scripts/migracion_coleccion.sql

No existe un endpoint HTTP de Supabase para DDL, asi que este runner
delega en la funcion RPC `ejecutar_sql` que debe existir en el proyecto:

    create or replace function ejecutar_sql(sql_texto text)
    returns text language plpgsql as $$
    begin
      execute sql_texto;
      return 'ok';
    end;
    $$;

Crear esa funcion requiere una sola vez acceso al SQL Editor del dashboard
(o al CLI de Supabase). A partir de ahi este script permite aplicar
migraciones de forma programatica y repetible.
"""
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

FUNCION_SQL = "ejecutar_sql"


def dividir_sentencias(texto):
    """Divide en sentencias respetando $$ ... $$ (funciones PL/pgSQL)."""
    sentencias = []
    buffer = []
    dentro_dinero = False
    i = 0
    while i < len(texto):
        if texto[i] == "-" and i + 1 < len(texto) and texto[i + 1] == "-":
            while i < len(texto) and texto[i] != "\n":
                i += 1
            continue
        par = texto[i:i + 2]
        if par == "$$":
            dentro_dinero = not dentro_dinero
            buffer.append(par)
            i += 2
            continue
        if texto[i] == ";" and not dentro_dinero:
            frag = "".join(buffer).strip()
            if frag:
                sentencias.append(frag)
            buffer = []
            i += 1
            continue
        buffer.append(texto[i])
        i += 1
    resto = "".join(buffer).strip()
    if resto:
        sentencias.append(resto)
    return sentencias


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/aplicar_sql.py <archivo.sql>")
        sys.exit(1)

    ruta = sys.argv[1]
    with open(ruta, encoding="utf-8") as f:
        contenido = f.read()

    sentencias = dividir_sentencias(contenido)
    print(f"[*] {ruta}: {len(sentencias)} sentencias detectadas")

    cliente = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )

    aplicadas = 0
    for idx, sentencia in enumerate(sentencias, start=1):
        encabezado = " ".join(sentencia.split())[:88]
        try:
            cliente.rpc(FUNCION_SQL, {"sql_texto": sentencia}).execute()
            aplicadas += 1
            print(f"  [OK] {idx}/{len(sentencias)} {encabezado}")
        except Exception as e:
            print(f"  [X]  {idx}/{len(sentencias)} {encabezado}")
            print(f"      {str(e)[:400]}")
            sys.exit(1)

    print(f"[✓] {aplicadas} sentencias aplicadas.")


if __name__ == "__main__":
    main()