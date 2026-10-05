"""Respalda la tabla normativa_bancaria a un JSON local antes de migrar.

Uso: python scripts/respaldar_datos.py
Salida: respaldos/normativa_bancaria_<timestamp>.json
"""
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

DIR_RESPALDOS = "respaldos"
TABLA = "normativa_bancaria"


def main():
    cliente = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )

    print(f"[*] Leyendo '{TABLA}' completo desde Supabase...")
    res = (
        cliente.table(TABLA)
        .select("*", count="exact")
        .order("id", desc=False)
        .execute()
    )

    filas = res.data or []
    total = getattr(res, "count", None)
    if total is None:
        total = len(filas)

    os.makedirs(DIR_RESPALDOS, exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(DIR_RESPALDOS, f"normativa_bancaria_{sello}.json")

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tabla": TABLA,
                "generado_en": datetime.now().isoformat(),
                "total_filas": total,
                "filas": filas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tam_mb = os.path.getsize(ruta) / (1024 * 1024)
    print(f"[✓] {total} filas respaldadas en {ruta} ({tam_mb:.2f} MB)")


if __name__ == "__main__":
    main()