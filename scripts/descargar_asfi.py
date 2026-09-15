import os
import urllib3
import requests
import pymupdf
from urllib.parse import urljoin

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PDF_MAESTRO = "documentos_pdf/Circulares.pdf"
DESTINO = "documentos_pdf"
BASE_URL = "https://servdmzw.asfi.gob.bo/circular/Circulares/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

def extraer_enlaces(ruta_pdf):
    doc = pymupdf.open(ruta_pdf)
    circulares = set()
    otros = set()

    for pag in doc:
        for l in pag.get_links():
            f = l.get("file")
            if f and isinstance(f, str) and f.lower().endswith(".pdf"):
                nombre = os.path.basename(f.replace("\\", "/"))
                if nombre.upper().startswith("ASFI") or "CIRCULAR" in nombre.upper():
                    circulares.add((f, nombre))
                else:
                    otros.add((f, nombre))

    print(f"[✓] Circulares ASFI prioritarias identificadas: {len(circulares)}")
    print(f"[✓] Textos y anexos complementarios: {len(otros)}")
    return sorted(list(circulares)), sorted(list(otros))

def descargar_lista(items, limite=30):
    descargados = 0
    for idx, (rel_path, nombre) in enumerate(items[:limite], start=1):
        ruta_local = os.path.join(DESTINO, nombre)
        if os.path.exists(ruta_local):
            print(f"[{idx}/{limite}] [=] Ya existe: {nombre}")
            continue

        url = urljoin(BASE_URL, rel_path.replace("\\", "/"))
        print(f"[{idx}/{limite}] [↓] Descargando: {nombre}...")
        try:
            r = requests.get(url, headers=HEADERS, verify=False, timeout=25, stream=True)
            if r.status_code == 200:
                with open(ruta_local, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                descargados += 1
                print(f"       [✓] OK")
            else:
                print(f"       [!] HTTP {r.status_code}")
        except Exception as e:
            print(f"       [X] Error: {e}")

    print(f"\n[✓] Descarga completada: {descargados} archivos guardados.")

if __name__ == "__main__":
    circulares, otros = extraer_enlaces(PDF_MAESTRO)
    # Descargar un lote de 25 circulares normativas clave
    descargar_lista(circulares, limite=25)
