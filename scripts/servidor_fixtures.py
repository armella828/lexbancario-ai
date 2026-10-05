"""Servidor local de documentos de prueba para el benchmark de concurrencia.

Sirve 10 paginas HTML generadas con tamano controlado y latencia artificial
configurable. El objetivo es que el corpus sea identico en cada corrida: si
las mediciones se hicieran contra sitios reales, la variabilidad de DNS, TLS
y del servidor remoto se mezclaría con el efecto del paralelismo y el speedup
medido no seria atribuible al pipeline.

Rutas disponibles:
  /doc/{n}        documento normal de tamano estandar (n entre 1 y 10)
  /roto           responde 404, para probar el aislamiento de fallos
  /lento          tarda en responder, para probar el timeout
  /salud          comprobacion de estado
"""
import argparse
import asyncio
import random

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

app = FastAPI(title="Servidor de fixtures para benchmark RAG")

TAMANO_PALABRAS = 1200
LATENCIA_SEGUNDOS = 0.35

TEMAS = [
    ("Reglamento de Operaciones Bancarias", "ASFI", "Resolucion"),
    ("Normativa de Capital y Solvencia", "ASFI", "Circular"),
    ("Reglamento para Cooperativas Financieras", "ASFI", "Resolucion"),
    ("Normativa de Lavado de Activos", "ASFI", "Circular"),
    ("Reglamento de Cuentas de Ahorro", "ASFI", "Resolucion"),
    ("Normativa de Provisionamiento", "ASFI", "Circular"),
    ("Reglamento de Garante de Depositos", "ASFI", "Resolucion"),
    ("Normativa de Servicios Financieros Digitales", "ASFI", "Circular"),
    ("Reglamento de Adminstraccion de Riesgos", "ASFI", "Resolucion"),
    ("Normativa de Supervision Bancaria", "ASFI", "Circular"),
]

ORACIONES = [
    "El capital mínimo exigible a la entidad financiera se determina en función "
    "del volumen de activos y del perfil de riesgo de sus operaciones.",
    "La reserva mínima deberá permanecer disponible para absorber pérdidas "
    "no esperadas bila generadas por el ciclo de negocio.",
    "El provisioning de la cartera crediticia se calcula según la categoría de "
    "riesgo asignada al deudor y la naturaleza de la garantía.",
    "Las entidades deberán reportar trimestralmente el nivel de cobertura de "
    "sus provisiones al supervisor del sistema financiero.",
    "El margen de requerimiento de capital se aplica sobre los activos "
    "ponderados por riesgo según la categoría de cada operación.",
    "La supervisión verificará el cumplimiento de los límites de concentración "
    "establecidos para cada deudor y grupo vinculado.",
    "El ajuste de activos se computa descontando las deducciones expresamente "
    "permitidas por la normativa vigente.",
    "Las reservas legales no pueden utilizarse para distribuir dividendos ni "
    "para cubrir pérdidas operativas del ejercicio.",
    "El riesgo de mercado se mide mediante modelos de valor en riesgo "
    "paramétricos con escenarios de estrés previamente definidos.",
    "La gestión de riesgos debe contar con políticas formalizadas, documentadas "
    "y revisadas al menos una vez al año por el directorio.",
    "El índice de liquidez se calcula con los activos líquidos de realización "
    "inmediata y los flujos de efectivo proyectados a treinta días.",
    "Los requisitos de transparencia exigen información clara, comparable y "
    "verificable por parte del usuario final del servicio.",
]


def _texto_legible(tema: str, n_frases: int, semilla: int) -> str:
    generador = random.Random(semilla)
    partes = []
    palabras = 0
    while palabras < n_frases:
        oracion = generador.choice(ORACIONES)
        partes.append(oracion)
        palabras += len(oracion.split())
    partes.append(
        f"Este documento corresponde al ambito normativo de: {tema}. "
        f"Contenido generado para pruebas de carga deterministas. Referencia {semilla}."
    )
    return " ".join(partes)


def _html(titulo: str, organismo: str, tipo: str, cuerpo: str) -> str:
    parrafos = "".join(
        f"<p>{bloque}</p>"
        for bloque in cuerpo.split(". ")
        if bloque.strip()
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"><title>{titulo}</title></head>
<body>
  <nav><a href="/">Inicio</a> <a href="/doc/1">Documento</a></nav>
  <header><h1>{titulo}</h1><p>{organismo} - {tipo}</p></header>
  <article>{parrafos}</article>
  <footer><p>Pie de pagina sin valor normativo.</p></footer>
  <script>console.log("ruido que debe eliminarse");</script>
  <style>body{{color:#333}}</style>
</body>
</html>"""


@app.get("/salud")
async def salud():
    return {"status": "ok", "documentos": len(TEMAS)}


@app.get("/doc/{n}", response_class=HTMLResponse)
async def documento(n: int, latencia: float = LATENCIA_SEGUNDOS):
    if not 1 <= n <= len(TEMAS):
        return Response(status_code=404, content="Documento inexistente")
    titulo, organismo, tipo = TEMAS[n - 1]
    cuerpo = _texto_legible(titulo, TAMANO_PALABRAS, semilla=n)
    await asyncio.sleep(latencia)
    return HTMLResponse(_html(titulo, organismo, tipo, cuerpo))


@app.get("/roto")
async def roto():
    return Response(status_code=404, content="Este recurso no existe")


@app.get("/lento")
async def lento(segundos: float = 40.0):
    await asyncio.sleep(segundos)
    return HTMLResponse(_html("Lento", "ASFI", "Circular", "documento lento"))


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="Servidor de fixtures para el benchmark")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()