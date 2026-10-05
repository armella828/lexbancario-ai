"""Comprobacion del limite de cuota contra la API real de Gemini.

`probar_lotes.py` demuestra el backoff con errores simulados, de forma
determinista y sin coste. Esta comprobacion ataca la API de verdad para
demostrar que el limite existe y que el camino controlado lo respeta:

Fase 1. Carga normal con el limitador de cadencia activo, sobre cuota
        sana. Debe completarse sin un solo 429.
Fase 2. Rafaga concurrente sin control de cuota, hasta provocar un 429
        real. Demuestra que el limite es real y no una hipotesis.
Fase 3. Tras esperar a que se vacie la ventana de un minuto, otra carga
        controlada. Debe volver a completarse limpia: el limite de cuota
        no deja residuos.

El conteo de 429 se hace con un proxy que envuelve al cliente real, para no
ensuciar el codigo de produccion con instrumentacion de pruebas.

Consume cuota real del nivel gratuito (unos 170 requests). Ejecutar:
    python -u scripts/probar_limite_real.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.clients import (
    DIMENSIONES_EMBEDDING,
    ENTORNO,
    MODEL_EMBEDDING,
    config_embeddings,
)
from rag.embeddings import LimitadorCadencia, generar_embeddings

TEXTO = "Reserva minima de capital y requisitos de solvencia."
HILOS = 10
POR_HILO = 15
ESPERA_VENTANA = 65   # algo mas que la ventana de un minuto de la cuota

fallos = []
ok = 0


def comprobar(descripcion, condicion, detalle=""):
    global ok
    if condicion:
        ok += 1
        print(f"  [OK] {descripcion}", flush=True)
    else:
        fallos.append(descripcion)
        print(f"  [X]  {descripcion} {detalle}", flush=True)


class ContadorCuota:
    """Proxy del cliente real que cuenta aciertos y 429."""

    def __init__(self, real):
        self.real = real
        self.models = self
        self.exitos = 0
        self.cuota_agotada = 0

    def embed_content(self, model, contents, config):
        try:
            respuesta = self.real.models.embed_content(
                model=model, contents=contents, config=config
            )
            self.exitos += 1
            return respuesta
        except Exception as exc:
            mensaje = str(exc)
            if "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
                self.cuota_agotada += 1
            raise


def cuota_diaria_agotada(exc):
    """Distingue 'quota diaria agotada' de un fallo real del codigo.

    La cuota gratuita son 1000 requests de embeddings al dia. Si esta
    agotada, la prueba no puede medir nada y no debe contar como fallo del
    pipeline: lo honesto es saltarla y decirlo.
    """
    mensaje = str(exc)
    return (
        "horas" in mensaje
        or "DIARIA" in mensaje
        or "EmbedContentRequestsPerDay" in mensaje
    )


def carga_controlada(n, etiqueta):
    """Carga con el limitador de cadencia de produccion."""
    contador = ContadorCuota(ENTORNO.ai)
    t0 = time.perf_counter()
    try:
        vectores = generar_embeddings(
            [f"{TEXTO} {etiqueta}-{i}" for i in range(n)],
            tamano_lote=1,
            cliente_ai=contador,
            limitador=LimitadorCadencia(),
        )
    except Exception as exc:
        if cuota_diaria_agotada(exc):
            print(f"  [--] [{etiqueta}] OMITIDA: cuota diaria de embeddings "
                  f"agotada.\n       {exc}", flush=True)
            return False
        comprobar(f"[{etiqueta}] {n} textos vectorizados", False,
                  f"-> {type(exc).__name__}: {exc}")
        return False
    elapsed = time.perf_counter() - t0
    comprobar(f"[{etiqueta}] {n} textos vectorizados", len(vectores) == n, f"-> {len(vectores)}")
    comprobar(f"[{etiqueta}] Todos los vectores con {DIMENSIONES_EMBEDDING} dimensiones",
              all(len(v) == DIMENSIONES_EMBEDDING for v in vectores))
    comprobar(f"[{etiqueta}] Sin 429 con la cadencia activa", contador.cuota_agotada == 0,
              f"-> {contador.cuota_agotada}")
    comprobar(f"[{etiqueta}] Ningun request perdido ({contador.exitos}/{n})",
              contador.exitos == n, f"-> {contador.exitos}/{n}")
    minimo = (n - 1) * (60.0 / 80)
    comprobar(f"[{etiqueta}] Respeta el intervalo minimo entre requests",
              elapsed >= minimo * 0.9, f"-> {elapsed:.1f}s frente a {minimo:.1f}s")
    print(f"      ({n} requests en {elapsed:.1f}s, "
          f"{contador.cuota_agotada} 429, espera por cadencia {elapsed - minimo:.1f}s)",
          flush=True)
    return True


print("=" * 74, flush=True)
print("LIMITE DE CUOTA CONTRA LA API REAL", flush=True)
print("=" * 74, flush=True)

# ----------------------------------------------- fase 1: con control
print("\n[1] Carga normal con el limitador de cadencia activo", flush=True)
carga1 = carga_controlada(12, "controlada")

# ------------------------------------------- fase 2: rafaga sin control
print("\n[2] Rafaga concurrente sin control de cuota", flush=True)
# Un bucle secuencial NO serviria: cada request tarda lo suyo en volver, y
# esa latencia ya mantiene el ritmo por debajo de 100 r/min. Para exceder el
# limite hace falta concurrencia. Aqui no hay reintentos: se quiere observar
# el 429, no sobrevivirlo.
contador = ContadorCuota(ENTORNO.ai)
otros_errores = []


def disparo(indice):
    try:
        contador.embed_content(
            model=MODEL_EMBEDDING,
            contents=[f"{TEXTO} rafaga-{indice}"],
            config=config_embeddings(),
        )
    except Exception as exc:
        if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
            otros_errores.append(f"{type(exc).__name__}: {exc}")
    return indice


t0 = time.perf_counter()
enviados = 0
with ThreadPoolExecutor(max_workers=HILOS) as pool:
    futuros = [pool.submit(disparo, i) for i in range(HILOS * POR_HILO)]
    for f in as_completed(futuros):
        enviados += 1
        if contador.cuota_agotada:
            break
    pool.shutdown(wait=False, cancel_futures=True)
elapsed = time.perf_counter() - t0

comprobar("La rafaga solo produce 429, sin otros errores",
          not otros_errores, f"-> {otros_errores[:2]}")
comprobar("Se provoca al menos un 429 real", contador.cuota_agotada > 0,
          f"-> sin 429 tras {enviados} requests en {elapsed:.1f}s; el limite real "
          f"puede ser mayor que 100 r/min para esta clave")
if contador.cuota_agotada:
    print(f"      ({contador.cuota_agotada} 429 con {enviados} requests en "
          f"{elapsed:.1f}s usando {HILOS} en paralelo; {contador.exitos} con exito)",
          flush=True)

# ------------------------------------------ fase 3: recuperacion
print(f"\n[3] Recuperacion tras la rafaga (esperando {ESPERA_VENTANA}s a que "
      f"se vacie la ventana)", flush=True)
time.sleep(ESPERA_VENTANA)
carga3 = carga_controlada(3, "tras-rafaga")

print("\n" + "=" * 74, flush=True)
omitidas = [n for n, mal in ((1, not carga1), (3, not carga3)) if mal]
if fallos:
    print(f"[X] FAIL: {len(fallos)} de {ok + len(fallos)} comprobaciones fallidas", flush=True)
    for f in fallos:
        print(f"    - {f}", flush=True)
    sys.exit(1)
if omitidas:
    print(f"[OK] PASS: {ok} comprobaciones. Fases {omitidas} omitidas por "
          f"cuota diaria agotada.", flush=True)
    sys.exit(0)
print(f"[OK] PASS: {ok} comprobaciones del limite real de cuota.", flush=True)
sys.exit(0)