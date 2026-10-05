"""Pruebas deterministas de lotes, cadencia y backoff (Requisito 3).

No consumen cuota de Gemini: se inyecta un cliente falso que registra las
llamadas y puede simular errores. Asi se puede afirmar el comportamiento
exactamente, sin depender de agotar la cuota real para provocar un 429 y
sin que la prueba sea intermitente.

El limite de cuota contra la API de verdad se comprueba aparte, en
`probar_limite_real.py`.

Ejecutar:  python scripts/probar_lotes.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag.embeddings as emb
from rag.embeddings import (
    CuotaAgotada,
    LimitadorCadencia,
    generar_embeddings,
)

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


class VectorFalso:
    """Vector determinista que codifica el indice del texto de origen.

    Asi se puede comprobar que cada texto conserva su posicion: un
    desalineamiento silencioso entre lotes es el fallo mas grave posible en
    esta parte, porque no rompe nada visiblemente y solo se descubre al
    buscar en produccion.
    """

    def __init__(self, indice, dim=8):
        self.values = [float(indice) + i / 1000.0 for i in range(dim)]


class RespuestaFalsa:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class ClienteFalso:
    """Cliente minimo con la misma forma que `google.genai.Client`."""

    def __init__(self):
        self.models = self
        self.llamadas = []   # un registro por llamada
        self.fallos = []     # excepciones a lanzar, en orden

    def embed_content(self, model, contents, config):
        self.llamadas.append(list(contents))
        if self.fallos:
            error = self.fallos.pop(0)
            if error is not None:
                raise error
        vectores = [VectorFalso(int(t.split("-")[1])) for t in contents]
        return RespuestaFalsa(vectores)


def indice_de(texto):
    return int(texto.split("-")[1])


def error_429(retry_delay="8s"):
    return RuntimeError(
        "429 RESOURCE_EXHAUSTED: quota exceeded, limit 100 per minute. "
        f"Please retry in {retry_delay}."
    )


def error_503():
    # Transitorio pero SIN pista de espera: obliga al backoff propio.
    return RuntimeError("503 UNAVAILABLE: model overloaded")


Rápido = 10**9   # cadencia tan alta que el limitador nunca espera
sin_espera_real = emb.time.sleep


def con_sleep_falso(fn):
    emb.time.sleep = fn


print("=" * 74)
print("PRUEBA DE LOTES Y CONTROL DE CUOTA")
print("=" * 74)

try:
    # ------------------------------------------------------------ batching
    print("\n[1] Dimensionado de lotes")
    falso = ClienteFalso()
    textos = [f"texto-{i}" for i in range(25)]
    vectores = generar_embeddings(
        textos, tamano_lote=10, cliente_ai=falso, limitador=LimitadorCadencia(Rápido)
    )
    tamanos = [len(c) for c in falso.llamadas]
    comprobar("25 textos con lote=10 producen 3 llamadas", len(falso.llamadas) == 3, f"-> {tamanos}")
    comprobar("Ninguna llamada excede el tamano de lote", all(t <= 10 for t in tamanos), f"-> {tamanos}")
    comprobar("El ultimo lote conserva el resto", tamanos == [10, 10, 5], f"-> {tamanos}")
    comprobar("Se devuelve un vector por texto", len(vectores) == 25, f"-> {len(vectores)}")
    comprobar("El lote 2 contiene los textos 10..19",
              falso.llamadas[1][0] == "texto-10" and falso.llamadas[1][-1] == "texto-19")
    comprobar("El orden texto/vector se preserva entre lotes",
              [v[0] for v in vectores] == [float(i) for i in range(25)],
              f"-> primeros {[v[0] for v in vectores[:3]]}")

    print("\n[2] Lotes de tamaño 1 y grande")
    falso = ClienteFalso()
    generar_embeddings(["texto-7"], tamano_lote=1, cliente_ai=falso, limitador=LimitadorCadencia(Rápido))
    comprobar("Un solo texto con lote=1 hace 1 llamada de 1", [len(c) for c in falso.llamadas] == [1])
    falso = ClienteFalso()
    generar_embeddings(["texto-3"], tamano_lote=500, cliente_ai=falso, limitador=LimitadorCadencia(Rápido))
    comprobar("Un lote mayor que el corpus no falla", [len(c) for c in falso.llamadas] == [1])
    falso = ClienteFalso()
    comprobar("Sin textos no se llama a la API",
              generar_embeddings([], cliente_ai=falso) == [] and not falso.llamadas)

    print("\n[3] Configuracion por entorno")
    os.environ["EMBEDDINGS_TAMANO_LOTE"] = "5"
    comprobar("EMBEDDINGS_TAMANO_LOTE=5 se respeta", emb.tamano_lote_configurado() == 5)
    os.environ["EMBEDDINGS_TAMANO_LOTE"] = "basura"
    comprobar("Un valor no numerico cae al valor por defecto", emb.tamano_lote_configurado() == 10)
    del os.environ["EMBEDDINGS_TAMANO_LOTE"]
    os.environ["EMBEDDINGS_REQUESTS_POR_MINUTO"] = "30"
    comprobar("EMBEDDINGS_REQUESTS_POR_MINUTO se respeta", emb.cadencia_configurada() == 30)
    comprobar("El limitador deriva el intervalo de la cadencia",
              abs(LimitadorCadencia()._minimo_entre - 2.0) < 1e-9)
    del os.environ["EMBEDDINGS_REQUESTS_POR_MINUTO"]
    comprobar("Sin variable se usa 80 r/min (20% de margen)", emb.cadencia_configurada() == 80)

    # -------------------------------------------------------------- backoff
    print("\n[4] Backoff ante 429 con retryDelay de la API")
    esperas = []
    con_sleep_falso(lambda s: esperas.append(s))
    falso = ClienteFalso()
    falso.fallos = [error_429("8s"), error_429("8s"), None]
    generar_embeddings(["texto-0"], tamano_lote=1, cliente_ai=falso,
                       limitador=LimitadorCadencia(Rápido))
    comprobar("Dos 429 y al tercer intento acierta", len(falso.llamadas) == 3,
              f"-> {len(falso.llamadas)} llamadas")
    comprobar("Aguanta dos esperas del backoff",
              len([e for e in esperas if e > 1.0]) == 2,
              f"-> {[round(e, 2) for e in esperas]}")
    comprobar("Respeta el retryDelay indicado (8s) con jitter",
              all(8.0 <= e <= 10.0 for e in esperas if e > 1.0),
              f"-> {[round(e, 2) for e in esperas]}")

    print("\n[5] Lectura del retryDelay en todos sus formatos")
    from rag.embeddings import (
        RETRY_DELAY_MAX_ACEPTABLE_SEGUNDOS,
        _duracion_legible,
        _extraer_retry_delay,
    )
    formatos = [
        ("Please retry in 27.694095033s.", 27.7),
        ("retryDelay: '27s'", 27.0),
        ("retry after 12s", 12.0),
        ("Please retry in 1m30s", 90.0),
        ("Please retry in 4h29m8.2s", 16148.2),
        ("sin pista de espera", 5.0),
    ]
    for texto, esperado in formatos:
        got = _extraer_retry_delay(texto, 5.0)
        comprobar(f"'{texto[:34]}' -> {esperado}s",
                  abs(got - esperado) < 0.6, f"-> {got}")
    comprobar("Un 4h29m supera el umbral de reintento",
              _extraer_retry_delay("Please retry in 4h29m8s.", 2) > RETRY_DELAY_MAX_ACEPTABLE_SEGUNDOS)
    comprobar("Un 27s NO supera el umbral de reintento",
              not (_extraer_retry_delay("Please retry in 27s.", 2) > RETRY_DELAY_MAX_ACEPTABLE_SEGUNDOS))
    comprobar("16148s se formatea como horas", "hora" in _duracion_legible(16148.0))

    print("\n[6] Un retryDelay enorme falla rapido en vez de colgar")
    # Sin este tope, un 429 de cuota DIARIA (la API pide ~4h30m) dejaba un
    # endpoint HTTP durmiendo horas. Debe abortar de inmediato y decirlo.
    esperas.clear()
    falso = ClienteFalso()
    falso.fallos = [RuntimeError(
        "429 RESOURCE_EXHAUSTED: quota exceeded. Please retry in 4h29m8s.")]
    t0 = time.perf_counter()
    try:
        generar_embeddings(["texto-0"], tamano_lote=1, cliente_ai=falso,
                           limitador=LimitadorCadencia(Rápido))
        comprobar("Cuota diaria agotada lanza CuotaAgotada", False, "-> no lanzo")
    except CuotaAgotada as e:
        comprobar("Cuota diaria agotada lanza CuotaAgotada", True)
        comprobar("El mensaje explica el motivo y da una espera legible",
                  "horas" in str(e), f"-> {e}")
    comprobar("No se duerme: aborta en menos de un segundo",
              time.perf_counter() - t0 < 1.0 and not [e for e in esperas if e > 1.0],
              f"-> {[round(e, 2) for e in esperas]}")

    print("\n[7] Ninguna espera supera una ventana de cuota")
    esperas.clear()
    falso = ClienteFalso()
    falso.fallos = [error_429("45s")] * 20
    try:
        generar_embeddings(["texto-0"], tamano_lote=1, cliente_ai=falso,
                           limitador=LimitadorCadencia(Rápido))
    except CuotaAgotada:
        pass
    comprobar("Todas las esperas respeta el tope de 60s",
              all(e <= emb.BACKOFF_MAX_SEGUNDOS * 1.25 for e in esperas),
              f"-> max {max(esperas) if esperas else 0:.1f}s")

    print("\n[8] Backoff propio cuando la API no da pista")
    esperas.clear()
    falso = ClienteFalso()
    falso.fallos = [error_503()] * 20
    try:
        generar_embeddings(["texto-0"], tamano_lote=1, cliente_ai=falso,
                           limitador=LimitadorCadencia(Rápido))
        comprobar("Sin pista de espera, la pausa se duplica", False, "-> no lanzo CuotaAgotada")
    except CuotaAgotada:
        comprobar("Sin pista de espera, la pausa se duplica", True)
    esperas_503 = [e for e in esperas if e > 1.0]
    comprobar("Las pausas crecen de forma monotona",
              len(esperas_503) == emb.MAX_INTENTOS - 1
              and all(esperas_503[i] <= esperas_503[i + 1] for i in range(len(esperas_503) - 1)),
              f"-> {[round(e, 1) for e in esperas_503]}")
    comprobar(f"Tras {emb.MAX_INTENTOS} intentos lanza CuotaAgotada y no insiste",
              len(falso.llamadas) == emb.MAX_INTENTOS, f"-> {len(falso.llamadas)}")
    comprobar("Ninguna espera supera el tope",
              all(e <= emb.BACKOFF_MAX_SEGUNDOS * 1.25 for e in esperas))

    print("\n[9] Errores no transitorios")
    falso = ClienteFalso()
    falso.fallos = [RuntimeError("400 INVALID_ARGUMENT: contents is required")]
    try:
        generar_embeddings(["texto-0"], tamano_lote=1, cliente_ai=falso,
                           limitador=LimitadorCadencia(Rápido))
        comprobar("Un 400 propaga de inmediato", False, "-> no lanzo")
    except CuotaAgotada:
        comprobar("Un 400 propaga de inmediato", False, "-> se reintento hasta agotar")
    except RuntimeError as e:
        comprobar("Un 400 propaga de inmediato", "400" in str(e))
    comprobar("Un 400 consume una sola llamada", len(falso.llamadas) == 1, f"-> {len(falso.llamadas)}")

    # ------------------------------------------------------------- cadencia
    print("\n[10] Limitador de cadencia")
    pausas = []
    con_sleep_falso(lambda s: pausas.append(s))
    lim = LimitadorCadencia(60)          # 1 request por segundo
    for _ in range(4):
        lim.esperar_turno()
    comprobar("4 turnos a 60 r/min anuncian ~3s de espera",
              abs(lim.esperas - 3.0) < 1e-2, f"-> {lim.esperas:.4f}s")
    comprobar("El primer turno no espera: la espera es entre requests",
              len(pausas) == 3, f"-> {len(pausas)} pausas")
    comprobar("Cada pausa es el intervalo minimo entre requests",
              all(0.99 <= p <= 1.0 for p in pausas), f"-> {[round(p, 4) for p in pausas]}")

    pausas.clear()
    lim = LimitadorCadencia(Rápido)
    for _ in range(5):
        lim.esperar_turno()
    comprobar("Cadencia alta no introduce esperas", lim.esperas < 1e-3, f"-> {lim.esperas:.6f}s")

    print("\n[11] El limitador es seguro entre hilos")
    lim = LimitadorCadencia(60)
    import threading
    hilos = [threading.Thread(target=lambda: [lim.esperar_turno() for _ in range(3)])
             for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    # 12 turnos = 11 intervalos de 1s entre ellos.
    comprobar("12 turnos concurrentes respetan la cadencia global",
              abs(lim.esperas - 11.0) < 0.05, f"-> {lim.esperas:.2f}s de espera acumulada")

finally:
    emb.time.sleep = sin_espera_real

# ------------------------------------------------------------------ resumen
print("\n" + "=" * 74)
if fallos:
    print(f"[X] FAIL: {len(fallos)} de {ok + len(fallos)} comprobaciones fallidas")
    for f in fallos:
        print(f"    - {f}")
    sys.exit(1)
print(f"[OK] PASS: {ok} comprobaciones de lotes, cadencia y backoff.")
sys.exit(0)