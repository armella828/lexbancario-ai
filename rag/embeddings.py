"""Generacion de embeddings por lotes con control de cuota (Requisito 3).

La API de Google AI Studio en el nivel gratuito impone un limite de
requests por minuto (100 r/min para `embed_content` en el free tier). El
limite cuenta REQUESTES, no textos: enviar 100 textos en 100 llamadas agota
la cuota igual que enviar 1000 textos en 100 llamadas, pero el segundo caso
hace 10 veces mas trabajo por request consumido. Por eso el batching no es
solo una optimizacion de latencia, es unsustainable la ingesta.

Se combinan tres mecanismos:

1. Batching: se agrupan los fragmentos en lotes de N textos por llamada.
2. Limitador proactivo de cadencia: se espacia el envio de lotes para no
   superar el limite de requests por minuto, evitando el 429 en lugar de
   esperar a que ocurra.
3. Backoff exponencial con jitter: si aun asi llega un 429 o un 503, se
   reintenta esperando el `retryDelay` que indica la propia API.
"""
import random
import re
import threading
import time
from typing import List, Optional

from .clients import ENTORNO, MODEL_EMBEDDING, config_embeddings

TAMANO_LOTE_DEFAULT = 10
MAX_INTENTOS = 6
BACKOFF_BASE_SEGUNDOS = 2.0
BACKOFF_MAX_SEGUNDOS = 90.0

# Cadencia por defecto del nivel gratuito (100 requests/min). Se deja
# margen de seguridad del 20% frente al limite duro.
REQUESTS_POR_MINUTO = 100
MARGEN_SEGURIDAD = 0.8


class CuotaAgotada(RuntimeError):
    """Se agotaron los reintentos tras exhausting la cuota de la API."""


class LimitadorCadencia:
    """Espacia las llamadas para respetar un maximo de requests por minuto.

    No es un simple `sleep` entre lotes: guarda la marca del ultimo envio y
    espera solo el tiempo restante, de modo que el costo de la espera es
    exactamente el necesario y no un redondeo por lote.
    """

    def __init__(self, requests_por_minuto: int = REQUESTS_POR_MINUTO):
        self._minimo_entre = 60.0 / max(1, requests_por_minuto)
        self._ultimo = 0.0
        self._lock = threading.Lock()
        self.esperas = 0.0

    def esperar_turno(self):
        with self._lock:
            ahora = time.monotonic()
            objetivo = self._ultimo + self._minimo_entre
            if ahora < objetivo:
                espera = objetivo - ahora
                self.esperas += espera
                time.sleep(espera)
                ahora = time.monotonic()
            self._ultimo = ahora


def _extraer_retry_delay(mensaje: str, por_defecto: float) -> float:
    """Lee la espera que Gemini indica en el cuerpo del error 429.

    La API reporta el tiempo en varias formas textuales segun el cliente:
    "Please retry in 27.69s", "retryDelay: 27s" o el JSON anidado
    `retryDelay: '27s'`. Se cubren todas y si no aparece se usa el
    backoff propio.
    """
    patrones = (
        r"retry\s+in\s+([\d.]+)\s*s",
        r"retry\s+after\s+([\d.]+)\s*s",
        r"retry_?delay\D{0,25}?([\d.]+)\s*s",
    )
    for patron in patrones:
        m = re.search(patron, mensaje, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return por_defecto


def _es_transitorio(mensaje: str) -> bool:
    marcas = (
        "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
        "500", "502", "504", "quota", "rate", "overloaded",
    )
    return any(m in mensaje for m in marcas)


def embedir_lote(textos: List[str], limitador: Optional[LimitadorCadencia] = None):
    """Genera embeddings para un lote, con reintentos y backoff."""
    limitador = limitador or LimitadorCadencia()
    espera = BACKOFF_BASE_SEGUNDOS
    ultimo_error = None

    for intento in range(1, MAX_INTENTOS + 1):
        try:
            limitador.esperar_turno()
            respuesta = ENTORNO.ai.models.embed_content(
                model=MODEL_EMBEDDING,
                contents=textos,
                config=config_embeddings(),
            )
            return [e.values for e in respuesta.embeddings]
        except Exception as exc:
            ultimo_error = exc
            mensaje = str(exc)
            if not _es_transitorio(mensaje):
                raise
            if intento == MAX_INTENTOS:
                break

            if "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
                pausa = _extraer_retry_delay(mensaje, espera)
                espera = min(BACKOFF_MAX_SEGUNDOS, max(espera * 2, pausa))
            else:
                # Jitter del 25% para que varios workers no compitan por el
                # mismo hueco de cuota con reintentos sincronizados.
                pausa = min(BACKOFF_MAX_SEGUNDOS, espera)
                espera = min(BACKOFF_MAX_SEGUNDOS, espera * 2)

            pausa *= 1.0 + random.uniform(0.0, 0.25)
            print(
                f"    [!] {type(exc).__name__} transitorio "
                f"(intento {intento}/{MAX_INTENTOS}); pausa {pausa:.1f}s"
            )
            time.sleep(pausa)

    raise CuotaAgotada(
        f"No se pudieron generar embeddings tras {MAX_INTENTOS} intentos. "
        f"Ultimo error: {ultimo_error}"
    )


def lotes(textos: List[str], tamano: int = TAMANO_LOTE_DEFAULT):
    """Divide la lista en lotes de `tamano` elementos."""
    for i in range(0, len(textos), tamano):
        yield textos[i:i + tamano]


def generar_embeddings(
    textos: List[str],
    tamano_lote: int = TAMANO_LOTE_DEFAULT,
    limitador: Optional[LimitadorCadencia] = None,
    al_progresar=None,
) -> List[list]:
    """Genera embeddings de `textos` por lotes, con limite de cuota global.

    Se devuelve un solo limitador para toda la llamada: asi la cadencia se
    aplica sobre el total de requests, no por lote.
    """
    limitador = limitador or LimitadorCadencia()
    vectores: List[list] = []

    for indice, lote in enumerate(lotes(textos, tamano_lote), start=1):
        if al_progresar:
            al_progresar(indice, lote)
        vectores.extend(embedir_lote(lote, limitador))

    return vectores