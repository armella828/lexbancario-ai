"""Generacion de embeddings por lotes con control de cuota (Requisito 3).

La API de Google AI Studio en el nivel gratuito impone un limite de
requests por minuto (100 r/min para `embed_content` en el free tier). El
limite cuenta REQUESTES, no textos: enviar 100 textos en 100 llamadas agota
la cuota igual que enviar 1000 textos en 100 llamadas, pero el segundo caso
hace 10 veces mas trabajo por request consumido. Por eso el batching no es
solo una optimizacion de latencia: es lo que hace sostenible la ingesta.

Se combinan tres mecanismos:

1. Batching: se agrupan los fragmentos en lotes de N textos por llamada.
2. Limitador proactivo de cadencia: se espacia el envio de lotes para no
   superar el limite de requests por minuto, evitando el 429 en lugar de
   esperar a que ocurra.
3. Backoff exponencial con jitter: si aun asi llega un 429 o un 503, se
   reintenta esperando el `retryDelay` que indica la propia API.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional

from .clients import (
    DIMENSIONES_EMBEDDING,
    ENTORNO,
    MODEL_EMBEDDING,
    config_embeddings,
    config_entero,
)

# Cadencia por defecto: 80 requests/min, un 20% por debajo del limite duro
# de 100 r/min del nivel gratuito. El margen absorbe la facturacion de
# requests ya en vuelo cuando empieza la ventana siguiente.
REQUESTS_POR_MINUTO_DEFAULT = 80
TAMANO_LOTE_DEFAULT = 10
MAX_INTENTOS = 6
BACKOFF_BASE_SEGUNDOS = 2.0
# Ninguna espera puede pasar de una ventana completa de cuota: esperar mas
# de 60s no puede ayudar, porque la cuota por minuto se renueva cada minuto.
BACKOFF_MAX_SEGUNDOS = 60.0
# Si el servidor pide mas que esto, reintentar no va a funcionar: se trata de
# una cuota agotada de verdad (por ejemplo la cuota DIARIA de 1000 requests
# del nivel gratuito), no de un pico transitorio de peticiones por minuto.
# Ante ese caso se falla de inmediato con un mensaje util en lugar de dejar
# un endpoint HTTP colgado horas.
RETRY_DELAY_MAX_ACEPTABLE_SEGUNDOS = 120.0


def tamano_lote_configurado() -> int:
    return max(1, config_entero("EMBEDDINGS_TAMANO_LOTE", TAMANO_LOTE_DEFAULT))


def cadencia_configurada() -> int:
    return max(1, config_entero("EMBEDDINGS_REQUESTS_POR_MINUTO", REQUESTS_POR_MINUTO_DEFAULT))


class CuotaAgotada(RuntimeError):
    """Se agotaron los reintentos tras exhausting la cuota de la API."""


class LimitadorCadencia:
    """Espacia las llamadas para respetar un maximo de requests por minuto.

    No es un simple `sleep` entre lotes: guarda la marca del ultimo envio y
    espera solo el tiempo restante, de modo que el costo de la espera es
    exactamente el necesario y no un redondeo por lote.
    """

    def __init__(self, requests_por_minuto: int = None):
        rpm = requests_por_minuto or cadencia_configurada()
        self._minimo_entre = 60.0 / max(1, rpm)
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


def _duracion_legible(segundos: float) -> str:
    """Formatea una espera como '4h 29 min' o '27 s', para mensajes de error."""
    if segundos >= 3600:
        return f"{segundos / 3600:.1f} horas"
    if segundos >= 60:
        return f"{segundos / 60:.0f} minutos"
    return f"{segundos:.0f} s"


def _duracion_a_segundos(texto: str) -> float:
    """Convierte una duracion de Google a segundos.

    La API escribe el retraso en varios formatos segun el cliente: "27s",
    "4h29m8s" o "1m30s". Sin esto, un "4h29m" se leeria como 4 segundos o
    como un numero gigante, y en ambos casos la espera seria incorrecta.
    """
    total = 0.0
    for valor, unidad in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", texto):
        factor = {"h": 3600.0, "m": 60.0, "s": 1.0}[unidad]
        total += float(valor) * factor
    return total


def _extraer_retry_delay(mensaje: str, por_defecto: float) -> float:
    """Lee la espera que Gemini indica en el cuerpo del error 429.

    Se cubren los formatos textuales ("Please retry in 4h29m8s",
    "retry after 12s") y el campo JSON anidado `retryDelay`. Si no aparece
    ninguno se devuelve el backoff propio.
    """
    patrones = (
        # "Please retry in 4h29m8s" / "retry after 12s". Se captura todo el
        # token porque las duraciones pueden componerse (4h29m8s).
        r"retry\s+in\s+(\d[\d.]*(?:[hms][\d.]*)*[hms]?)",
        r"retry\s+after\s+(\d[\d.]*(?:[hms][\d.]*)*[hms]?)",
        # JSON: 'retryDelay': '27s' -> aqui la unidad va dentro del grupo
        r"retry_?delay\D{0,25}?(\d+(?:\.\d+)?\s*[hms])",
    )
    for patron in patrones:
        m = re.search(patron, mensaje, re.IGNORECASE)
        if m:
            valor = _duracion_a_segundos(m.group(1))
            if valor > 0:
                return valor
    return por_defecto


def _es_transitorio(mensaje: str) -> bool:
    marcas = (
        "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
        "500", "502", "504", "quota", "rate", "overloaded",
    )
    return any(m in mensaje for m in marcas)


def embedir_lote(
    textos: List[str],
    limitador: Optional[LimitadorCadencia] = None,
    cliente_ai=None,
):
    """Genera embeddings para un lote, con reintentos y backoff.

    `cliente_ai` permite inyectar un doble en las pruebas para ejercitar el
    backoff de forma determinista, sin depender de agotar la cuota real.
    """
    limitador = limitador or LimitadorCadencia()
    cliente = cliente_ai if cliente_ai is not None else ENTORNO.ai
    espera = BACKOFF_BASE_SEGUNDOS
    ultimo_error = None

    for intento in range(1, MAX_INTENTOS + 1):
        try:
            limitador.esperar_turno()
            respuesta = cliente.models.embed_content(
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
                pedido = _extraer_retry_delay(mensaje, espera)
                if pedido > RETRY_DELAY_MAX_ACEPTABLE_SEGUNDOS:
                    horas = pedido / 3600.0
                    raise CuotaAgotada(
                        f"La API pide esperar {horas:.1f} h antes de reintentar, "
                        f"asi que no es un pico transitorio de peticiones por "
                        f"minuto sino cuota agotada (con frecuencia la cuota "
                        f"DIARIA de embeddings del nivel gratuito). No se "
                        f"reintenta: espera {_duracion_legible(pedido)} y "
                        f"reintenta la ingesta."
                    )
                pausa = min(BACKOFF_MAX_SEGUNDOS, pedido)
                espera = min(BACKOFF_MAX_SEGUNDOS, max(espera * 2, pausa))
            else:
                # Jitter del 25% para que varios workers no compitan por el
                # mismo hueco de cuota con reintentos sincronizados.
                pausa = min(BACKOFF_MAX_SEGUNDOS, espera)
                espera = min(BACKOFF_MAX_SEGUNDOS, espera * 2)

            pausa = min(BACKOFF_MAX_SEGUNDOS, pausa * (1.0 + random.uniform(0.0, 0.25)))
            print(
                f"    [!] {type(exc).__name__} transitorio "
                f"(intento {intento}/{MAX_INTENTOS}); pausa {pausa:.1f}s"
            )
            time.sleep(pausa)

    raise CuotaAgotada(
        f"No se pudieron generar embeddings tras {MAX_INTENTOS} intentos. "
        f"Ultimo error: {ultimo_error}"
    )


class CacheEmbeddings:
    """Embeddings guardados en disco, indexados por hash del contenido.

    Existe por una razon concreta: el nivel gratuito de Google AI Studio
    permite 1000 requests de embeddings al dia. Un benchmark que repita el
    mismo corpus en varias replicas agotaria la cuota en la primera
    replica y las siguientes medirian backoff, no concurrencia.

    Con la cache, solo la primera pasada paga requests; las demas leen de
    disco. El vector guardado es siempre el mismo para el mismo texto,
    porque el modelo es determinista, de modo que repetir la medicion no
    cambia el resultado.

    La clave incluye el modelo y la dimension: si alguno cambia, la clave
    cambia y el vector se recalcula, en vez de servir un vector obsoleto
    de otra configuracion.
    """

    VERSION = 1

    def __init__(
        self,
        ruta: Optional[str] = None,
        modelo: str = MODEL_EMBEDDING,
        dimension: int = DIMENSIONES_EMBEDDING,
    ):
        self.ruta = ruta or os.getenv(
            "EMBEDDINGS_CACHE", "cache_embeddings/embeddings.json"
        )
        self.modelo = modelo
        self.dimension = dimension
        self._vectores: Dict[str, list] = {}
        self._lock = threading.Lock()
        self.aciertos = 0
        self.fallos = 0
        self.escrituras = 0
        self.peticiones_evitadas = 0

    def _clave(self, texto: str) -> str:
        crudo = f"{self.modelo}|{self.dimension}|{texto}".encode("utf-8")
        return hashlib.sha256(crudo).hexdigest()

    def cargar(self):
        """Lee la cache de disco. Una cache corrupta se ignora, no rompe."""
        if not os.path.exists(self.ruta):
            return self
        try:
            with open(self.ruta, encoding="utf-8") as f:
                datos = json.load(f)
        except (json.JSONDecodeError, OSError):
            return self
        if (
            datos.get("version") != self.VERSION
            or datos.get("modelo") != self.modelo
            or datos.get("dimension") != self.dimension
        ):
            # Configuracion distinta: no sirven estos vectores.
            return self
        self._vectores = datos.get("vectores", {})
        return self

    def guardar(self):
        os.makedirs(os.path.dirname(self.ruta) or ".", exist_ok=True)
        temporal = f"{self.ruta}.tmp"
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump({
                "version": self.VERSION,
                "modelo": self.modelo,
                "dimension": self.dimension,
                "vectores": self._vectores,
            }, f)
        # Sustitucion atomica: si el proceso muere a mitad, la cache
        # anterior sigue intacta en lugar de quedar truncada.
        os.replace(temporal, self.ruta)
        self.escrituras += 1
        return self

    def obtener(self, texto: str) -> Optional[list]:
        clave = self._clave(texto)
        vector = self._vectores.get(clave)
        if vector is None:
            self.fallos += 1
        else:
            self.aciertos += 1
        return vector

    def almacenar(self, texto: str, vector: list):
        with self._lock:
            self._vectores[self._clave(texto)] = list(vector)

    def faltan(self, textos: List[str]) -> List[str]:
        """Devuelve los textos sin vector, en orden y sin repetir.

        Deduplicar importa: si el mismo texto aparece en varios lotes,
        pedirlo una vez por lote multiplicaria el consumo de cuota sin
        aportar nada.
        """
        faltan = []
        vistos = set()
        for texto in textos:
            clave = self._clave(texto)
            if clave in self._vectores or clave in vistos:
                continue
            vistos.add(clave)
            faltan.append(texto)
        return faltan

    def resumen(self) -> dict:
        return {
            "aciertos": self.aciertos,
            "fallos": self.fallos,
            "vectores": len(self._vectores),
            "peticiones_evitadas": self.peticiones_evitadas,
        }


def lotes(textos: List[str], tamano: int = TAMANO_LOTE_DEFAULT):
    """Divide la lista en lotes de `tamano` elementos."""
    for i in range(0, len(textos), tamano):
        yield textos[i:i + tamano]


def generar_embeddings(
    textos: List[str],
    tamano_lote: int = None,
    limitador: Optional[LimitadorCadencia] = None,
    al_progresar=None,
    cliente_ai=None,
    cache: Optional["CacheEmbeddings"] = None,
) -> List[list]:
    """Genera embeddings de `textos` por lotes, con limite de cuota global.

    Se devuelve un solo limitador para toda la llamada: asi la cadencia se
    aplica sobre el total de requests, no por lote.

    Si se pasa `cache`, solo se piden a la API los textos que no estaban
    guardados. El orden de la salida sigue siendo el de `textos`.
    """
    limitador = limitador or LimitadorCadencia()
    tamano_lote = tamano_lote or tamano_lote_configurado()

    if cache is None:
        vectores: List[list] = []
        for indice, lote in enumerate(lotes(textos, tamano_lote), start=1):
            if al_progresar:
                al_progresar(indice, lote)
            vectores.extend(embedir_lote(lote, limitador, cliente_ai))
        return vectores

    faltantes = cache.faltan(textos)
    nuevos: Dict[str, list] = {}
    for indice, lote in enumerate(lotes(faltantes, tamano_lote), start=1):
        if al_progresar:
            al_progresar(indice, lote)
        for texto, vector in zip(lote, embedir_lote(lote, limitador, cliente_ai)):
            nuevos[texto] = vector
            cache.almacenar(texto, vector)

    resueltos = []
    for texto in textos:
        vector = nuevos.get(texto)
        if vector is None:
            vector = cache.obtener(texto)
        resueltos.append(vector)
    cache.peticiones_evitadas = len(textos) - len(faltantes)
    cache.guardar()
    return resueltos