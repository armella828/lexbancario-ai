"""Clientes compartidos de Supabase y Gemini.

Se centralizan aqui para evitar estado global duplicado entre el pipeline,
los benchmarks y la API: todos consumen la misma instancia en lugar de crear
una por modulo, lo que ademas reduce el numero de conexiones HTTP abiertas.
"""
import os
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

MODEL_EMBEDDING = "models/gemini-embedding-001"
DIMENSIONES_EMBEDDING = 768

MODELOS_GENERACION = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
]

TIMEOUT_MS_EMBEDDING = 60_000
TIMEOUT_MS_GENERACION = 60_000


class ConfigError(RuntimeError):
    """Faltan variables de entorno indispensables."""


def _exigir(nombre: str) -> str:
    valor = os.getenv(nombre)
    if not valor:
        raise ConfigError(f"Falta la variable de entorno {nombre} en el archivo .env")
    return valor


def config_embeddings() -> types.EmbedContentConfig:
    return types.EmbedContentConfig(output_dimensionality=DIMENSIONES_EMBEDDING)


def config_entero(nombre: str, por_defecto: int) -> int:
    """Lee un entero opcional del entorno.

    A diferencia de `_exigir`, aqui la variable es un ajuste de Rendimiento y
    no un requisito: si falta o no es un entero valido se usa el valor por
    defecto, para que un `.env` incompleto no impida arrancar el pipeline.
    """
    valor = os.getenv(nombre)
    if valor is None or valor.strip() == "":
        return por_defecto
    try:
        return int(valor)
    except ValueError:
        return por_defecto


class Entorno:
    """Envolvimiento de los clientes externos con inicializacion perezosa.

    La inicializacion se protege con un lock. Sin el, dos peticiones
    simultaneas que demanden el cliente por primera vez crearian dos
    instancias a la vez, abriendo el doble de conexiones HTTP y
    descartando una. FastAPI ejecuta los endpoints sincronicos en un thread
    pool, de modo que esa carrera es real y no teorica.
    """

    def __init__(self):
        self._supabase = None
        self._ai = None
        self._lock = threading.Lock()

    @property
    def supabase(self):
        if self._supabase is None:
            with self._lock:
                if self._supabase is None:
                    self._supabase = create_client(
                        _exigir("SUPABASE_URL"),
                        _exigir("SUPABASE_SERVICE_ROLE_KEY"),
                    )
        return self._supabase

    @property
    def ai(self):
        if self._ai is None:
            with self._lock:
                if self._ai is None:
                    self._ai = genai.Client(
                        api_key=_exigir("GEMINI_API_KEY"),
                        http_options=types.HttpOptions(timeout=TIMEOUT_MS_GENERACION),
                    )
        return self._ai

    @property
    def url(self) -> str:
        return _exigir("SUPABASE_URL")


# Instancia compartida. Es de solo lectura tras inicializarse y no se
# comparte entre hilos del ProcessPoolExecutor, por lo que no introduce
# condiciones de carrera.
ENTORNO = Entorno()