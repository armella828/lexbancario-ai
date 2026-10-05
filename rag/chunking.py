"""Particionado de texto (fase CPU-intensiva).

El chunking es trabajo de CPU:manipula cadenas largas sin tocar la red. Por
eso se ejecuta en ProcessPoolExecutor y no en hilos. El GIL serializa la
ejecucion de bytecode en un solo hilo, de modo que un ThreadPoolExecutor
rendiria aqui una aceleracion casi nula; los procesos si aproveban los
nucleos disponibles.

Las funciones que se envian a los procesos deben ser importables por nombre
(modulo + cualificado) para que `pickle` pueda serializarlas. Por eso estan
definidas a nivel de modulo y reciben solo argumentos simples.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Trozo:
    """Fragmento de texto listo para generar su embedding."""

    url: str
    documento: str
    indice: int
    contenido: str


def limpiar_texto(texto: str) -> str:
    return " ".join(texto.split())


def trocear(
    texto: str,
    url: str,
    documento: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Ventana deslizante sobre el texto.

    El solapamiento evita que una frase partida por la frontera de dos
    fragmentos quede incompleta en ambos, lo que degradaria la recuperacion
    de esa frase concreta.
    """
    limpio = limpiar_texto(texto)
    if not limpio:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) debe ser menor que "
            f"chunk_size ({chunk_size}); en caso contrario la ventana "
            "no avanza y el bucle seria infinito."
        )

    trozos = []
    inicio = 0
    largo = len(limpio)
    while inicio < largo:
        fin = min(inicio + chunk_size, largo)
        trozo = limpio[inicio:fin].strip()
        if trozo:
            trozos.append(trozo)
        if fin >= largo:
            break
        inicio += chunk_size - chunk_overlap
    return trozos


def trocear_documento(paquete: dict) -> List[Trozo]:
    """Punto de entrada del proceso hijo (debe ser picklable)."""
    textos = trocear(
        paquete["texto"],
        paquete["url"],
        paquete["documento"],
        paquete["chunk_size"],
        paquete["chunk_overlap"],
    )
    return [
        Trozo(
            url=paquete["url"],
            documento=paquete["documento"],
            indice=idx,
            contenido=texto,
        )
        for idx, texto in enumerate(textos)
    ]


def trocear_lote(
    paquetes: List[dict],
    procesos: Optional[int] = None,
) -> List[Trozo]:
    """Trocea varios documentos en paralelo y devuelve los trozos en orden.

    `executor.map` conserva el orden de entrada, de modo que el fragmento
    mantiene su correspondencia con su documento aunque los procesos
    terminen en desorden.
    """
    paquetes = [p for p in paquetes if p.get("texto")]
    if not paquetes:
        return []

    if procesos is None:
        return [t for p in paquetes for t in trocear_documento(p)]

    if procesos <= 1:
        return [t for p in paquetes for t in trocear_documento(p)]

    with ProcessPoolExecutor(max_workers=procesos) as executor:
        resultados = executor.map(trocear_documento, paquetes)
        return [trozo for grupo in resultados for trozo in grupo]