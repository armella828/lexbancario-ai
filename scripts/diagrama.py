"""Diagrama arquitectonico del flujo concurrente (entregable de la guia).

Se genera con matplotlib y no con una herramienta externa para que el
diagrama quede sujeto al mismo pipeline que los demas artefactos: si
cambia el numero de workers, los lotes o las fases, se regenera con
`python scripts/diagrama.py` y no se queda una imagen desactualizada.

La maquetacion va en bandas horizontales que no se cruzan, de arriba
abajo: API externa, maquina de medicion, y camino de consulta. Antes la
API y la fila de consulta se solapaban; el orden se comprueba mirando
los limites de cada banda.

Salida: resultados/diagrama_flujo.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "resultados")

AZUL = "#1f6fb4"
VERDE = "#2e8b57"
AMBAR = "#c98a2b"
ROJO = "#b04a3f"
GRIS = "#666"
FONDO = "#f4f7fa"

# Bandas verticales (no solaparse):
#   API    y 66-76
#   maquina y 16-58   (pipeline 42-54, cache/base 17-27)
#   consulta y  4-12
ANCHO, ALTO = 116, 79


def caja(ax, x, y, w, h, titulo, lineas, color=AZUL, relleno="white",
         fs_titulo=8.5, fs_texto=7.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=0.02",
        linewidth=1.4, edgecolor=color, facecolor=relleno, zorder=3))
    ax.text(x + w / 2, y + h - 0.65, titulo, ha="center", va="center",
            fontsize=fs_titulo, fontweight="bold", color=color, zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.5, "\n".join(lineas), ha="center",
            va="center", fontsize=fs_texto, color="#333", zorder=4,
            linespacing=1.45)
    return (x, y, w, h)


def flecha(ax, p0, p1, color=GRIS, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, zorder=5,
                                 shrinkA=2, shrinkB=2))


def der(c):
    return (c[0] + c[2], c[1] + c[3] / 2)


def izq(c):
    return (c[0], c[1] + c[3] / 2)


def dibujar():
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    ax.set_xlim(0, ANCHO)
    ax.set_ylim(0, ALTO)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- banda 1: API externa --------------------------------------------
    api = caja(ax, 68.5, 66, 18.5, 10, "Gemini API",
               ["cuota 1000 req/día", "429 RESOURCE_EXHAUSTED"],
               color=AMBAR, relleno="#fff8ec")

    # --- banda 2: maquina de medicion ------------------------------------
    ax.add_patch(FancyBboxPatch(
        (1.5, 16), 113, 42, boxstyle="round,pad=0.4,rounding_size=0.4",
        linewidth=1.6, edgecolor=GRIS, facecolor=FONDO,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(3.6, 55.6, "Máquina de medición · 2 vCPU (límite del contenedor "
            "bench: cpu.max 200000/100000)", fontsize=7.6, color=GRIS,
            style="italic", va="center", zorder=2)

    y, h = 42, 12
    entrada = caja(ax, 3.0, y, 13.5, h, "Corpus",
                   ["10 URLs", "81 KB c/u"], color=GRIS)
    descarga = caja(ax, 20.5, y, 19.5, h, "1 · Descarga",
                    ["asyncio + httpx", "concurrencia = 8 hilos",
                     "I/O-bound"], color=AZUL)
    troceo = caja(ax, 44.0, y, 19.5, h, "2 · Chunking",
                  ["ProcessPoolExecutor", "p = 8 procesos",
                   "CPU-bound · evita la GIL"], color=VERDE)
    emb = caja(ax, 67.5, y, 20.5, h, "3 · Embeddings",
               ["lotes de 10", "cadencia 80 req/min",
                "backoff ante 429"], color=AMBAR)
    inser = caja(ax, 92.0, y, 19.5, h, "4 · Inserción",
                 ["POST en lotes", "de 200 filas",
                  "secuencial"], color=ROJO)

    for a, b in ((entrada, descarga), (descarga, troceo),
                 (troceo, emb), (emb, inser)):
        flecha(ax, der(a), izq(b), color="#555")

    # API encima de la fase 3
    flecha(ax, (77.75, y + h), (77.75, 66), color=AMBAR, lw=1.6)
    ax.text(79.6, 60, "1 request por lote", fontsize=6.9, color=AMBAR,
            rotation=90, ha="left", va="center")

    # --- cache y base, dentro de la maquina -------------------------------
    cache = caja(ax, 67.5, 17.5, 20.5, 8.8, "Caché SHA-256",
                 ["texto → vector en disco",
                  "sin repetir requests ya pagadas"],
                 color=VERDE, relleno="#f0f9f3", fs_titulo=8, fs_texto=7)
    ax.plot([77.75, 77.75], [26.3, y - 0.6], color=VERDE, lw=1.5,
            linestyle=(0, (4, 3)), zorder=2)
    ax.text(79.4, 33.5, "se consulta antes", fontsize=6.9, color=VERDE,
            rotation=90, ha="left", va="center")

    ax.add_patch(FancyBboxPatch(
        (92.0, 17), 19.5, 9.6, boxstyle="round,pad=0.5,rounding_size=0.3",
        linewidth=1.4, edgecolor=ROJO, facecolor="#fdf1ef", zorder=3))
    ax.text(101.75, 24.4, "Supabase + pgvector", ha="center", va="center",
            fontsize=8, fontweight="bold", color=ROJO, zorder=4)
    ax.text(101.75, 20.3, "coleccion_id · B-Tree\n768 dimensiones",
            ha="center", va="center", fontsize=7.2, color="#333",
            zorder=4, linespacing=1.45)
    ax.plot([101.75, 101.75], [26.6, y - 0.6], color=ROJO, lw=1.5,
            linestyle=(0, (4, 3)), zorder=2)

    # --- banda 3: camino de consulta --------------------------------------
    ax.text(3.0, 13.4, "Camino de consulta (aislamiento multi-tenant)",
            fontsize=8.5, fontweight="bold", color="#333")
    yq, hq = 4, 8
    q1 = caja(ax, 3.0, yq, 20.0, hq, "Consulta",
              ["pregunta + coleccion_id"], color=GRIS, fs_titulo=8)
    q2 = caja(ax, 27.5, yq, 30.0, hq, "RPC match_normativa_coleccion",
              ["WHERE coleccion_id = $1", "ORDER BY embedding <=> $2"],
              color=AZUL, fs_titulo=8, fs_texto=7)
    q3 = caja(ax, 62.0, yq, 24.0, hq, "Índice B-Tree",
              ["filtra candidatos", "antes del vector"], color=VERDE,
              fs_titulo=8, fs_texto=7)
    q4 = caja(ax, 90.5, yq, 21.0, hq, "Top-N",
              ["0 filas ajenas", "en 120 consultas"], color=VERDE,
              fs_titulo=8, fs_texto=7)
    for a, b in ((q1, q2), (q2, q3), (q3, q4)):
        flecha(ax, der(a), izq(b), color="#555")

    ax.text(58.0, 0.6, "La inserción (4) se ejecuta en serie y fija el techo "
            "de Amdahl: 45.8 % del tiempo total en la línea base.",
            ha="center", va="bottom", fontsize=7.4, color=ROJO, style="italic")

    fig.tight_layout(pad=0.4)
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, "diagrama_flujo.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    print("[OK]", dibujar())
