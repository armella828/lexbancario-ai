"""Graficos del benchmark, a partir de resultados/benchmark.json.

Por que el grafico incluye el techo de Amdahl y no solo el speedup real:
Al poner la curva de Amdahl al lado del speedup medido se ve de inmediato
que el paralelismo no escala de forma lineal y desde donde empieza a
perder eficiencia. Si se dibuja solo el speedup real, el lector supone que
podria haber seguido creciendo hasta los 24 nucleos.

Se separan en dos archivos:
  - speedup_vs_amdahl.png : Serie A, escalado + Amdahl
  - fases_serie_a.png     : donde se va el tiempo por fase
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "resultados", "benchmark.json")
SALIDA = os.path.join(RAIZ, "resultados")


def amdahl(fraccion_serial, p):
    return 1.0 / (fraccion_serial + (1.0 - fraccion_serial) / p)


def cargar():
    with open(ENTRADA, encoding="utf-8") as fh:
        return json.load(fh)


def serie_a_config(data):
    """Devuelve los datos de Serie A, o None si no hay.

    Se comprueba que sea un diccionario y no un entero: una corrida
    fallida deja codigo de retorno en el JSON, y tratar eso como resultado
    daria un error mas adelante, en el grafico, en lugar de aqui.
    """
    valor = data.get("resultados", {}).get("serie_a")
    return valor if isinstance(valor, dict) and "configuraciones" in valor else None


def grafico_speedup(serie_a):
    filas = serie_a["configuraciones"]
    ps = [f["procesos"] for f in filas]
    reales = [f["speedup"] for f in filas]
    frac = serie_a["fraccion_serial_medida"]
    techo = [amdahl(frac, p) for p in ps]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Panel izquierdo: real vs techo, escala log para que quepan los 24.
    ax1.plot(ps, techo, "--", color="#888", linewidth=1.6,
             label="techo de Amdahl (chunking serial)")
    ax1.plot(ps, reales, "o-", color="#1f6fb4", linewidth=2.2,
             markersize=6, label="speedup medido")
    ax1.plot(ps, ps, ":", color="#ccc", linewidth=1,
             label="escalado lineal ideal")

    for p, real in zip(ps, reales):
        if real and p > 1:
            ax1.annotate(f"{real / p:.0%}", (p, real),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=7, color="#555")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ps)
    ax1.set_xticklabels([str(p) for p in ps])
    ax1.set_yscale("log", base=2)
    ax1.set_yticks([1, 2, 4, 8, 16, 32])
    ax1.set_yticklabels(["1x", "2x", "4x", "8x", "16x", "32x"])
    ax1.set_xlabel("procesos (p)")
    ax1.set_ylabel("speedup")
    ax1.set_title("Serie A: speedup medido vs techo de Amdahl\n"
                  "(etiquetas = eficiencia = S_p / p)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_ylim(0.8, max(techo) * 2)

    # Panel derecho: eficiencia = S_p / p, la definicion de la guia.
    # El techo de Amdahl es otra pregunta ("cuanto daria un paralelismo
    # perfecto"); mezclarlo aqui haria parecer mejor al sistema.
    ef = [r / p if p else 0 for r, p in zip(reales, ps)]
    barras = ax2.bar([str(p) for p in ps], ef,
                     color=["#2e8b57" if e > 0.7 else
                            "#c98a2b" if e > 0.45 else "#b04a3f"
                            for e in ef])
    ax2.axhline(0.7, color="#999", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("eficiencia = S_p / p")
    ax2.set_xlabel("procesos (p)")
    ax2.set_title("Eficiencia del paralelismo", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.25)
    for barra, e in zip(barras, ef):
        ax2.text(barra.get_x() + barra.get_width() / 2, e + 0.02,
                 f"{e:.0%}", ha="center", fontsize=8)

    fig.tight_layout()
    ruta = os.path.join(SALIDA, "speedup_vs_amdahl.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_fases(serie_a):
    filas = serie_a["configuraciones"]
    fases = list(filas[-1]["fases"].keys())
    colores = ["#1f6fb4", "#c98a2b", "#2e8b57"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Absolutos: donde se va el tiempo real.
    ancho = 0.8 / len(fases)
    x = range(len(filas))
    for i, fase in enumerate(fases):
        vals = [f["fases"][fase]["media"] for f in filas]
        posiciones = [p + i * ancho - 0.4 + ancho / 2 for p in x]
        ax1.bar(posiciones, vals, width=ancho, label=fase,
                color=colores[i % len(colores)])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"c={f['concurrencia']}\np={f['procesos']}"
                         for f in filas], fontsize=8)
    ax1.set_ylabel("segundos (media)")
    ax1.set_yscale("log")
    ax1.set_title("Tiempo por fase (escala log)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    # Porcentaje: que parte queda serial al final.
    ref = filas[-1]
    total = sum(ref["fases"][f]["media"] for f in fases)
    pcts = [100 * ref["fases"][f]["media"] / total for f in fases]
    ax2.barh(fases, pcts, color=colores[:len(fases)])
    ax2.set_xlabel("% del tiempo total en la config mas rapida")
    ax2.set_title(f"Reparto del tiempo (c={ref['concurrencia']}, "
                  f"p={ref['procesos']})", fontsize=10)
    for i, (f, pc) in enumerate(zip(fases, pcts)):
        ax2.text(pc + 1.5, i, f"{pc:.1f}%", va="center", fontsize=9)
    ax2.set_xlim(0, max(pcts) * 1.35)
    ax2.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    ruta = os.path.join(SALIDA, "fases_serie_a.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def main():
    data = cargar()
    serie_a = serie_a_config(data)
    if not serie_a:
        print("[X] No hay datos de Serie A en benchmark.json.")
        print("    Ejecuta primero:  python scripts/benchmark.py --serie a")
        return 1

    print(f"[*] URLs medidas: {serie_a['urls']}")
    for ruta in (grafico_speedup(serie_a), grafico_fases(serie_a)):
        print(f"[OK] {ruta}")

    # Si Serie B esta, se listan sus fases para el informe.
    serie_b = data.get("resultados", {}).get("serie_b")
    if serie_b and "una_pasada" in serie_b:
        print()
        print("[*] Serie B:")
        for fase in ("descarga", "chunking", "embeddings", "insercion"):
            v = serie_b["una_pasada"].get(fase)
            if isinstance(v, (int, float)):
                print(f"      {fase:<12} {v:>7.2f}s")
    else:
        print()
        print("[i] Serie B sin datos: falta calentar la cache o la cuota.")
    return 0


if __name__ == "__main__":
    sys.exit(main())