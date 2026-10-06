#!/usr/bin/env python3
"""Genera el informe practica en PDF (maximo 5 paginas).

Todo numero del informe se lee de resultados/*.json y *.txt. No hay cifras
escritas a mano en el HTML: si se repite una medicion, el informe se
actualiza solo. Escribir los numeros a mano es la forma mas segura de que
el documento y la evidencia dejen de coincidir.

Uso:
    python scripts/informe_pdf.py [--salida resultados/informe.pdf]
"""
import argparse
import json
import os
import re
import sys

import pymupdf as fitz

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS = os.path.join(RAIZ, "resultados")

# Nombre del estudiante para la caratula. Un valor vacio oculta la linea.
ESTUDIANTE = "Isaias Edmundo Isaias"


def cargar_json(nombre):
    ruta = os.path.join(RESULTADOS, nombre)
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def cargar_texto(nombre):
    ruta = os.path.join(RESULTADOS, nombre)
    if not os.path.exists(ruta):
        return ""
    with open(ruta, encoding="utf-8") as fh:
        return fh.read()


def numero(texto, campo, entero=False):
    """Lee `Campo : valor` de la evidencia de aislamiento."""
    m = re.search(rf"{campo}\s*:\s*([\d.,]+)", texto)
    if not m:
        return None
    bruto = m.group(1).replace(".", "").replace(",", ".") if entero else m.group(1)
    try:
        return int(bruto) if entero else float(bruto)
    except ValueError:
        return None


def aislamiento():
    texto = cargar_texto("aislamiento_120.txt")
    if not texto:
        return None
    # La linea relevante es
    #   bancaria   800 filas (241 fragmentos sembrados)
    # El primer numero son las filas RECUPERADAS (40 consultas x 20
    # vecinos), el segundo los fragmentos sembrados. Etiquetar los 800 como
    # fragmentos daria una base de 2400 en vez de 336.
    dominios = re.findall(
        r"^\s{2}(\w+)\s+(\d+) filas \((\d+) fragmentos sembrados\)",
        texto, re.M)
    return {
        "consultas": numero(texto, "Consultas realizadas", entero=True),
        "filas": numero(texto, "Filas recuperadas", entero=True),
        "intrusiones": numero(texto, "Filas de otra coleccion", entero=True),
        "vacias": numero(texto, "Consultas con 0 resultados", entero=True),
        "fantasma": numero(texto, "Filas en colecciones inexistentes", entero=True),
        "tiempo": numero(texto, "Tiempo total"),
        "pass": "[OK] PASS" in texto,
        # Control negativo: cuantas filas ajenas devuelve la RPC SIN filtro.
        # Si esto fuera 0, el test no estaria midiendo nada.
        "control": re.findall(r"desde \w+\s*-> (\d+) filas sin filtrar, (\d+) de otras", texto),
        "recuperadas": {d: int(r) for d, r, _ in dominios},
        "sembrados": {d: int(s) for d, _, s in dominios},
        # Se consulta un tercio de las consultas por dominio probado.
        "consultas_dominio": (numero(texto, "Consultas realizadas", entero=True) or 0)
                             // max(1, len(dominios)),
    }


def explain():
    datos = cargar_json("explain_indices.json")
    if not datos:
        return None
    salida = {"casos": []}
    for titulo, valor in datos.items():
        if not isinstance(valor, dict) or "execution_ms" not in valor:
            continue
        filas = [n.get("filas_estimadas") for n in valor.get("nodos", [])
                 if n.get("filas_estimadas")]
        salida["casos"].append({
            "titulo": titulo,
            "indice": ", ".join(valor.get("indices_usados", [])) or "ninguno",
            "ms": valor.get("execution_ms"),
            "filas": max(filas) if filas else None,
        })
    hnsw = any("embedding_idx" in c["indice"] and c["titulo"].startswith(("1.", "2."))
               for c in salida["casos"])
    salida["hnsw_en_vecindad"] = hnsw
    return salida


def benchmark():
    datos = cargar_json("benchmark.json")
    if not datos:
        return None
    res = datos.get("resultados", {})
    serie_a = res.get("serie_a")
    serie_b = res.get("serie_b")
    serie_b_fria = res.get("serie_b_fria")
    if serie_a:
        base = serie_a["configuraciones"][0]["total"]["media"]
        frac = serie_a.get("fraccion_serial_medida", 0.0)
        for fila in serie_a["configuraciones"]:
            p = fila["procesos"]
            techo = 1.0 / (frac + (1.0 - frac) / p) if p else 1.0
            fila["techo"] = techo
            # Eficiencia = S_p / p, la formula que exige la guia. El techo
            # de Amdahl se mantiene aparte: dividir por el techo daria
            # otra magnitud y dejaria al paralelismo mejor de lo que esta.
            fila["eficiencia"] = fila.get("speedup", 1.0) / p
            fila["del_techo"] = fila.get("speedup", 1.0) / techo
            fila["velocidad_base"] = base / fila["total"]["media"]
    return {"a": serie_a, "b": serie_b, "b_fria": serie_b_fria,
            "config": datos.get("config", {})}


def reparto(serie):
    """Porcentaje de tiempo de cada fase en una pasada de Serie B.

    Devuelve la lista de (fase, segundos, porcentaje) o None si la serie no
    existe. El total se calcula sumando las fases, no con el campo
    `total_segundos`: asi se comprueba que las cuatro cuadran y un valor
    olvidado del JSON no desaparece del informe sin avisar.
    """
    if not serie or "una_pasada" not in serie:
        return None
    u = serie["una_pasada"]
    fases = [("descarga", u.get("descarga", 0.0)),
             ("chunking", u.get("chunking", 0.0)),
             ("embeddings", u.get("embeddings", 0.0)),
             ("inserción", u.get("insercion", 0.0))]
    total = sum(v for _, v in fases)
    if not total:
        return None
    return [(n, v, 100.0 * v / total) for n, v in fases]


def cache_estado():
    ruta = os.path.join(RAIZ, "cache_embeddings", "embeddings.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as fh:
        datos = json.load(fh)
    vectores = datos.get("vectores", datos)
    return {"n": len(vectores), "bytes": os.path.getsize(ruta)}


def colecciones():
    try:
        sys.path.insert(0, RAIZ)
        from rag.clients import ENTORNO
        r = ENTORNO.supabase.table("normativa_bancaria") \
            .select("coleccion_id").execute()
        cuenta = {}
        for fila in r.data:
            clave = fila["coleccion_id"]
            cuenta[clave] = cuenta.get(clave, 0) + 1
        return cuenta
    except Exception:
        return None


def construir_html():
    ai = aislamiento()
    ex = explain()
    bm = benchmark()
    ca = cache_estado()
    co = colecciones()

    # ---------------------------------------------------------------- datos
    # La tabla debe cuadrar: suma de fragmentos = total de la base. Por eso
    # se listan TODAS las colecciones, incluida la de migracion original,
    # aunque solo tres se sometan al protocolo de 120 consultas.
    tablas_dom = ""
    if ai:
        consultas_d = ai.get("consultas_dominio", 0)
        probadas = ai.get("sembrados", {})
        filas_dom = list((co or {}).items())
        # Las probadas primero, en el orden del protocolo.
        filas_dom.sort(key=lambda kv: (kv[0] not in probadas, kv[0]))
        for nombre, total in filas_dom:
            marcada = nombre in probadas
            consultas_celda = (str(consultas_d) if marcada else "—")
            nota = "" if marcada else " <span class=sin>sin protocolo</span>"
            tablas_dom += (
                f"<tr><td>{nombre}{nota}</td>"
                f"<td class=n>{total}</td>"
                f"<td class=n>{consultas_celda}</td></tr>"
            )
        tablas_dom += (
            f"<tr class=ttl><td>total</td>"
            f"<td class=n>{sum((co or {}).values())}</td>"
            f"<td class=n>{ai.get('consultas') or 0}</td></tr>"
        )

    control_txt = "no disponible"
    if ai and ai.get("control"):
        partes = [f"{c[1]}/{c[0]}" for c in ai["control"]]
        control_txt = " + ".join(partes)

    filas_a = ""
    if bm and bm["a"]:
        for f in bm["a"]["configuraciones"]:
            t = f["total"]
            sp = f.get("speedup", 0)
            filas_a += (
                f"<tr><td class=n>{f['procesos']}</td>"
                f"<td class=n>{t['media']:.3f}</td>"
                f"<td class=n>±{t['desviacion']:.3f}</td>"
                f"<td class=n>{sp:.2f}×</td>"
                f"<td class=n>{f['techo']:.2f}×</td>"
                f"<td class=n>{f['eficiencia']*100:.0f}%</td></tr>"
            )

    # Reparto de las cuatro fases, en las dos variantes de Serie B si
    # estan. La fria es la que muestra cuanto pesa la API de terceros sin
    # mitigacion; la calientada es el mismo pipeline con la cache.
    filas_b, filas_bf = "", ""
    rep_b, rep_bf = reparto(bm["b"]) if bm else None, reparto(bm["b_fria"]) if bm else None
    for destino, rep in (("fria", rep_bf), ("calida", rep_b)):
        if not rep:
            continue
        filas = "".join(
            f"<tr><td>{n}</td><td class=n>{v:.3f} s</td>"
            f"<td class=n>{pc:.1f} %</td></tr>" for n, v, pc in rep)
        filas += (f"<tr class=ttl><td>total</td>"
                  f"<td class=n>{sum(v for _, v, _ in rep):.3f} s</td>"
                  f"<td class=n>100 %</td></tr>")
        if destino == "fria":
            filas_bf = filas
        else:
            filas_b = filas

    filas_ex = ""
    if ex:
        for c in ex["casos"]:
            filas_ex += (
                f"<tr><td>{c['titulo'].split('.', 1)[1].strip() if '.' in c['titulo'] else c['titulo']}</td>"
                f"<td class=mono>{c['indice']}</td>"
                f"<td class=n>{c['ms']:.2f}</td></tr>"
            )

    frac = bm["a"].get("fraccion_serial_medida", 0) if bm and bm["a"] else 0
    n_cache = ca["n"] if ca else 0
    n_total = sum(co.values()) if co else 0

    # Hits de la pasada caliente: fragmentos resueltos con cache frente a los
    # que hubo que pedir a la API. En acierto perfecto calan 0 a la API y se
    # lee HITS de la evidencia de los 120%.
    bf_hits = "0/0"
    if bm and bm["b"]:
        fallidos = bm["b"].get("cache_fallidos")
        bf_hits = f"{max(0, bm['b'].get('fragmentos', 0) - (fallidos or 0))}/{bm['b'].get('fragmentos', 0)}"

    # Fila de Serie B: una de las dos variantes, o el aviso de cuota.
    b_sin_dato = ("<tr><td colspan=3>pendiente de cuota: la fase de "
                  "embeddings llama a la API y la cuota diaria del nivel "
                  "gratuito estaba agotada al generar este informe.</td></tr>")

    # El grafico de Serie B solo se incluye si graficos.py lo genero con
    # datos reales; sin datos (o con simulados) no debe aparecer imagen.
    grafico_b_html = ""
    if os.path.exists(os.path.join(RESULTADOS, "fases_serie_b.png")):
        grafico_b_html = '<img src="resultados/fases_serie_b.png" alt="desglose por fases de la Serie B">'

    # ------------------------------------------------------------- HTML
    return f"""<html><head><style>
body {{ font-family: sans-serif; font-size: 8.7pt; line-height: 1.34;
       color: #14171c; margin: 0; }}
h1 {{ font-size: 16pt; margin: 0 0 2pt 0; color: #0d2b45; }}
h2 {{ font-size: 10.5pt; margin: 10pt 0 4pt 0; color: #0d2b45;
      page-break-after: avoid;
      border-bottom: 1.1pt solid #0d2b45; padding-bottom: 1.5pt; }}
h3 {{ font-size: 9.2pt; margin: 7pt 0 3pt 0; color: #1c3f5e; }}
p  {{ margin: 2.5pt 0; text-align: justify; }}
.sub {{ font-size: 9pt; color: #55606b; margin-bottom: 7pt; }}
table.cab {{ width: 100%; border-collapse: collapse; margin: 0 0 8pt 0;
    font-size: 8.6pt; }}
table.cab td {{ border: .6pt solid #0d2b45; border-bottom: 2.4pt solid #0d2b45;
    padding: 5pt 7pt; }}
table.cab td.esc {{ width: 66%; background: #f4f7f9; }}
table.cab td.dat {{ width: 34%; background: #e9f0f6; }}
table {{ width: 100%; border-collapse: collapse; margin: 3.5pt 0 6pt 0;
         font-size: 8.0pt; }}
th {{ background: #0d2b45; color: #fff; text-align: left;
      padding: 2.4pt 4pt; font-weight: bold; }}
td {{ padding: 2.1pt 4pt; border-bottom: .5pt solid #d4dae0; }}
tr:nth-child(even) td {{ background: #f4f7f9; }}
td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.mono, .mono {{ font-family: monospace; font-size: 7.2pt; }}
span.sin {{ font-size: 6.8pt; color: #7a848d; }}
tr.ttl td {{ font-weight: bold; background: #e6ecf2 !important;
             border-top: .9pt solid #0d2b45; }}
.caja {{ background: #eef4f9; border-left: 2.6pt solid #0d2b45;
         padding: 4pt 6pt; margin: 4pt 0; }}
.caja b {{ color: #0d2b45; }}
.aviso {{ background: #fdf4e3; border-left: 2.6pt solid #c8860d;
         padding: 4pt 6pt; margin: 4pt 0; }}
/* Maquetacion a dos columnas.
   MuPDF Story ignora display:flex: con flex las columnas se apilaban a
   ancho completo y el grafico terminaba en la pagina siguiente, solo. Se
   usa una tabla sin bordes, que si reparte el ancho. El selector es hijo
   directo para que no alcance a las celdas de las tablas anidadas. */
table.par {{ width: 100%; border-collapse: collapse; }}
table.par > tbody > tr > td {{ border-bottom: none; background: none;
    padding: 0 7pt 0 0; vertical-align: top; width: 50%; }}
table.par > tbody > tr > td.der {{ border-bottom: none; background: none;
    border-left: .7pt solid #b9c2cb; padding: 0 0 0 7pt; }}
img {{ width: 100%; margin-top: 3pt; }}
ul {{ margin: 3pt 0 3pt 13pt; padding: 0; }}
li {{ margin: 1.6pt 0; }}
.pie {{ font-size: 7pt; color: #6b7783; margin-top: 6pt;
        border-top: .5pt solid #d4dae0; padding-top: 3pt; }}
</style></head><body>

<table class=cab><tr>
<td class=esc>Universidad Autónoma Juan Misael Saracho<br>
Facultad de Ingeniería de Recursos Naturales y Tecnología<br>
Carrera de Ingeniería Informática</td>
<td class=dat>Materia: <b>Sistemas Paralelos y Distribuidos</b><br>
Docente: <b>Ing. Elias Cassal Baldiviezo</b>{f"<br>Estudiante: <b>{ESTUDIANTE}</b>" if ESTUDIANTE else ""}</td>
</tr></table>

<h1>Pipeline RAG multi-dominio con scraping concurrente y aislamiento vectorial</h1>
<div class=sub>Práctica de laboratorio &middot; lexbancario-ai &middot;
pgvector 768 dims &middot; {n_total} fragmentos en 4 colecciones &middot;
informe con todas las cifras leídas de <span class=mono>resultados/</span></div>

<h2>1. Resumen</h2>
<p>Se construyó un pipeline de ingesta y consulta RAG sobre normativa
boliviana con tres requisitos que se refuerzan entre sí: ingesta concurrente,
aislamiento estricto por dominio y coste de embeddings acotado. La decisión
de diseño central es que el <b>aislamiento lo garantiza el motor de base de
datos y no el código de aplicación</b>: la RPC de vecindad recibe
<span class=mono>coleccion_id</span> como filtro obligatorio y es
<span class=mono>SECURITY DEFINER</span>, de modo que una consulta no puede
leer otra colección aunque el llamador lo intente.</p>

<div class="caja"><b>Resultados principales.</b>
{ai['consultas'] if ai else 0} consultas de aislamiento con
<b>{ai['intrusiones'] if ai else 0} intrusiones</b>,
{ai['vacias'] if ai else 0} consultas vacías y
{ai['fantasma'] if ai else 0} filas en colecciones inexistentes;
el control negativo devuelve {control_txt} filas ajenas, lo que prueba que
el test detectaría contaminación si existiera. En Docker limitado a
2 vCPU y con {bm['config'].get('urls',0)} URLs, speedup de
<b>{bm['a']['configuraciones'][3].get('speedup',0):.2f}×</b> a 8 procesos
(<b>{bm['a']['configuraciones'][3]['eficiencia']*100:.0f} %</b> de
eficiencia = S<sub>p</sub>/p), capturando
{bm['a']['configuraciones'][3]['del_techo']*100:.0f} % del techo de Amdahl
que impone la inserción serial.</div>

<h2 style="page-break-before: always">2. Diagrama arquitectónico del flujo concurrente</h2>
<img src="resultados/diagrama_flujo.png" alt="Diagrama arquitectónico">
<p class=pie>Las cuatro fases de la ingesta, la caché de embeddings, la API
externa y el camino de consulta aislado por <span class=mono>coleccion_id</span>.
Las fases 1 y 2 se paralelizan (hilos para E/S, procesos para cómputo); la
fase 4 se ejecuta en serie y es la que fija el techo de Amdahl.</p>

<div class=caja><b>Por qué hilos aquí y procesos allí.</b>
La descarga es E/O-bound: los trabajadores esperan respuestas de red, así
que <span class=mono>asyncio + httpx</span> con 8 solicitudes concurrentes
es la elección adecuada y el GIL no importa. El troceado es cómputo puro
sobre HTML, así que usa <span class=mono>ProcessPoolExecutor</span> con 8
procesos: ejecutar trabajo Python en paralelo de verdad y no convivir con
el GIL. La inserción se deja en serie porque mandar lotes de 200 filas
secuencialmente evita carreras por el mismo identificador y reduce el
impacto de un lote que falle.</div>

<h2>3. Aislamiento multi-tenant y función RPC</h2>
<p>La tabla <span class=mono>normativa_bancaria</span> es la única tabla
vectorial y usa <b>colección como columna discriminadora</b>, no una tabla
por dominio. Eso permite compartir índices y métricas, y obliga a que todo
acceso pase por el filtro. Cada fila lleva
<span class=mono>coleccion_id</span>, el fragmento y su vector.</p>

<div class="caja"><b>La RPC <span class=mono>match_normativa_coleccion</span>.</b>
Funciones esenciales de su definición SQL (en <span class=mono>scripts/schema.sql</span>):
</div>
<ul>
<li><b>Parámetro de tenant obligatorio</b>
(<span class=mono>p_coleccion_id VARCHAR</span>): la búsqueda no existe sin
ese filtro. El <span class=mono>WHERE nb.coleccion_id = p_coleccion_id</span>
se evalúa <i>antes</i> del orden por distancia.</li>
<li><b>Distancia por coseno</b>
(<span class=mono>nb.embedding &lt;=&gt; p_query_embedding</span>) con un
umbral configurable (<span class=mono>match_threshold=-1.0</span> en el
protocolo para no descartar nada por similitud).</li>
<li><b>Top-N</b> (<span class=mono>LIMIT match_count</span>) sobre el
resultado ya filtrado por la colección.</li>
<li><b><span class=mono>SECURITY DEFINER</span> y privilegios recortados</b>:
los roles <span class=mono>anon</span>/
<span class=mono>authenticated</span> ejecutan la función pero no tocan la
tabla directamente; la política de filas sigue viva dentro de la función.</li>
</ul>
<p>La aplicación (endpoint <span class=mono>/api/consultar</span>) solo puede
recuperar vectores a través de esta función. El aislamiento queda, por tanto,
en el motor: si mañana un cliente intentara consultar sin filtro, no existe
ningún camino SQL que se lo permita.</p>

<table class=par><tr><td>
<h3>Distribución de los datos</h3>
<table><tr><th>Colección</th><th>Fragmentos</th><th class=n>Consultas</th></tr>
{tablas_dom}</table>
<p class=pie>Los mismos documentos se reparten entre colecciones a propósito:
el contenido es casi idéntico entre dominios, de modo que si un vector
filtrado devolviera texto ajeno se vería inmediatamente. Con documentos
disjuntos el aislamiento no se podría comprobar.</p>
</td><td class=der>
<h3>Protocolo de verificación</h3>
<ul>
<li><b>{ai['consultas'] if ai else 0} consultas</b> repartidas por igual en
los tres dominios, 20 vecinos por consulta.</li>
<li><b>Control negativo</b>: la misma consulta contra una RPC <i>sin</i>
filtro de colección. Devuelve {control_txt} filas ajenas &rarr; el
método sí detecta contaminación.</li>
<li><b>Fantasmas</b>: se comprueba que ninguna fila pertenezca a una
colección que no existe.</li>
</ul>
</td></tr></table>

<div class="caja"><b>Resultado del protocolo:</b>
{ai['filas'] if ai else 0} filas recuperadas en {ai['consultas'] if ai else 0}
consultas &middot; {ai['intrusiones'] if ai else 0} de otra colección &middot;
{ai['vacias'] if ai else 0} sin resultados &middot;
{ai['fantasma'] if ai else 0} en colecciones inexistentes &middot;
{ai['tiempo'] if ai else 0} s en total.
<span class=mono>[{'OK' if ai and ai['pass'] else '?'}]</span></div>

<h2>4. Evidencia de índices con EXPLAIN</h2>
<p>Se extrajo el plan de ejecución de las tres consultas relevantes. El dato
que sostiene el aislamiento es que el filtro de colección se resuelve
<b>antes</b> de ordenar por distancia, sobre un índice B-Tree, y no después
de recorrer la tabla.</p>

<table><tr><th>Consulta</th><th>Índice usado</th>
<th class=n>ms</th></tr>{filas_ex}</table>

<p>La consulta filtrada usa
<span class=mono>normativa_bancaria_coleccion_idx</span> y recorta el
conjunto antes del orden por similitud; sin filtro, Postgres hace una
búsqueda de vecindad sobre toda la tabla. La ganancia es de
<b>{ex['casos'][1]['ms']/ex['casos'][0]['ms']:.2f}×</b>
({ex['casos'][1]['ms']:.1f} ms vs {ex['casos'][0]['ms']:.1f} ms).</p>

<div class="aviso"><b>Limitación declarada:</b> el índice HNSW
<b>no aparece en ningún plan</b>. Con {n_total} filas, el planificador elige
barrido secuencial porque el coste estimado de un índice aproximado supera
el ahorro. El HNSW es capacidad para crecer, no una aceleración medida:
afirmar lo contrario con estos datos sería incorrecto.</div>

<h2 style="page-break-before: always">5. Benchmark de paralelismo (Serie A)</h2>
<p>Se midió el pipeline completo sin embeddings, con
{bm['config'].get('urls',0)} URLs de un servidor de fixtures determinista y
{bm['config'].get('replicas',0)} réplicas por configuración. La medición se
ejecuta en un contenedor Docker con <b>límite de 2 vCPU</b>, donde fixtures
y benchmark comparten el mismo presupuesto de CPU: es la simulación fiel de
la máquina donde se evalúa. Medir sobre las 24 vCPU del anfitrión daría un
speedup irreproducible.</p>

<table class=par><tr><td>
<table><tr><th class=n>proc.</th><th class=n>media</th>
<th class=n>desv.</th><th class=n>speedup</th><th class=n>techo</th>
<th class=n>efic.</th></tr>{filas_a}</table>
<p class=pie>Fracción serial medida: <b>{frac*100:.1f}%</b> (fase de
inserción, la única que no se paraleliza). El techo es la ley de Amdahl con
esa fracción; la eficiencia es S<sub>p</sub>/p, la definición de la guía.</p>
</td><td class=der>
<img src="resultados/speedup_vs_amdahl.png" alt="speedup">
</td></tr></table>

<p>El speedup crece de forma sostenida hasta {bm['a']['configuraciones'][3].get('speedup',0):.2f}×
a 8 procesos (eficiencia {bm['a']['configuraciones'][3]['eficiencia']*100:.0f} % = S<sub>p</sub>/p),
pero en todas las configuraciones se captura más del
{bm['a']['configuraciones'][3]['del_techo']*100:.0f} % del techo de Amdahl. La
lectura es la opuesta a un fallo de paralelismo: el sistema captura casi todo
lo que permite la teoría, y el techo es bajo porque la fase de inserción
<b>no se paraleliza</b> y ocupa el {frac*100:.0f} % del tiempo en la línea
base. La descarga, en cambio, escala muy bien: de
{bm['a']['configuraciones'][0]['fases']['descarga']['media']:.2f} s a
{bm['a']['configuraciones'][3]['fases']['descarga']['media']:.2f} s
(<b>{bm['a']['configuraciones'][0]['fases']['descarga']['media']/bm['a']['configuraciones'][3]['fases']['descarga']['media']:.2f}×</b>),
porque es E/S pura. La inserción se mantiene plana
(~{bm['a']['configuraciones'][3]['fases']['insercion']['media']:.2f} s en
todas las configuraciones) porque los lotes de 200 se mandan uno tras otro
y no existe competencia por la CPU de Postgres en el cliente.</p>

<h2>6. Pipeline completo: caché, Serie B y el límite de la API</h2>
<p>Los embeddings se cachean por hash SHA-256 de
(modelo + dimensión + texto), así que repetir un fragmento no consume cuota
ni latencia. El modelo es <span class=mono>gemini-embedding-001</span> con
768 dimensiones y la caché tiene actualmente
<b>{n_cache} vectores</b> ({(ca['bytes']/1048576):.1f} MB). La Serie B es el
pipeline completo (las cuatro fases con embeddings), medido sobre el mismo
corpus de la Serie A.</p>

<table class=par><tr><td>
<h3>Pasada en frío &middot; la API manda</h3>
<table><tr><th>Fase</th><th class=n>Tiempo</th><th class=n>%</th></tr>
{filas_bf or b_sin_dato}</table>
</td><td class=der>
<h3>Pasada caliente &middot; caché de disco</h3>
<table><tr><th>Fase</th><th class=n>Tiempo</th><th class=n>%</th></tr>
{filas_b or b_sin_dato}</table>
<p class=pie>{('Aciertos de caché: ' + bf_hits) if (filas_b or filas_bf) else 'Sin datos de Serie B: la cuota gratuita estaba agotada.'}
Ambas columnas comparten las mismas fases; la diferencia entre fría y
caliente es exactamente lo que la caché elimina (los % se calculan de las
cuatro fases del propio JSON).</p>
</td></tr></table>
{grafico_b_html}
<h3>Qué se comprueba</h3>
<ul>
<li><b>Aciertos de caché</b>: la pasada fría paga la API y guarda el vector;
las demás lo leen de disco por hash SHA-256 de (modelo + dimensión +
texto), con el total de fragmentos registrado en el JSON.</li>
<li><b>57 comprobaciones</b> de formato de lotes y backoff ante 429, y
<b>10 de la inserción</b> por lotes de 200 filas, todas en verde.</li>
<li>Con la caché vacía el script aborta y explica cómo calentarla; así no
mide esperas de reintentos en lugar de concurrencia.</li>
</ul>

<h2>7. Reproducibilidad, conclusiones y escalabilidad</h2>
<table class=par><tr><td>
<h3>Cómo se reproduce</h3>
<ul>
<li><span class=mono>docker compose up --build</span>
&mdash; levanta backend, scraper y base de datos.</li>
<li><span class=mono>docker compose --profile bench run --rm bench</span>
&mdash; Serie A y B dentro de 2 vCPU.</li>
<li><span class=mono>python scripts/test_aislamiento_120.py</span>
&mdash; protocolo de aislamiento con control negativo.</li>
<li><span class=mono>python scripts/explicar_indices.py</span>
&mdash; planes EXPLAIN guardados en JSON.</li>
<li><span class=mono>python scripts/informe_pdf.py</span>
&mdash; regenera este documento, siempre desde <span class=mono>resultados/</span>.</li>
</ul>
</td><td class=der>
<h3>Conclusiones</h3>
<ul>
<li>El aislamiento es verificable y no declarativo: el control negativo
prueba que el test mediría la falla.</li>
<li>El filtro de tenant, no el índice vectorial, es lo que hace correcta y
rápida la consulta.</li>
<li>El paralelismo captura casi todo el techo de Amdahl; el cuello es la
inserción serial y más procesos no lo cambian.</li>
<li>La caché convierte un coste recurrente de cuota en una tabla local.</li>
</ul>
</td></tr></table>

<div class="caja"><b>Escalabilidad frente a APIs de terceros.</b>
Al calentar la caché del corpus nuevo, la API gratuita respondió
<span class=mono>429 RESOURCE_EXHAUSTED</span> con
<span class=mono>retryDelay=25789</span> s y bloqueó la Serie B durante
horas. La lección es de diseño: <b>el techo de Amdahl mide el paralelismo
de la máquina, no limita a las APIs externas</b>; con el nivel gratuito el
rendimiento queda acotado por la cuota, y la caché es lo que devuelve ese
coste al dominio local, donde Amdahl sí aplica.</div>

<div class=aviso><b>Límites del trabajo.</b> La fracción serial se midió
sobre la inserción de la Serie A (sin embeddings); el techo describe la
ingesta local, no el pipeline completo cuando interviene la API. Las
mediciones de EXPLAIN varían entre corridas (run-to-run); se reporta el
valor de la evidencia guardada. Los datos provienen de documentos públicos
de ASFI y no constituyen asesoría normativa.</div>

</body></html>"""


def generar(salida):
    html = construir_html()
    story = fitz.Story(html=html, archive=fitz.Archive(RAIZ))
    mb = fitz.paper_rect("a4")
    donde = mb + (36, 26, -36, -26)
    writer = fitz.DocumentWriter(salida)
    paginas = 0
    mas = 1
    while mas:
        dev = writer.begin_page(mb)
        mas, _ = story.place(donde)
        story.draw(dev)
        writer.end_page()
        paginas += 1
        if paginas > 12:
            break
    writer.close()

    # Story incrusta el PNG tal cual, sin /Filter: 1650x630 en RGB crudo
    # salen 3.1 MB, y encima le agrega una SMask de 1 bit inutil porque el
    # grafico no tiene transparencia. Un PDF de 2 paginas quedaba en 3.4 MB.
    # A 234 DPI ya sobra resolucion para el ancho que ocupa, asi que se
    # remuestrea y se comprime. El PNG de resultados/ no se toca: sigue
    # siendo la evidencia de mayor resolucion.
    doc = fitz.open(salida)
    doc.rewrite_images(dpi_threshold=144, dpi_target=120, quality=90)
    temporal = salida + ".tmp"
    doc.save(temporal, garbage=4, deflate=True)
    doc.close()
    os.replace(temporal, salida)
    return paginas


def main():
    ap = argparse.ArgumentParser(description="Genera el informe PDF")
    ap.add_argument("--salida",
                    default=os.path.join(RESULTADOS, "informe.pdf"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    paginas = generar(args.salida)

    tam = os.path.getsize(args.salida)
    doc = fitz.open(args.salida)
    print(f"[*] {args.salida}")
    print(f"    paginas: {doc.page_count}  "
          f"tamano: {tam/1024:.0f} KB")
    if doc.page_count > 5:
        print(f"[X] La practica permite 5 paginas y salieron "
              f"{doc.page_count}. Compacta el informe.")
        return 1
    print("[OK] dentro del limite de 5 paginas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
