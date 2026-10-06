# Asistente Normativo RAG - ASFI / Banco Unión

Sistema RAG multi-tenant para consultar normativa de la ASFI y de Banco
Unión, con ingesta concurrente de fuentes web, aislamiento estricto por
colección y coste de embeddings acotado por caché.

## Arquitectura

| Capa | Tecnología |
| --- | --- |
| Backend | FastAPI (`/api/consultar`, `/api/ingestar-web`, `/api/colecciones`) |
| Frontend | Streamlit (chat con selector de colección) |
| Vector store | Supabase + pgvector (768 dims), particionado por `coleccion_id` |
| Ingesta PDF | PyMuPDF + Tesseract OCR (es) |
| Ingesta web | httpx + asyncio (I/O) y `ProcessPoolExecutor` para el troceado (CPU) |
| Embeddings | `gemini-embedding-001` (768 dims), por lotes con backoff ante 429 |
| LLM | `gemini-3.6-flash` con fallback a `gemini-3-flash-preview`, `temperature=0.0` |
| Caché de embeddings | local, por hash SHA-256 de (modelo + dimensión + texto) |

## Levantar el entorno

Requiere Docker. Un solo comando levanta backend, frontend y el scraper de
PDFs (este último descarga a `documentos_pdf/` y sale; los archivos ya
descargados se saltan y no consume cuota de la API):

```bash
cp .env.example .env    # completa SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY
docker compose up --build
```

- API / Swagger: http://localhost:8000/docs
- Chat: http://localhost:8501

La base de datos es externa (Supabase) y se conecta por variables de
entorno. Para crearla, ejecutar `scripts/schema.sql` (esquema base con la
RPC `match_normativa_coleccion`) y luego `scripts/migracion_coleccion.sql`
en el SQL Editor del proyecto.

### Benchmark (Serie A y B) dentro de 2 vCPU

```bash
docker compose --profile bench run --rm bench
```

Corre las series A y B dentro de un único contenedor con límite de 2 vCPU,
donde fixtures y pipeline comparten el mismo presupuesto de CPU. La Serie B
paga la API de embeddings si la caché no está caliente:

```bash
# Estado de la caché (sin gastar cuota)
python scripts/cache_embeddings.py --estado
```

## Ingesta de una colección

```bash
curl -X POST http://localhost:8000/api/ingestar-web \
  -H "Content-Type: application/json" \
  -d '{"coleccion_id":"mi_coleccion","urls":["https://ejemplo.org/doc"],"concurrency_workers":4}'
```

Cada consulta exige `coleccion_id` y la RPC `match_normativa_coleccion`
acota la búsqueda a esa colección antes de ordenar por distancia, de modo
que dos dominios distintos nunca mezclan vectores.

## Documentación

| Documento | Contenido |
| --- | --- |
| `resultados/informe.pdf` | Informe de la práctica (≤5 páginas), regenerable con `python scripts/informe_pdf.py` |
| `scripts/schema.sql` | Esquema base: pgvector, índices y RPC `match_normativa_coleccion` |
| `scripts/migracion_coleccion.sql` | Migración multi-tenant |
| `scripts/servidor_fixtures.py` | Corpus determinista de 10 URLs (81–82 KB, dentro del rango de la guía) |
| `resultados/benchmark.json` | Series A y B medidas |
| `resultados/explain_indices.json` | Planes EXPLAIN guardados como evidencia |
| `resultados/aislamiento_120.txt` | Protocolo de aislamiento con control negativo |

## Criterios de evaluación

| Criterio | Evidencia |
| --- | --- |
| Aislamiento de dominios en BD | `scripts/schema.sql`: columna `coleccion_id` con `CHECK (coleccion_id is not null and length > 0)`, índice B-Tree `normativa_bancaria_coleccion_idx` y RPC PL/pgSQL `match_normativa_coleccion` con `nb.coleccion_id = p_coleccion_id` dentro de la función, `SECURITY DEFINER` con `search_path` fijado. Verificado con `scripts/test_aislamiento_120.py`: **120 consultas, 2400 filas, 0 de otra colección, 0 en colecciones inexistentes**; el control negativo (RPC sin filtro) devuelve 20/20 + 20/20 + 20/20 filas ajenas, lo que prueba que el test detectaría contaminación. |
| Implementación concurrente | Ingesta web con `asyncio + httpx` para la descarga (I/O-bound) y `ProcessPoolExecutor` para el troceado (CPU-bound), ambos gobernados por `concurrency_workers`. Cada URL se aísla: un timeout o fallo se registra en `errores` sin abortar el lote. La inserción se hace en serie por lotes para evitar carreras y simplificar los reintentos. |
| Lotes y rate limiting | Embeddings por lotes de 10 (configurable con `EMBEDDINGS_TAMANO_LOTE`) con reintentos exponenciales 2 s → 60 s ante `429/RESOURCE_EXHAUSTED` y cadencia por minuto; inserción masiva por lotes de 200 filas con reintentos propios. La caché por SHA-256 evita pagar dos veces el mismo fragmento: en caliente la fase de embeddings resuelve 0 peticiones a la API. |
| Análisis experimental | `resultados/informe.pdf` + `resultados/benchmark.json`: Serie A con 10 URLs y 5 réplicas por configuración, `T_s`, `T_p`, `S_p` y **`E_p = S_p / p`** para `p={1,2,4,8}` (media, desviación, min/max), **desglose % por fase** (descarga, chunking, inserción; y Serie B añade embeddings fría vs caliente) y **Amdahl como techo**: con fracción serial medida 0.458 (la inserción no se paraleliza) predice 1.37/1.69/1.90 y se mide 1.38/1.57/1.73, es decir 91–100 % del techo. Todo se lee de `resultados/` (JSON y PNG). |
| Contenedorización y reproducibilidad | `Dockerfile` con `tesseract-ocr` (es), `poppler-utils` y `libmupdf`; `docker-compose.yml` con `backend`, `frontend`, `scraper` y `bench` (límite de 2 vCPU) sobre redes separadas. `docker compose up --build` levanta backend, frontend y scraper con un solo comando; el bench se lanza aparte con el perfil `bench`. La Serie B aborta de forma controlada si falta cuota y conserva la Serie A ya medida. |

## Estructura

```
├── main.py                     # API: consultas, colecciones e ingesta web
├── app_streamlit.py            # Chat con selector de colección
├── rag/
│   ├── clients.py              # Clientes lazy, modelos y ajustes de cuota
│   └── embeddings.py           # Caché SHA-256, lotes, backoff 429
├── scripts/
│   ├── schema.sql              # Esquema base, índices y RPC
│   ├── migracion_coleccion.sql # Migración multi-tenant
│   ├── ingest_normativa.py     # PDF: PyMuPDF + Tesseract OCR
│   ├── descargar_asfi.py       # Scraper concurrente de PDFs
│   ├── benchmark.py            # Serie A y B (10 URLs, réplicas, Amdahl)
│   ├── servidor_fixtures.py    # Corpus determinista para medir
│   ├── cache_embeddings.py     # Calentamiento y estado de la caché
│   ├── graficos.py             # Speedup/Amdahl, fases y desglose Serie B
│   ├── test_aislamiento_120.py # Protocolo de 120 consultas
│   ├── explicar_indices.py     # Planes EXPLAIN a JSON
│   └── informe_pdf.py          # Informe ≤5 páginas desde resultados/
├── resultados/                 # Evidencia: JSON, PNG y informe.pdf
├── cache_embeddings/           # Vectores cacheados (sin coste de API)
└── .env.example                # Variables de entorno documentadas
```

## Desarrollo fuera de Docker

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
streamlit run app_streamlit.py
```

Tras modificar `.env` en Docker usa `docker compose up -d --force-recreate`:
`restart` no recarga el entorno del contenedor.