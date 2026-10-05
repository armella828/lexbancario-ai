import os
import socket
import time
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import Client
from rag.clients import ENTORNO, MODEL_EMBEDDING, MODELOS_GENERACION, config_embeddings
from rag.pipeline import ingestar

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Faltan variables de entorno requeridas en el archivo .env")

supabase: Client = ENTORNO.supabase
ai_client = ENTORNO.ai

app = FastAPI(
    title="API RAG Normativa Bancaria - ASFI / Banco Unión",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConsultaRequest(BaseModel):
    pregunta: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    coleccion_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Coleccion tematica obligatoria. La busqueda se restringe a este espacio.",
    )
    top_k: Optional[int] = Field(4, ge=1, le=20)
    match_threshold: Optional[float] = Field(0.35, ge=-1.0, le=1.0)

class FuenteNormativa(BaseModel):
    documento: str
    articulo_ref: str
    coleccion_id: Optional[str] = None
    similitud: float
    contenido: str

class ConsultaResponse(BaseModel):
    pregunta: str
    coleccion_id: str
    respuesta: str
    fuentes: List[FuenteNormativa]

class IngestaWebRequest(BaseModel):
    """Payload de ingesta web concurrente (Requisito 2 de la guia)."""

    coleccion_id: str = Field(
        ..., min_length=1, max_length=50,
        description="Namespace destino de los vectores generados",
    )
    urls: List[str] = Field(..., min_length=1, description="URLs a descargar y trocear")
    chunk_size: Optional[int] = Field(1000, ge=100, le=8000)
    chunk_overlap: Optional[int] = Field(200, ge=0)
    concurrency_workers: Optional[int] = Field(
        4, ge=1, le=32,
        description="Peticiones HTTP simultaneas maximas",
    )
    batch_size: Optional[int] = Field(
        None, ge=1, le=100,
        description=(
            "Fragmentos por llamada a la API de embeddings. Si se omite se usa "
            "EMBEDDINGS_TAMANO_LOTE del entorno."
        ),
    )

    def validar_solapamiento(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) debe ser menor que "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class IngestaWebResponse(BaseModel):
    coleccion_id: str
    urls_recibidas: int
    urls_ok: int
    urls_fallidas: int
    fragmentos_generados: int
    fragmentos_insertados: int
    segundos_totales: float
    segundos_por_fase: dict
    porcentaje_por_fase: dict
    errores: List[str]
    detalle_por_url: List[dict]

def reintentar(fn, intentos=3, espera_base=3):
    """Reintenta operaciones de red ante errores transitorios (DNS, conexion, timeouts, 429)."""
    ultimo_error = None
    for i in range(intentos):
        try:
            return fn()
        except Exception as e:
            ultimo_error = e
            err_str = str(e)
            es_transitorio = (
                isinstance(e, (socket.gaierror, httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout))
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "Name or service not known" in err_str
            )
            if es_transitorio:
                time.sleep(espera_base * (i + 1))
                continue
            break
    raise ultimo_error

def buscar_contexto(
    pregunta: str,
    coleccion_id: str,
    top_k: int,
    match_threshold: float,
):
    """Busca fragmentos restringidos a una coleccion (aislamiento multi-tenant).

    La llamada va a `match_normativa_coleccion`, no a `match_normativa`: la
    primera aplica el filtro `coleccion_id = p_coleccion_id` dentro de la
    base, de modo que las tuplas de otros tenants ni siquiera llegan a la
    comparacion coseno.
    """
    res_emb = reintentar(lambda: ai_client.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=[pregunta],
        config=config_embeddings()
    ))
    query_vector = res_emb.embeddings[0].values

    rpc_res = reintentar(lambda: supabase.rpc("match_normativa_coleccion", {
        "query_embedding": query_vector,
        "match_threshold": match_threshold,
        "match_count": top_k,
        "p_coleccion_id": coleccion_id,
    }).execute())

    return rpc_res.data or []

def generar_con_respaldo(prompt: str) -> str:
    """Genera respuesta con Gemini recorriendo modelos de respaldo ante 503/429."""
    ultimo_error = None
    for modelo in MODELOS_GENERACION:
        try:
            res = ai_client.models.generate_content(
                model=modelo,
                contents=prompt,
                # temperature=0.0: la guia exige inferencia determinista,
                # de modo que la misma pregunta sobre el mismo contexto
                # produzca siempre la misma respuesta.
                config={"temperature": 0.0, "max_output_tokens": 1024}
            )
            if not res.text and res.candidates and res.candidates[0].content:
                res.text = "".join(p.text or "" for p in res.candidates[0].content.parts)
            if res.text and res.text.strip():
                return res.text
            ultimo_error = ValueError("Respuesta vacia del modelo")
        except Exception as e:
            ultimo_error = e
            err_str = str(e)
            if any(k in err_str for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "quota")):
                time.sleep(2)
                continue
            break
    raise HTTPException(
        status_code=503,
        detail="Google AI Studio no respondio (cuota diaria agotada o alta demanda en los modelos). Reintenta en unos segundos o con otra API key."
    )

@app.get("/")
def estado():
    return {"status": "online", "mensaje": "API RAG de Normativa Bancaria activa"}

@app.post("/api/consultar", response_model=ConsultaResponse)
def consultar_normativa(req: ConsultaRequest):
    if not req.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        docs = buscar_contexto(
            req.pregunta, req.coleccion_id, req.top_k, req.match_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda vectorial: {e}")

    # Regla de no-alucinacion: si la coleccion solicitada no aporta
    # antecedentes, se declara explicitamente en lugar de inventar una
    # respuesta a partir de conocimiento general del modelo.
    if not docs:
        return ConsultaResponse(
            pregunta=req.pregunta,
            coleccion_id=req.coleccion_id,
            respuesta=(
                f"La colección '{req.coleccion_id}' no contiene antecedentes "
                "normativos suficientes para responder esta consulta. "
                "No puedo fundamentar una respuesta con la información disponible "
                "y no debo inferirla de conocimiento externo al corpus indexado."
            ),
            fuentes=[]
        )

    contexto_texto = ""
    fuentes_resp = []
    for d in docs:
        contexto_texto += (
            f"\n--- DOCUMENTO: {d.get('documento_origen')} | "
            f"REFERENCIA: {d.get('articulo_ref')} | "
            f"COLECCION: {d.get('coleccion_id')} ---\n"
            f"{d.get('contenido')}\n"
        )
        fuentes_resp.append(FuenteNormativa(
            documento=d.get("documento_origen", ""),
            articulo_ref=d.get("articulo_ref", ""),
            coleccion_id=d.get("coleccion_id"),
            similitud=round(d.get("similarity", 0) * 100, 2),
            contenido=d.get("contenido", "")
        ))

    prompt = f"""Eres un asesor legal bancario especializado en normativa de la ASFI y Banco Unión.
Responde de forma rigurosa, clara y estructurada utilizando EXCLUSIVAMENTE el siguiente contexto normativo.
Cita siempre el artículo o resolución de respaldo.

REGLAS INVIOLABLES:
1. Solo puedes afirmar lo que aparece literalmente en el CONTEXTO NORMATIVO.
2. No complementes con conocimiento general, memoria o suposiciones, aunque lo creas correcto.
3. Si el contexto no contiene la respuesta, indícalo de forma expresa y explica qué información adicional haría falta.
4. Los fragmentos provienen exclusivamente de la colección '{req.coleccion_id}'. Si el texto de la
   pregunta sugiere que pertenece a otra colección, indícalo en lugar de forzar una respuesta.
5. No inventes números de artículo, montos, plazos ni referencias que no aparezcan en el contexto.

CONTEXTO NORMATIVO:
{contexto_texto}

PREGUNTA:
{req.pregunta}

RESPUESTA:"""

    respuesta_texto = generar_con_respaldo(prompt)

    return ConsultaResponse(
        pregunta=req.pregunta,
        coleccion_id=req.coleccion_id,
        respuesta=respuesta_texto,
        fuentes=fuentes_resp
    )


@app.get("/api/colecciones")
def listar_colecciones():
    """Inventario de colecciones indexadas y su tamano."""
    try:
        filas = supabase.rpc("listar_colecciones").execute().data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar colecciones: {e}")
    return {"colecciones": filas}


@app.post("/api/ingestar-web", response_model=IngestaWebResponse)
def ingestar_web(req: IngestaWebRequest):
    """Ingesta concurrente de una lista de URLs en una coleccion.

    Las peticiones HTTP se atienden con asyncio (I/O-bound) y el troceado
    se delega a procesos (CPU-bound). Cada URL se aísla: un fallo o timeout
    se registra en `errores` sin abortar el resto del lote.
    """
    try:
        req.validar_solapamiento()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Si los dos parametros no coinciden, el benchmark no seria comparable.
    # `concurrency_workers` gobierna las descargas y el troceado usa el
    # mismo numero de procesos salvo que se fije explicitamente.
    resultado = ingestar(
        urls=req.urls,
        coleccion_id=req.coleccion_id,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        concurrencia=req.concurrency_workers,
        procesos_chunking=req.concurrency_workers,
        tamano_lote=req.batch_size,
    )

    resumen = resultado.resumen()
    return IngestaWebResponse(
        coleccion_id=resultado.coleccion_id,
        urls_recibidas=len(req.urls),
        urls_ok=resultado.descargadas_ok,
        urls_fallidas=resultado.descargadas_fallidas,
        fragmentos_generados=resultado.fragmentos,
        fragmentos_insertados=resultado.insertados,
        segundos_totales=resumen["segundos_totales"],
        segundos_por_fase=resumen["segundos_por_fase"],
        porcentaje_por_fase=resumen["porcentaje_por_fase"],
        errores=resultado.errores,
        detalle_por_url=resultado.por_url,
    )
