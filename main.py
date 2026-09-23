import os
import socket
import time
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError("Faltan variables de entorno requeridas en el archivo .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_EMBEDDING = "models/gemini-embedding-001"
MODELOS_GENERACION = ["gemini-3.6-flash", "gemini-3-flash-preview"]

app = FastAPI(
    title="API RAG Normativa Bancaria - ASFI / Banco Unión",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConsultaRequest(BaseModel):
    pregunta: str
    top_k: Optional[int] = 4
    match_threshold: Optional[float] = 0.35

class FuenteNormativa(BaseModel):
    documento: str
    articulo_ref: str
    similitud: float
    contenido: str

class ConsultaResponse(BaseModel):
    pregunta: str
    respuesta: str
    fuentes: List[FuenteNormativa]

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

def buscar_contexto(pregunta: str, top_k: int, match_threshold: float):
    res_emb = reintentar(lambda: ai_client.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=[pregunta],
        config=types.EmbedContentConfig(output_dimensionality=768)
    ))
    query_vector = res_emb.embeddings[0].values

    rpc_res = reintentar(lambda: supabase.rpc("match_normativa", {
        "query_embedding": query_vector,
        "match_threshold": match_threshold,
        "match_count": top_k
    }).execute())

    return rpc_res.data or []

def generar_con_respaldo(prompt: str) -> str:
    """Genera respuesta con Gemini aplicando reintentos y fallback a modelo alternativo si hay 503."""
    ultimo_error = None
    for modelo in MODELOS_GENERACION:
        for intento in range(2):
            try:
                res = ai_client.models.generate_content(
                    model=modelo,
                    contents=prompt
                )
                return res.text
            except Exception as e:
                ultimo_error = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(2)
                    continue
                else:
                    break
    raise HTTPException(status_code=503, detail=f"Servicio LLM no disponible temporalmente: {ultimo_error}")

@app.get("/")
def estado():
    return {"status": "online", "mensaje": "API RAG de Normativa Bancaria activa"}

@app.post("/api/consultar", response_model=ConsultaResponse)
def consultar_normativa(req: ConsultaRequest):
    if not req.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    try:
        docs = buscar_contexto(req.pregunta, req.top_k, req.match_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda vectorial: {e}")

    if not docs:
        return ConsultaResponse(
            pregunta=req.pregunta,
            respuesta="No cuento con información suficiente en los documentos cargados para responder esto con certeza.",
            fuentes=[]
        )

    contexto_texto = ""
    fuentes_resp = []
    for d in docs:
        contexto_texto += f"\n--- DOCUMENTO: {d.get('documento_origen')} | REFERENCIA: {d.get('articulo_ref')} ---\n{d.get('contenido')}\n"
        fuentes_resp.append(FuenteNormativa(
            documento=d.get("documento_origen", ""),
            articulo_ref=d.get("articulo_ref", ""),
            similitud=round(d.get("similarity", 0) * 100, 2),
            contenido=d.get("contenido", "")
        ))

    prompt = f"""Eres un asesor legal bancario especializado en normativa de la ASFI y Banco Unión.
Responde de forma rigurosa, clara y estructurada utilizando EXCLUSIVAMENTE el siguiente contexto normativo.
Cita siempre el artículo o resolución de respaldo. Si algo no figura en el texto provisto, acláralo expresamente.

CONTEXTO NORMATIVO:
{contexto_texto}

PREGUNTA:
{req.pregunta}

RESPUESTA:"""

    respuesta_texto = generar_con_respaldo(prompt)

    return ConsultaResponse(
        pregunta=req.pregunta,
        respuesta=respuesta_texto,
        fuentes=fuentes_resp
    )
