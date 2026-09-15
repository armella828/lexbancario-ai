import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_EMBEDDING = "models/gemini-embedding-001"
MODEL_GENERACION = "gemini-2.5-flash"

def buscar_contexto(pregunta: str, top_k: int = 4):
    """Genera el embedding de la consulta y llama a la función RPC en Supabase."""
    res_emb = ai_client.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=[pregunta],
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    query_vector = res_emb.embeddings[0].values

    # Llamada a la función RPC match_normativa creada en PostgreSQL
    respuesta = supabase.rpc("match_normativa", {
        "query_embedding": query_vector,
        "match_threshold": 0.35,
        "match_count": top_k
    }).execute()

    return respuesta.data

def responder_consulta(pregunta: str):
    print(f"\n[*] Pregunta: {pregunta}")
    print("[*] Buscando fragmentos normativos relevantes en Supabase...")
    
    docs = buscar_contexto(pregunta)
    
    if not docs:
        print("[!] No se encontraron fragmentos normativos con suficiente similitud.")
        return

    print(f"[✓] Se recuperaron {len(docs)} fragmentos relevantes.\n")
    
    # Construcción del contexto y listado de fuentes
    contexto_texto = ""
    for idx, d in enumerate(docs, start=1):
        print(f"    Fuente [{idx}]: {d.get('documento_origen')} | {d.get('articulo_ref')} (Similitud: {round(d.get('similarity', 0) * 100, 2)}%)")
        contexto_texto += f"\n--- DOCUMENTO: {d.get('documento_origen')} | REFERENCIA: {d.get('articulo_ref')} ---\n{d.get('contenido')}\n"

    prompt_instruccion = f"""Eres un asistente legal y normativo experto para Banco Unión y regulaciones ASFI.
Tu tarea es responder con total rigurosidad a la consulta del usuario basándote EXCLUSIVAMENTE en los fragmentos de normativa provistos.

Reglas:
1. Si la respuesta no se encuentra en el contexto, declara: "No cuento con información suficiente en los documentos cargados para responder esto con certeza."
2. Cita siempre el artículo, resolución o sección específica que respalda cada afirmación.
3. Mantén un tono técnico, claro y profesional.

CONTEXTO NORMATIVO:
{contexto_texto}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA FUNDAMENTADA:"""

    print("\n[*] Generando respuesta con Gemini...")
    res_llm = ai_client.models.generate_content(
        model=MODEL_GENERACION,
        contents=prompt_instruccion
    )
    
    print("\n" + "="*70)
    print("RESPUESTA NORMATIVA:")
    print("="*70)
    print(res_llm.text)
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        consulta = " ".join(sys.argv[1:])
    else:
        # Pregunta de prueba predeterminada
        consulta = "¿Qué establece la normativa respecto a las facultades de la ASFI y la Constitución Política del Estado?"
    
    responder_consulta(consulta)
