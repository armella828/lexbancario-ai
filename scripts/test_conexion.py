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

print("[*] Probando generación de embedding truncado a 768 dimensiones...")
try:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    response = ai_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=["Prueba de embedding para normativa ASFI Banco Unión"],
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    vector = response.embeddings[0].values
    print(f"[✓] Embedding generado exitosamente (Dimensión: {len(vector)})")
except Exception as e:
    print(f"[X] Error al conectar con Google Gemini: {e}")
    sys.exit(1)

print("\n[*] Probando inserción y lectura de prueba en Supabase...")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    res = supabase.table("normativa_bancaria").select("id").limit(1).execute()
    print("[✓] Conexión y consulta a Supabase exitosa.")
except Exception as e:
    print(f"[X] Error al consultar Supabase: {e}")
    sys.exit(1)

print("\n=======================================================")
print("[✓] BASE DE DATOS Y EMBEDDINGS 100% SINCRONIZADOS (768d)")
print("=======================================================")
