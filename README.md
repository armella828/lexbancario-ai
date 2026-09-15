# 🏛️ Asistente Normativo RAG - ASFI / Banco Unión

Sistema RAG para consulta y análisis de normativa ASFI y Banco Unión.

## Arquitectura
- Backend: FastAPI
- Frontend: Streamlit
- Vector Store: Supabase (pgvector, 768 dims)
- Modelos: gemini-2.5-flash y gemini-embedding-001
- Ingestión: PyMuPDF + Tesseract OCR

## Despliegue Local
1. Instalar dependencias: pip install -r requirements.txt
2. Configurar .env con llaves de Supabase y Gemini
3. Iniciar API: uvicorn main:app --reload --port 8000
4. Iniciar Frontend: streamlit run app_streamlit.py
