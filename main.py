import asyncio
import os
import sys
import time
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ultra_core_engine import ZoliUltraEngine
from ultra_retriever import UltraHybridRetriever
from ultra_web_agent import UltraWebAgent

app = FastAPI(
    title="ZoliGPT API",
    version="3.0.0",
    description="Zéró-latenciájú, aszinkron hibrid RAG és webes kereső motor."
)

def load_embedder():
    if SentenceTransformer is None:
        logging.warning("sentence-transformers nincs telepítve; a keresés BM25 módban indul.")
        return None
    try:
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as error:
        logging.warning("A szemantikus modell nem tölthető be; BM25 fallback: %s", error)
        return None


embedder_model = load_embedder()
retriever = UltraHybridRetriever(
    embedder=embedder_model,
    database_path=os.getenv("ZOLI_DATABASE_PATH", "zoli_gpt_enterprise.db"),
    username=os.getenv("ZOLI_USERNAME"),
    documents_dir=os.getenv("ZOLI_DOCUMENTS_DIR"),
    min_dense_similarity=float(os.getenv("ZOLI_MIN_DENSE_SIMILARITY", "0.45")),
)
web_agent = UltraWebAgent(timeout_seconds=5.0)

groq_key = os.getenv("GROQ_API_KEY", "")
engine = ZoliUltraEngine(groq_api_key=groq_key, embedder=embedder_model)

class ChatRequest(BaseModel):
    query: str
    use_web: bool = True
    use_rag: bool = True


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "indexed_chunks": retriever.document_count,
        "dense_search": embedder_model is not None,
    }


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="A kérdés nem lehet üres.")

    rag_provider = retriever.retrieve_and_compress if request.use_rag else None
    web_provider = web_agent.search_and_extract if request.use_web else None

    async def generate_chunks() -> AsyncGenerator[str, None]:
        async for chunk in engine.stream_response(
            user_query=request.query,
            rag_context_provider=rag_provider,
            web_search_provider=web_provider
        ):
            yield chunk

    return StreamingResponse(generate_chunks(), media_type="text/plain")

async def run_cli():
    print("\n" + "=" * 50)
    print("  ZoliGPT Ultra v3.0 - Interaktív CLI Engine")
    print("  Gépelj 'exit'-et a kilépéshez.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("\n[Kérdés] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "kilépés"]:
                print("Viszlát!")
                break

            print("[Válasz] > ", end="", flush=True)
            async for chunk in engine.stream_response(
                user_query=user_input,
                rag_context_provider=retriever.retrieve_and_compress,
                web_search_provider=web_agent.search_and_extract
            ):
                print(chunk, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nMegszakítva.")
            break


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(run_cli())
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)