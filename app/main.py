from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.ingest import router as ingest_router
from app.routes.ask import router as ask_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up Supabase connection
    from app.services.supabase_service import get_client
    await get_client()
    yield
    # Shutdown: nothing to clean up for now


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API powered by Supabase (pgvector) and OpenAI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, tags=["Ingest"])
app.include_router(ask_router, tags=["Ask"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
