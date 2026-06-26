import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.ingest import router as ingest_router
from app.routes.ask import router as ask_router

# ── Logging configuration ─────────────────────────────────────────────────────
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
        "loggers": {
            "app": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        },
    }
)

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG API v1...")
    try:
        from app.services.supabase_service import get_client
        await get_client()
        logger.info("Supabase connection established.")
    except Exception as exc:
        logger.warning(f"Supabase connection failed at startup: {exc}")
    yield
    logger.info("Shutting down RAG API.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG API",
    description=(
        "Retrieval-Augmented Generation API powered by Supabase (pgvector) and OpenAI.\n\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# All routes versioned under /v1
app.include_router(ingest_router, prefix="/v1", tags=["v1 · Ingest"])
app.include_router(ask_router, prefix="/v1", tags=["v1 · Ask"])


@app.get("/v1/health", tags=["v1 · Health"])
async def health():
    """Health check — returns service status."""
    return {"status": "ok", "version": "1.0.0"}
