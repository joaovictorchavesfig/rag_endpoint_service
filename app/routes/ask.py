import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from app.config import settings
from app.services.openai_service import embed_text, generate_answer
from app.services.supabase_service import match_documents

logger = logging.getLogger(__name__)

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer")
    match_count: int = Field(default=10, ge=1, le=20, description="Number of documents to retrieve")
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")


class SourceDocument(BaseModel):
    id: int
    content: str
    similarity: float
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Embed the question, search Supabase for similar documents,
    and generate an LLM answer grounded in the retrieved context.
    """
    logger.info(f"Ask request received — question: {body.question!r}")
    try:
        query_embedding = await embed_text(body.question)
        matches = await match_documents(
            query_embedding=query_embedding,
            match_count=body.match_count,
            match_threshold=body.match_threshold,
        )
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(f"Found {len(matches)} matching document(s)")

    if not matches:
        return AskResponse(
            answer="Não encontrei informações relevantes para responder à sua pergunta.",
            sources=[],
        )

    context_chunks = [m["content"] for m in matches]

    try:
        answer = await generate_answer(body.question, context_chunks)
    except Exception as exc:
        logger.error(f"LLM answer generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceDocument(
            id=m["id"],
            content=m["content"],
            similarity=round(m["similarity"], 4),
            metadata=m.get("metadata", {}),
        )
        for m in matches
    ]

    logger.info(f"Answer generated successfully for question: {body.question!r}")
    return AskResponse(answer=answer, sources=sources)
