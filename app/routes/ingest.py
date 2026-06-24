from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.openai_service import embed_text
from app.services.supabase_service import insert_document

router = APIRouter()


class IngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Text to store and embed")
    metadata: dict = Field(default_factory=dict, description="Optional metadata (source, tag, etc.)")


class IngestResponse(BaseModel):
    id: int
    message: str


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(body: IngestRequest):
    """
    Receive text, convert to embedding via OpenAI, and store in Supabase.
    """
    try:
        embedding = await embed_text(body.content)
        document = await insert_document(
            content=body.content,
            embedding=embedding,
            metadata=body.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(id=document["id"], message="Document stored successfully")
