from supabase import acreate_client, AsyncClient
from app.config import settings

_client: AsyncClient | None = None


async def get_client() -> AsyncClient:
    """Return a lazily-created async Supabase client."""
    global _client
    if _client is None:
        _client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
    return _client


async def insert_document(
    content: str,
    embedding: list[float],
    metadata: dict,
) -> dict:
    """Insert a document with its embedding into the documents table."""
    client = await get_client()
    response = (
        await client.table("documents")
        .insert(
            {
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            }
        )
        .execute()
    )
    return response.data[0]


async def match_documents(
    query_embedding: list[float],
    match_count: int,
    match_threshold: float,
) -> list[dict]:
    """Run pgvector similarity search via the match_documents RPC function."""
    client = await get_client()
    response = await client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "match_threshold": match_threshold,
        },
    ).execute()
    return response.data or []
