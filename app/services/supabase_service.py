import logging
from supabase import acreate_client, AsyncClient
from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncClient | None = None


async def get_client() -> AsyncClient:
    """Return a lazily-created async Supabase client."""
    global _client
    if _client is None:
        logger.debug("Creating new Supabase async client")
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
    logger.debug("Inserting document into Supabase")
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
    logger.debug(f"Document inserted with id={response.data[0]['id']}")
    return response.data[0]


async def match_documents(
    query_embedding: list[float],
    match_count: int,
    match_threshold: float,
) -> list[dict]:
    """Run pgvector similarity search via the match_documents RPC function."""
    logger.debug(f"Running vector search — count={match_count}, threshold={match_threshold}")
    client = await get_client()
    response = await client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "match_threshold": match_threshold,
        },
    ).execute()
    results = response.data or []
    logger.debug(f"Vector search returned {len(results)} result(s)")
    return results
