import logging
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=settings.openai_api_key,
)


async def embed_text(text: str) -> list[float]:
    logger.debug(f"Generating embedding for text ({len(text)} chars)")
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    logger.debug("Embedding generated successfully")
    return response.data[0].embedding


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    logger.debug(f"Generating LLM answer — {len(context_chunks)} context chunk(s)")
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the context does not contain enough information "
        "to answer the question, say so clearly.\n\n"
        f"Context:\n{context}"
    )
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    logger.debug("LLM response received")
    return response.choices[0].message.content
