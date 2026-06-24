from openai import AsyncOpenAI
from app.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_text(text: str) -> list[float]:
    """Convert text to an embedding vector using the cheapest model."""
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Generate a grounded LLM answer from retrieved context chunks."""
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
    return response.choices[0].message.content
