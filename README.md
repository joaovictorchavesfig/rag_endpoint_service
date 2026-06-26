# RAG API

A **Retrieval-Augmented Generation (RAG)** REST API built with Python + FastAPI, Supabase (pgvector), and OpenAI.

## Architecture

```
POST /ingest  →  embed text (text-embedding-3-small)  →  store in Supabase (pgvector)
POST /ask     →  embed question  →  vector search  →  gpt-4o-mini  →  grounded answer
```

---

## 1. Local Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

Edit `.env`:
```
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

---

## 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs available at: `http://localhost:8000/docs`

---

## 4. Endpoints

### `POST /ingest`
Store a text document with its embedding.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI is a modern web framework for building APIs with Python.",
    "metadata": {"source": "docs", "tag": "python"}
  }'
```

**Response:**
```json
{ "id": 1, "message": "Document stored successfully" }
```

---

### `POST /ask`
Ask a question — returns an LLM answer grounded in retrieved documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is FastAPI used for?",
    "match_count": 5,
    "match_threshold": 0.5
  }'
```

**Response:**
```json
{
  "answer": "FastAPI is a modern web framework...",
  "sources": [
    { "id": 1, "content": "...", "similarity": 0.91, "metadata": {} }
  ]
}
```

---

### `GET /health`
```bash
curl http://localhost:8000/health
# { "status": "ok" }
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key |
| `SUPABASE_URL` | ✅ | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✅ | Supabase **service role** key (bypasses RLS) |
| `PORT` | ❌ | Port to run on (default: 8000) |
