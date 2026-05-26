# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Ingest all PDFs into ChromaDB
python ingest_all.py

# Start the API server
uvicorn app.main:app --reload

# Single-turn RAG query
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is duration?"}'

# Authentication
curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d '{"username":"alice","password":"test1234"}'
curl -X POST http://localhost:8000/auth/login    -H "Content-Type: application/json" -d '{"username":"alice","password":"test1234"}'

# Sessions CRUD (use token from auth response)
TOKEN="<token>"
curl -X POST  http://localhost:8000/sessions -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"title":"My Chat"}'
curl http://localhost:8000/sessions -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/sessions/<id> -H "Authorization: Bearer $TOKEN"
curl -X DELETE http://localhost:8000/sessions/<id> -H "Authorization: Bearer $TOKEN"

# Conversational RAG (history-aware)
curl -X POST http://localhost:8000/chat/sessions/<id>/ask -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"question":"What is duration?"}'

# UI (dev server, proxied to backend on port 8000)
cd cfa-rag-ui && npm run dev
```

- Virtual environment: `.venv/` (already activated in context)
- API keys in `.env`: `DEEPSEEK_API_KEY`, `HF_TOKEN`, `HF_RERANKER_URL`
- Data is stored in `chroma_db/` (vector DB), `data/pdfs/` (source PDFs), and `app.db` (SQLite for auth/sessions/messages) — all gitignored

## Architecture

### Ingestion (`app/ingest.py`)
- Extracts PDF text using PyMuPDF (`fitz`), sorts blocks top-to-bottom by y-coordinate
- Uses `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150) with a paragraph-buffering strategy that groups paragraphs until ~1200 chars before splitting
- Embeds chunks with `BAAI/bge-small-en-v1.5` via `FlagEmbedding.FlagModel`
- Stores in ChromaDB (persistent, `./chroma_db`)
- **Manifest system**: MD5 hash per file in `chroma_db/ingested_files.json` — only re-ingests when content changes; `force=True` to override
- Heuristics tag chunks as `table`, `formula`, or `text` in metadata (used downstream but not by routing currently)
- `remove_pdf(source_name)` deletes a file's chunks and updates manifest

### RAG Pipeline (`app/rag.py`)
The `ask_rag()` function runs this pipeline on every query:

1. **Routing** (`route_query`): LLM (DeepSeek-chat) classifies the query into: relevant/not, query_type (ethics/quant/etc), reasoning_mode (rule_application/calculation/conceptual), use_hyde, prefer_bm25, difficulty
2. **Query Expansion** (`app/query_expansion.py`): Optionally generates a HyDE passage and up to 3 multi-queries via DeepSeek. Number of multi-queries varies by reasoning_mode (0 for rule_application, 1 for calculation, 3 for conceptual)
3. **Hybrid Search** (`hybrid_search`): Runs vector search (same BGE embedding) + BM25 search, merges via RRF. When `prefer_bm25=true`, BM25 weight is 2.5x vs vector's 1.0x; otherwise the reverse. Retrieval k scales by difficulty (high=100, medium=60, low=40)
4. **Reranking**: Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) hosted on a HF endpoint. Truncates to 512 tokens. Batches of 32, retries on 503 (cold start). Top-k scales by difficulty (high=20, else=15)
5. **Confidence Gate** (`retrieval_confident`): If top rerank score < 0.01 or the top doc isn't clearly better than the rest, returns a low-confidence response
6. **Generation**: Mode-specific system prompts (rule_application/calculation/conceptual) fed to DeepSeek-chat with temperature=0

### Conversational RAG (`app/chat.py`)
The `ask_chat()` function extends the RAG pipeline with session awareness:

1. **Query Rewriting**: Standalone query rewrite using conversation history to resolve pronouns and implicit references
2. **Relevance Pre-check**: Lightweight LLM gate (faster than full routing) — cheap filter for clearly non-CFA queries
3. **Routing**: Same `route_query` as single-turn pipeline (on the rewritten query)
4. **Pipeline**: Query expansion → hybrid search → rerank → confidence gate → generation (same as single-turn)
5. **Generation with History**: Injects conversation history into the LLM prompt alongside retrieved context
6. **Persistence**: Saves user/assistant messages with metadata (rewritten query, route, context IDs)
7. **Auto-titling**: Titles session from first user message
8. **Periodic Summaries**: Regenerates session summary every 5 exchanges for efficient context recall

Key differences from single-turn: history-aware rewrite, `quick_relevance_check()` gate before routing, conversation history injected into generation prompt, full message persistence, auto-summary.

### Frontend (`cfa-rag-ui/`)
- **Stack**: React 18 + TypeScript, Vite, no router (simple page-switching via state)
- **Proxy**: Vite dev server proxies `/auth`, `/sessions`, `/chat`, `/ask` to `localhost:8000`
- **Auth flow**: Login/register stores JWT in `sessionStorage`, attaches as `Authorization: Bearer` header

### API (`app/main.py`)
- `GET /` — health check
- `POST /ask` — single-turn RAG with `{"question": "..."}`
- `include_router(auth_router)` at `/auth`
- `include_router(sessions_router)` at `/sessions`
- `include_router(chat_router)` at `/chat`

### Authentication (`app/auth.py`)
- **JWT (HS256)**: No external dependencies — hand-rolled `hmac` + `base64` token creation/verification
- **Password hashing**: `bcrypt` (via `bcrypt` package)
- **Endpoints**: `POST /auth/register`, `POST /auth/login`
- **Auth dependency**: `get_current_user()` — FastAPI `Depends` that reads `Authorization: Bearer <token>`, decodes JWT, and returns the `User` model; raises 401 on failure
- Configurable `JWT_SECRET` env var (defaults to dev secret), 24h TTL

### Session Management (`app/sessions.py`)
- **Endpoints**: `POST /sessions` (create), `GET /sessions` (list), `GET /sessions/{id}` (detail with messages), `DELETE /sessions/{id}`, `PATCH /sessions/{id}/title`
- **Ownership**: All endpoints verify `session.user_id == current_user.id` — return 403 on mismatch
- Messages are returned in chronological order on detail view

### Storage (`app/storage.py`)
- **Backend**: SQLite via SQLAlchemy ORM
- **Models**: `User` (id, username, password_hash, created_at), `ChatSession` (id, user_id, title, summary, created_at, updated_at), `Message` (id, session_id, role, content, metadata_json, created_at)
- Tables auto-created on import via `Base.metadata.create_all()`
- Functions: `get_user_by_username`, `get_user_by_id`, `create_user`, `create_session`, `get_user_sessions`, `get_session`, `update_session`, `delete_session`, `add_message`, `get_session_messages`, `count_session_messages`

### Pydantic Schemas (`app/schemas.py`)
Request/response models for auth (`RegisterRequest`, `LoginRequest`, `AuthResponse`), sessions (`SessionCreateRequest`, `SessionResponse`, `SessionDetailResponse`), and chat (`AskRequest`, `ChatAskResponse`).

### Key Design Decisions
- **No streaming** — synchronous request/response
- **Router forces rules**: rule_application → no HyDE, prefer BM25; calculation/conceptual → HyDE on, prefer vector
- **BM25 index is rebuilt on every deploy** (or lazily on first search) from all ChromaDB docs
- **Ingestion is idempotent** via manifest hashes — safe to re-run `ingest_all.py`
- **JWT zero-dependency**: Uses stdlib `hmac` + `base64` instead of `PyJWT` — no extra dependency needed
- **SQLite for state**: Conversation history, auth, and sessions live in `app.db` (sqlite) — no external DB needed
