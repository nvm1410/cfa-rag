# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Ingest all PDFs into ChromaDB
python ingest_all.py

# Start the API server
uvicorn app.main:app --reload

# Query the API
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is duration?"}'
```

- Virtual environment: `.venv/` (already activated in context)
- API key is in `.env` (DeepSeek + HF token for reranker endpoint)
- ChromaDB and PDFs are gitignored under `chroma_db/` and `data/pdfs/`

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
3. **Hybrid Search** (`hybrid_search`): Runs vector search (same BGE embedding) + BM25 search, merges via RRF. When `prefer_bm25=true`, BM25 weight is 2.5x vs vector's 1.0x; otherwise the reverse
4. **Reranking**: Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) hosted on a HF endpoint. Truncates to 512 tokens. Batches of 32, retries on 503 (cold start)
5. **Confidence Gate** (`retrieval_confident`): If top rerank score < 0.01 or the top doc isn't clearly better than the rest, returns a low-confidence response
6. **Generation**: Mode-specific system prompts (rule_application/calculation/conceptual) fed to DeepSeek-chat with temperature=0

### API (`app/main.py`)
- FastAPI on `POST /ask` with JSON `{"question": "..."}`
- Response includes `answer`, `route` (routing metadata), `contexts` (retrieved chunks with scores), `multi_queries`, `hyde_passage`

### Key Design Decisions
- **No streaming** — synchronous request/response
- **Router forces rules**: rule_application → no HyDE, prefer BM25; calculation/conceptual → HyDE on, prefer vector
- **BM25 index is rebuilt on every deploy** (or lazily on first search) from all ChromaDB docs
- **Ingestion is idempotent** via manifest hashes — safe to re-run `ingest_all.py`
