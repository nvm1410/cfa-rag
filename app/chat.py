"""Conversational RAG with history-aware query rewriting, relevance gating,
and session-backed message persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.storage import User, get_session, get_session_messages, add_message, update_session, count_session_messages
from app.schemas import AskRequest, ChatAskResponse
from app.rag import (
    llm,
    route_query,
    hybrid_search,
    rerank,
    retrieval_confident,
    build_context,
    get_system_prompt,
    _FINAL_DOCS_K,
    _N_MULTI,
)
from app.query_expansion import expand_query, ExpandedQuery

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMMARY_INTERVAL = 5  # regenerate summary every N exchanges (user+assistant = 1 exchange)
MAX_HISTORY_FOR_REWRITE = 8  # last N messages fed to query rewriter
MAX_HISTORY_FOR_GENERATION = 6  # last N messages injected into generation prompt


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a query rewriter for a CFA RAG system. Given a conversation history and the latest user query, rewrite the query into a standalone search query that captures all necessary context from the conversation.

Rules:
- If the latest query is self-contained, return it verbatim.
- If it refers to prior context (pronouns like "it", "that", "they", "these"), expand them with concrete terms from the history.
- Keep precise CFA and financial terminology.
- Output ONLY the rewritten query — no explanations, no prefixes.

Conversation history:
{history}

Latest query: {query}

Rewritten query:"""


def _format_history(messages, limit: int) -> str:
    """Format recent messages as a dialogue for prompt injection."""
    recent = messages[-limit:] if len(messages) > limit else messages
    parts = []
    for m in recent:
        role = "User" if m.role == "user" else "Assistant"
        parts.append(f"{role}: {m.content}")
    return "\n\n".join(parts)


def rewrite_query(query: str, history: str, summary: str = "") -> str:
    """Standalone retrieval query that absorbs conversation context."""
    if not history:
        return query

    context = history
    if summary:
        context = f"Session summary: {summary}\n\nRecent messages:\n{history}"

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "user", "content": REWRITE_PROMPT.format(history=context, query=query)},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Lightweight relevance pre-check
# ---------------------------------------------------------------------------

RELEVANCE_CHECK_PROMPT = """You are a classifier for a CFA exam RAG system. Determine if the query below is related to finance, investing, economics, accounting, ethical standards, or the CFA curriculum.

Be conservative. Only return "irrelevant" if the query is clearly, unambiguously unrelated to finance or investing (e.g. cooking recipes, sports scores, pop culture, tech support, general trivia not connected to finance).

When in doubt, return "relevant".

Query: {query}

Reply with exactly one word: "relevant" or "irrelevant"."""


def quick_relevance_check(query: str) -> bool:
    """Lightweight LLM gate — fast, cheap, conservative."""
    resp = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "user", "content": RELEVANCE_CHECK_PROMPT.format(query=query)},
        ],
    )
    return resp.choices[0].message.content.strip().lower() != "irrelevant"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Summarize this CFA tutoring conversation in 2-3 sentences. Focus on the main topics discussed and the key questions asked.

{history}

Summary:"""


def _update_summary(session_id: str) -> None:
    """Regenerate the session summary from recent messages."""
    messages = get_session_messages(session_id)
    history = _format_history(messages[-10:], limit=10)

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "user", "content": SUMMARY_PROMPT.format(history=history)},
        ],
    )
    summary = resp.choices[0].message.content.strip()
    update_session(session_id, summary=summary)


# ---------------------------------------------------------------------------
# Conversational RAG
# ---------------------------------------------------------------------------

def ask_chat(session_id: str, user_id: int, question: str) -> dict:
    """Run the full conversational RAG pipeline inside a session."""

    # 1. Verify ownership
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # 2. Load history
    messages = get_session_messages(session_id)
    history = _format_history(messages, MAX_HISTORY_FOR_REWRITE)

    # 3. Rewrite query using conversation context
    rewritten = rewrite_query(question, history, session.summary)

    # 4. Lightweight relevance pre-check (on rewritten query)
    relevant = quick_relevance_check(rewritten)
    if not relevant:
        # Persist the exchange even for irrelevant queries
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", "This question does not appear to be related to the CFA curriculum.")
        return {
            "answer": "This question does not appear to be related to the CFA curriculum.",
            "session_id": session_id,
            "relevant": False,
        }

    # 5. Route query
    route = route_query(rewritten)

    if not route["relevant"]:
        add_message(session_id, "user", question)
        add_message(session_id, "assistant",
                     "This question does not appear to be related to the CFA curriculum.")
        return {
            "answer": "This question does not appear to be related to the CFA curriculum.",
            "session_id": session_id,
            "relevant": False,
            "route": route,
        }

    reasoning_mode = route["reasoning_mode"]
    difficulty = route["difficulty"]

    # 6. Query expansion
    n_multi = _N_MULTI.get(reasoning_mode, 3)
    expanded: ExpandedQuery = expand_query(rewritten, n_multi=n_multi, use_hyde=route["use_hyde"])

    # 7. Determine retrieval size
    retrieval_k = {"high": 100, "medium": 60}.get(difficulty, 40)

    # 8. Hybrid search
    candidates = hybrid_search(expanded, route, k=retrieval_k)

    # 9. Rerank
    rerank_top_k = 20 if difficulty == "high" else 15
    reranked = rerank(rewritten, candidates, top_k=rerank_top_k)

    # 10. Confidence gate
    if not retrieval_confident(reranked):
        add_message(session_id, "user", question)
        add_message(
            session_id, "assistant",
            "I could not find sufficiently relevant context to answer confidently.",
        )
        return {
            "answer": "I could not find sufficiently relevant context to answer confidently.",
            "session_id": session_id,
            "relevant": True,
            "route": route,
            "contexts": reranked[:5],
            "multi_queries": expanded.all_queries,
            "hyde_passage": expanded.hyde_passage,
        }

    # 11. Final context
    final_k = _FINAL_DOCS_K.get(reasoning_mode, 6)
    final_docs = reranked[:final_k]
    context = build_context(final_docs)

    # 12. Build generation prompt with conversation history
    gen_history = _format_history(messages, MAX_HISTORY_FOR_GENERATION)
    system_prompt = get_system_prompt(route)

    user_content = f"Conversation history:\n{gen_history}\n\n---\n\nContext:\n{context}\n\nQuestion:\n{rewritten}"
    if not gen_history:
        user_content = f"Context:\n{context}\n\nQuestion:\n{rewritten}"

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    answer = resp.choices[0].message.content

    # 13. Persist messages
    add_message(session_id, "user", question, metadata={
        "rewritten_query": rewritten,
        "route": route,
    })
    add_message(session_id, "assistant", answer, metadata={
        "contexts": [{"id": d["id"], "score": d.get("rerank_score")} for d in final_docs],
    })

    # 14. Auto-title on first exchange
    if len(messages) <= 1:
        title = question[:50] + "..." if len(question) > 50 else question
        update_session(session_id, title=title)

    # 15. Periodic summary update (every N exchanges)
    total = count_session_messages(session_id)
    if total > 0 and total % (SUMMARY_INTERVAL * 2) == 0:
        _update_summary(session_id)

    return {
        "answer": answer,
        "session_id": session_id,
        "relevant": True,
        "route": route,
        "contexts": final_docs,
        "multi_queries": expanded.all_queries,
        "hyde_passage": expanded.hyde_passage,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/ask", response_model=ChatAskResponse)
def ask(
    session_id: str,
    body: AskRequest,
    user: User = Depends(get_current_user),
):
    """Ask a question inside a session (conversation-aware)."""
    result = ask_chat(session_id, user.id, body.question)
    return ChatAskResponse(**result)
