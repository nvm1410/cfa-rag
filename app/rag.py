import os
import re
import json
import chromadb
import numpy as np
import httpx
import time

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from FlagEmbedding import FlagModel
from openai import OpenAI
from dotenv import load_dotenv

from app.query_expansion import expand_query, ExpandedQuery

load_dotenv()

CHROMA_PATH  = "./chroma_db"
HF_TOKEN     = os.getenv("HF_TOKEN")
RERANKER_URL = os.getenv("HF_RERANKER_URL")


# =========================================================
# RERANKER (HF Endpoint)
# =========================================================
reranker_tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-reranker-v2-m3"
)

MAX_RERANK_TOKENS    = 512
RERANKER_SPECIAL_TOKENS = 4


def truncate_for_reranker(query: str, text: str) -> str:
    query_tokens = reranker_tokenizer.encode(
        query,
        add_special_tokens=False,
    )
    budget = MAX_RERANK_TOKENS - len(query_tokens) - RERANKER_SPECIAL_TOKENS

    text_tokens = reranker_tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if len(text_tokens) <= budget:
        return text

    return reranker_tokenizer.decode(
        text_tokens[:budget],
        skip_special_tokens=True,
    )


def rerank_via_hf(
    query: str,
    texts: list[str],
    max_retries: int = 5,
    initial_wait: float = 5.0,
    batch_size: int = 32,
) -> list[float]:
    wait = initial_wait

    # split thành batches <= 32
    all_scores: list[float] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]

        for attempt in range(max_retries):
            res = httpx.post(
                RERANKER_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"query": query, "texts": batch},
                timeout=60,
            )

            if res.status_code == 503:
                print(f"[RERANKER] cold start, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                wait *= 2
                continue

            if res.status_code == 422:
                print(f"[RERANKER] 422 body: {res.text}")

            res.raise_for_status()

            results = sorted(res.json(), key=lambda x: x["index"])
            all_scores.extend([r["score"] for r in results])
            break

        else:
            raise RuntimeError("HF reranker unavailable after max retries")

    return all_scores


# =========================================================
# MODELS
# =========================================================
embedding_model = FlagModel(
    "BAAI/bge-small-en-v1.5",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,
)

llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("cfa_docs")

# =========================================================
# BM25
# =========================================================
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self):
        self.texts:     list[str]        = []
        self.metadatas: list[dict]       = []
        self.ids:       list[str]        = []
        self.bm25:      BM25Okapi | None = None

    def build(self):
        data = collection.get()

        self.texts     = data["documents"]
        self.metadatas = data["metadatas"]
        self.ids       = data["ids"]

        tokenized = [tokenize(t) for t in self.texts]
        self.bm25  = BM25Okapi(tokenized)

        print(f"[BM25] built: {len(self.texts)} chunks")

    def search(self, query: str, k: int = 20) -> list[dict]:
        if self.bm25 is None:
            self.build()

        scores  = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            {
                "id":         self.ids[i],
                "text":       self.texts[i],
                "meta":       self.metadatas[i],
                "bm25_score": float(scores[i]),
            }
            for i in top_idx
        ]


bm25_index = BM25Index()


# =========================================================
# ROUTER
# =========================================================
ROUTER_PROMPT = """
You are a query router for a CFA RAG system.

Return ONLY valid JSON.

Tasks:
1. Determine whether the query is CFA-related.
2. Determine query type.
3. Determine reasoning_mode based on what the question actually requires:
   - "rule_application": question asks whether something violates a Standard,
     is compliant, or requires applying a specific CFA rule to a scenario
   - "calculation": question requires computing a numerical answer or
     applying a formula (e.g. duration, NPV, Sharpe ratio, EPS)
   - "conceptual": question asks to explain, compare, or describe a concept
     without requiring a numerical answer
4. Decide whether HyDE should be enabled.

Rules:
- rule_application => use_hyde=false, prefer_bm25=true
- calculation      => use_hyde=true,  prefer_bm25=false
- conceptual       => use_hyde=true,  prefer_bm25=false
- Non-CFA          => relevant=false

Note: query_type reflects the CFA topic area; reasoning_mode reflects
what kind of thinking is needed. A fixed_income question can be either
"calculation" (compute modified duration) or "conceptual" (explain yield
curve theories). Let the question content decide reasoning_mode.

Allowed query_type values:
- ethics
- quant
- economics
- fra
- portfolio
- equity
- fixed_income
- derivatives
- alternative
- corporate_issuers
- general
- unrelated

Output schema:
{
  "relevant": boolean,
  "query_type": string,
  "reasoning_mode": "rule_application" | "calculation" | "conceptual",
  "use_hyde": boolean,
  "prefer_bm25": boolean,
  "difficulty": "low" | "medium" | "high"
}
"""

DEFAULT_ROUTE: dict = {
    "relevant":       True,
    "query_type":     "general",
    "reasoning_mode": "conceptual",
    "use_hyde":       False,
    "prefer_bm25":    False,
    "difficulty":     "medium",
}


def route_query(query: str) -> dict:
    try:
        res = llm.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": query},
            ],
        )
        raw  = res.choices[0].message.content
        raw  = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

    except Exception:
        return DEFAULT_ROUTE.copy()

    route = DEFAULT_ROUTE.copy()
    route.update({k: v for k, v in data.items() if k in DEFAULT_ROUTE})

    # coerce types
    route["relevant"] = bool(route["relevant"])
    if route["difficulty"] not in ("low", "medium", "high"):
        route["difficulty"] = "medium"
    if route["reasoning_mode"] not in (
        "rule_application", "calculation", "conceptual"
    ):
        route["reasoning_mode"] = "conceptual"

    return route


# =========================================================
# VECTOR SEARCH
# =========================================================
def vector_search(
    query: str,
    k:     int = 20,
    where: dict | None = None,
) -> list[dict]:

    q_emb = embedding_model.encode_queries([query])[0]

    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    ids   = res["ids"][0]
    dists = res["distances"][0]

    return [
        {
            "id":           ids[i],
            "text":         docs[i],
            "meta":         metas[i],
            "vector_score": float(1 - dists[i]),
        }
        for i in range(len(docs))
    ]


# =========================================================
# RRF
# =========================================================
def rrf(
    vec_results:  list[dict],
    bm25_results: list[dict],
    k:     int   = 60,
    w_vec: float = 1.0,
    w_bm:  float = 1.0,
) -> list[dict]:

    merged: dict[str, dict] = {}

    def add(item: dict, rank: int, weight: float):
        cid = item["id"]
        if cid not in merged:
            merged[cid] = {**item, "rrf_score": 0.0}
        merged[cid]["rrf_score"] += weight / (k + rank + 1)

    for rank, item in enumerate(vec_results):
        add(item, rank, w_vec)

    for rank, item in enumerate(bm25_results):
        add(item, rank, w_bm)

    return sorted(
        merged.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )


# =========================================================
# HYBRID SEARCH
# =========================================================
def hybrid_search(
    expanded: ExpandedQuery,
    route:    dict,
    k:        int = 80,
) -> list[dict]:

    # dedup by best score per branch, not first occurrence
    seen_vec:  dict[str, dict] = {}
    seen_bm25: dict[str, dict] = {}

    where = None

    # -------------------------
    # query strategy
    # -------------------------
    if route["use_hyde"]:
        vector_queries = list(dict.fromkeys([
            expanded.hyde_passage,
            *expanded.all_queries,
        ]))
    else:
        vector_queries = expanded.all_queries

    # -------------------------
    # vector retrieval
    # keep highest vector_score per doc across all queries
    # -------------------------
    for q in vector_queries:
        for doc in vector_search(q, k=k, where=where):
            cid = doc["id"]
            if (
                cid not in seen_vec
                or doc["vector_score"] > seen_vec[cid]["vector_score"]
            ):
                seen_vec[cid] = doc

    vec_results = sorted(
        seen_vec.values(),
        key=lambda x: x["vector_score"],
        reverse=True,
    )

    # -------------------------
    # BM25 retrieval
    # keep highest bm25_score per doc across all queries
    # -------------------------
    for q in expanded.all_queries:
        for doc in bm25_index.search(q, k=k):
            cid = doc["id"]
            if (
                cid not in seen_bm25
                or doc["bm25_score"] > seen_bm25[cid]["bm25_score"]
            ):
                seen_bm25[cid] = doc

    bm25_results = sorted(
        seen_bm25.values(),
        key=lambda x: x["bm25_score"],
        reverse=True,
    )

    # -------------------------
    # weighting
    # -------------------------
    if route["prefer_bm25"]:
        w_vec, w_bm = 1.0, 2.5
    else:
        w_vec, w_bm = 2.5, 1.0

    return rrf(
        vec_results,
        bm25_results,
        k=60,
        w_vec=w_vec,
        w_bm=w_bm,
    )[:k]


# =========================================================
# RERANK
# =========================================================
def rerank(
    query: str,
    docs:  list[dict],
    top_k: int = 15,
) -> list[dict]:

    if not docs:
        return []

    texts  = [truncate_for_reranker(query, d["text"]) for d in docs]
    scores = rerank_via_hf(query, texts)

    for i, s in enumerate(scores):
        docs[i]["rerank_score"] = float(s)

    return sorted(
        docs,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:top_k]


# =========================================================
# CONFIDENCE GATE
# =========================================================
def retrieval_confident(docs: list[dict]) -> bool:
    if not docs:
        return False

    scores = [d["rerank_score"] for d in docs]
    top    = scores[0]

    # HF reranker trả 0-1
    if top < 0.01:
        return False

    if len(docs) > 1:
        mean = sum(scores) / len(scores)
        gap  = top - scores[1]
        if top < mean + 0.1 and gap < 0.05:
            return False

    return True


# =========================================================
# CONTEXT
# =========================================================
def build_context(docs: list[dict]) -> str:
    parts = []

    for i, d in enumerate(docs, 1):
        src  = d["meta"].get("source", "unknown")
        page = d["meta"].get("page", "?")
        parts.append(
            f"[{i}] (Source: {src}, p.{page})\n"
            f"{d['text']}"
        )

    return "\n\n---\n\n".join(parts)


# =========================================================
# SYSTEM PROMPTS
# =========================================================
SYSTEM_PROMPT_RULE_APPLICATION = """
You are a CFA expert tutor specializing in Ethics and Professional Standards.

Rules:
- Read ALL context chunks carefully before answering.
- First identify which Standard(s) directly govern the facts in the question.
- If a context chunk contains a rule that matches the facts, apply it and
  state a clear conclusion. Do NOT introduce uncertainty that is not present
  in the question itself.
- Only say "insufficient information" when a required fact is genuinely
  absent from both the question and all context chunks.
- If MCQ options exist, explicitly identify the correct option.
- Always cite source and page.
"""

SYSTEM_PROMPT_CALCULATION = """
You are a CFA expert tutor specializing in quantitative analysis.

Rules:
- Use ONLY the provided context for formulas and theory.
- State the formula before applying it.
- Show all calculation steps explicitly, including units.
- Round only at the final step unless instructed otherwise.
- If MCQ options exist, compute the answer and match to the correct option;
  if your result is slightly off, check rounding and recompute before
  concluding the answer is not among the options.
- Always cite source and page.
"""

SYSTEM_PROMPT_CONCEPTUAL = """
You are a CFA expert tutor.

Rules:
- Use ONLY the provided context.
- Explain the mechanism or concept clearly; use examples from the context
  if available.
- If comparing two concepts, structure the answer with clear distinction
  between them.
- If insufficient information, say so explicitly.
- If MCQ options exist, explicitly identify the correct option and explain
  why the others are incorrect.
- Always cite source and page.
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "rule_application": SYSTEM_PROMPT_RULE_APPLICATION,
    "calculation":      SYSTEM_PROMPT_CALCULATION,
    "conceptual":       SYSTEM_PROMPT_CONCEPTUAL,
}


def get_system_prompt(route: dict) -> str:
    mode = route.get("reasoning_mode", "conceptual")
    return _SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPT_CONCEPTUAL)


# =========================================================
# PER-MODE PIPELINE PARAMS
# =========================================================

# number of multi-queries to generate per reasoning_mode:
# rule_application => 0 (off): ethics needs exact Standard match,
#                              multi-query only adds noise
# calculation      => 1: one reformulation to catch formula chunks
# conceptual       => 3: concepts appear under varied phrasings
_N_MULTI: dict[str, int] = {
    "rule_application": 0,
    "calculation":      1,
    "conceptual":       3,
}

# number of final chunks passed to generation:
# rule_application gets more to compensate for reranker misordering
_FINAL_DOCS_K: dict[str, int] = {
    "rule_application": 10,
    "calculation":       6,
    "conceptual":        6,
}


# =========================================================
# MAIN PIPELINE
# =========================================================
def ask_rag(query: str) -> dict:

    # -------------------------------------------------
    # 1. ROUTING
    # -------------------------------------------------
    route = route_query(query)

    if not route["relevant"]:
        return {
            "answer": (
                "This question does not appear "
                "to be related to the CFA curriculum."
            ),
            "route": route,
        }

    difficulty     = route["difficulty"]
    reasoning_mode = route["reasoning_mode"]

    # -------------------------------------------------
    # 2. QUERY EXPANSION
    # n_multi and use_hyde both driven by reasoning_mode / router
    # -------------------------------------------------
    n_multi = _N_MULTI.get(reasoning_mode, 3)

    expanded = expand_query(
        query,
        n_multi=n_multi,
        use_hyde=route["use_hyde"],
    )

    # -------------------------------------------------
    # 3. RETRIEVAL SIZE
    # -------------------------------------------------
    if difficulty == "high":
        retrieval_k = 100
    elif difficulty == "medium":
        retrieval_k = 60
    else:
        retrieval_k = 40

    # -------------------------------------------------
    # 4. HYBRID SEARCH
    # -------------------------------------------------
    candidates = hybrid_search(expanded, route, k=retrieval_k)

    # -------------------------------------------------
    # 5. RERANK
    # scale top_k with difficulty
    # -------------------------------------------------
    rerank_top_k = 20 if difficulty == "high" else 15

    reranked = rerank(query, candidates, top_k=rerank_top_k)

    # -------------------------------------------------
    # 6. CONFIDENCE GATE
    # -------------------------------------------------
    if not retrieval_confident(reranked):
        return {
            "answer": (
                "I could not find sufficiently "
                "relevant context to answer confidently."
            ),
            "route":         route,
            "contexts":      reranked[:5],
            "multi_queries": expanded.all_queries,
            "hyde_passage":  expanded.hyde_passage,
        }

    # -------------------------------------------------
    # 7. FINAL CONTEXT
    # -------------------------------------------------
    final_k    = _FINAL_DOCS_K.get(reasoning_mode, 6)
    final_docs = reranked[:final_k]
    context    = build_context(final_docs)

    # -------------------------------------------------
    # 8. GENERATION
    # -------------------------------------------------
    system_prompt = get_system_prompt(route)

    res = llm.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                ),
            },
        ],
    )

    return {
        "answer":        res.choices[0].message.content,
        "route":         route,
        "contexts":      final_docs,
        "multi_queries": expanded.all_queries,
        "hyde_passage":  expanded.hyde_passage,
    }