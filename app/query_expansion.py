# query_expansion.py

from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

_llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_HYDE_PROMPT = """
You are generating a hypothetical CFA curriculum passage for retrieval purposes.

Given a user question, write a concise CFA-style textbook passage that is likely
to contain the information needed to answer the question.

Requirements:
- 3-5 sentences maximum
- Use precise CFA and financial terminology
- Include formulas or key relationships if relevant
- Focus on concepts directly related to the question
- Avoid introductions, filler, and meta commentary
- Do not explain unnecessary background concepts
- Write in the style of an official CFA curriculum text

Return ONLY the passage.

Question:
{query}
"""

_MULTI_QUERY_PROMPT = """
You are generating retrieval queries for a CFA RAG system.

Given a user question, generate {n} alternative search queries.

Requirements:
- Each line must be a search query, not an answer.
- Use concise CFA terminology.
- Use different semantic angles and related terminology.
- Preserve the original intent.
- Prefer short keyword-rich queries over conversational questions.
- Do not explain concepts.
- Do not provide examples or calculations.

Return ONLY the queries, one per line.
No numbering or extra text.

User question:
{query}
"""


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def generate_hyde_passage(query: str) -> str:
    """
    HyDE: generate một đoạn văn giả theo style CFA rồi dùng
    embedding của đoạn đó thay vì embedding của câu hỏi gốc.
    Lý do: embedding space của passage gần corpus hơn là câu hỏi.
    """
    resp = _llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        temperature=0.3,   # chút creativity để cover nhiều góc, không cần deterministic
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    """
    Multi-query: expand câu hỏi gốc thành n cách hỏi khác nhau.
    Giúp BM25 bắt được các thuật ngữ mà query gốc không dùng.
    """
    resp = _llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": _MULTI_QUERY_PROMPT.format(query=query, n=n)}],
        temperature=0.4,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    return queries[:n]  # guard — đề phòng LLM trả về nhiều hơn n


@dataclass
class ExpandedQuery:
    original: str
    hyde_passage: str
    multi_queries: list[str]
    all_queries: list[str]      # original + multi


def expand_query(
    query: str,
    n_multi: int = 3,
    use_hyde: bool = False,
) -> ExpandedQuery:
    """
    Chỉ gọi generate_hyde_passage khi use_hyde=True.
    """
    hyde = generate_hyde_passage(query) if use_hyde else ""
    multi = generate_multi_queries(query, n=n_multi) if n_multi > 0 else []

    return ExpandedQuery(
        original=query,
        hyde_passage=hyde,
        multi_queries=multi,
        all_queries=[query] + multi,
    )
