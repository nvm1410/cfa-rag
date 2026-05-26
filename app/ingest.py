import fitz
import re
import uuid
from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "cfa_docs"

MIN_CHUNK_CHARS = 80

# =========================
# MODELS / DB
# =========================

embedding_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    COLLECTION_NAME
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# =========================
# HEURISTICS
# =========================


def is_table(text: str) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]

    if not lines:
        return False

    pipe_lines = sum(
        1 for l in lines
        if l.count("|") >= 2 or "\t" in l
    )

    return pipe_lines > len(lines) * 0.4


def is_formula(text: str) -> bool:
    latex_pattern = r"\\frac|\\sum|\\int|\\sigma|\\mu|\\sqrt|\\times"

    finance_pattern = (
        r"\b(PV|FV|NPV|IRR|WACC|EPS|P/E)\s*[=<>]"
    )

    math_pattern = (
        r"\b\d+\s*[\+\-\*\/]\s*\d+.*[=]"
    )

    return bool(
        re.search(latex_pattern, text)
        or re.search(finance_pattern, text)
        or re.search(math_pattern, text)
    )


# =========================
# CLEANING
# =========================

def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# CHUNK IDS
# =========================

def make_chunk_id(
    source: str,
    page: int,
    text: str
) -> str:
    normalized = clean_text(text)

    key = f"{source}:{page}:{normalized}"

    return str(
        uuid.uuid5(uuid.NAMESPACE_DNS, key)
    )


# =========================
# PDF EXTRACTION
# =========================

def extract_pages(pdf_path: str):
    doc = fitz.open(pdf_path)

    pages = []

    for i, page in enumerate(doc):

        blocks = page.get_text("blocks")

        blocks.sort(
            key=lambda b: (
                round(b[1] / 10),
                b[0]
            )
        )

        text = "\n\n".join(
            b[4].strip()
            for b in blocks
            if b[4].strip()
            and len(b[4].strip()) > 10
        )

        if text:
            pages.append((i + 1, text))

    return pages


# =========================
# CHUNK BUILDING
# =========================

def make_chunk(
    text: str,
    page: int,
    source: str
) -> dict:

    text = clean_text(text)

    has_table = is_table(text)
    has_formula = is_formula(text)

    types = []

    if has_table:
        types.append("table")

    if has_formula:
        types.append("formula")

    if not types:
        types.append("text")

    return {
        "id": make_chunk_id(source, page, text),
        "text": text,
        "metadata": {
            "source": source,
            "page": page,
            "type": ",".join(types),
            "has_formula": has_formula,
            "has_table": has_table,
        }
    }


def build_chunks(
    page_num: int,
    text: str,
    source: str
) -> list:

    chunks = []

    paras = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    buffer = []

    for p in paras:

        buffer.append(p)

        joined = "\n\n".join(buffer)

        if len(joined) > 1200:

            if len(buffer) > 1:

                chunk_text = "\n\n".join(buffer[:-1])

                sub_texts = splitter.split_text(
                    chunk_text
                )

                for st in sub_texts:

                    if len(st) >= MIN_CHUNK_CHARS:
                        chunks.append(
                            make_chunk(
                                st,
                                page_num,
                                source
                            )
                        )

                buffer = [buffer[-1]]

            else:
                sub_texts = splitter.split_text(
                    buffer[0]
                )

                for st in sub_texts:

                    if len(st) >= MIN_CHUNK_CHARS:
                        chunks.append(
                            make_chunk(
                                st,
                                page_num,
                                source
                            )
                        )

                buffer = []

    if buffer:

        remaining = "\n\n".join(buffer)

        if len(remaining) >= MIN_CHUNK_CHARS:

            chunks.append(
                make_chunk(
                    remaining,
                    page_num,
                    source
                )
            )

    return chunks


# =========================
# INGEST
# =========================

def ingest_pdf(pdf_path: str) -> int:

    path = Path(pdf_path).resolve()

    source = path.name

    print(f"[INGEST] {source}")

    pages = extract_pages(str(path))

    all_chunks = []

    for page_num, text in pages:

        chunks = build_chunks(
            page_num,
            text,
            source
        )

        all_chunks.extend(chunks)

    if not all_chunks:
        print("[WARN] no chunks")
        return 0

    texts = [
        c["text"]
        for c in all_chunks
    ]

    print(f"[EMBED] {len(texts)} chunks")

    embeddings = embedding_model.encode(
        texts,
        batch_size=64,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )["dense_vecs"]

    collection.upsert(
        ids=[
            c["id"]
            for c in all_chunks
        ],

        documents=texts,

        embeddings=embeddings.tolist(),

        metadatas=[
            c["metadata"]
            for c in all_chunks
        ]
    )

    print(
        f"[DONE] {source} -> "
        f"{len(all_chunks)} chunks"
    )

    return len(all_chunks)
