import hashlib
import json
import re
import uuid
from pathlib import Path

import chromadb
import fitz
from FlagEmbedding import FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "cfa_docs"
MANIFEST_PATH = "./chroma_db/ingested_files.json"
MIN_CHUNK_CHARS = 80


# =========================
# MODELS / DB
# =========================

embedding_model = FlagModel(
    "BAAI/bge-small-en-v1.5",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


# =========================
# MANIFEST
# =========================

def load_manifest() -> dict:
    p = Path(MANIFEST_PATH)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2))


def file_md5(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


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
    finance_pattern = r"\b(PV|FV|NPV|IRR|WACC|EPS|P/E)\s*[=<>]"
    math_pattern = r"\b\d+\s*[\+\-\*\/]\s*\d+.*[=]"
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

def make_chunk_id(source: str, page: int, text: str) -> str:
    normalized = clean_text(text)
    key = f"{source}:{page}:{normalized}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


# =========================
# PDF EXTRACTION
# =========================

def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))
        text = "\n\n".join(
            b[4].strip()
            for b in blocks
            if b[4].strip() and len(b[4].strip()) > 10
        )
        if text:
            pages.append((i + 1, text))
    return pages


# =========================
# CHUNK BUILDING
# =========================

def make_chunk(text: str, page: int, source: str) -> dict:
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
            "source":      source,
            "page":        page,
            "type":        ",".join(types),
            "has_formula": has_formula,
            "has_table":   has_table,
        },
    }


def build_chunks(page_num: int, text: str, source: str) -> list[dict]:
    chunks = []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    buffer = []

    for p in paras:
        buffer.append(p)
        joined = "\n\n".join(buffer)

        if len(joined) > 1200:
            if len(buffer) > 1:
                chunk_text = "\n\n".join(buffer[:-1])
                for st in splitter.split_text(chunk_text):
                    if len(st) >= MIN_CHUNK_CHARS:
                        chunks.append(make_chunk(st, page_num, source))
                buffer = [buffer[-1]]
            else:
                for st in splitter.split_text(buffer[0]):
                    if len(st) >= MIN_CHUNK_CHARS:
                        chunks.append(make_chunk(st, page_num, source))
                buffer = []

    if buffer:
        remaining = "\n\n".join(buffer)
        if len(remaining) >= MIN_CHUNK_CHARS:
            chunks.append(make_chunk(remaining, page_num, source))

    return chunks


# =========================
# INGEST (single file)
# =========================

def ingest_pdf(pdf_path: str, force: bool = False) -> int:
    """
    Ingest một PDF vào ChromaDB.

    - Nếu file đã được ingest và nội dung không thay đổi → skip (trừ khi force=True).
    - Manifest chỉ được cập nhật sau khi upsert thành công,
      nên nếu crash giữa chừng thì lần sau sẽ chạy lại toàn bộ file đó.

    Args:
        pdf_path: Đường dẫn tới file PDF.
        force:    Bỏ qua manifest, luôn re-ingest.

    Returns:
        Số chunk đã upsert (0 nếu skip).
    """
    path = Path(pdf_path).resolve()
    source = path.name
    manifest = load_manifest()
    current_hash = file_md5(str(path))

    if not force and manifest.get(source) == current_hash:
        print(f"[SKIP] {source} (không thay đổi)")
        return 0

    print(f"[INGEST] {source}")

    pages = extract_pages(str(path))
    all_chunks = []
    for page_num, text in pages:
        all_chunks.extend(build_chunks(page_num, text, source))

    if not all_chunks:
        print(f"[WARN] {source}: không có chunk nào được tạo")
        return 0

    texts = [c["text"] for c in all_chunks]
    print(f"[EMBED] {len(texts)} chunks ...")

    embeddings = embedding_model.encode_corpus(texts)

    collection.upsert(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[c["metadata"] for c in all_chunks],
    )

    # Chỉ lưu manifest sau khi upsert thành công
    manifest[source] = current_hash
    save_manifest(manifest)

    print(f"[DONE] {source} → {len(all_chunks)} chunks")
    return len(all_chunks)


# =========================
# INGEST (batch / folder)
# =========================

def ingest_folder(folder_path: str, force: bool = False) -> dict[str, int]:
    """
    Ingest tất cả file PDF trong một thư mục.
    Tự động skip file đã ingest và nội dung không thay đổi.

    Args:
        folder_path: Đường dẫn thư mục chứa PDF.
        force:       Bỏ qua manifest, re-ingest tất cả.

    Returns:
        Dict {tên_file: số_chunk} cho từng file được xử lý.
    """
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] Không tìm thấy file PDF trong: {folder_path}")
        return {}

    print(f"[BATCH] Tìm thấy {len(pdf_files)} file PDF")

    results = {}
    skipped = 0

    for pdf in pdf_files:
        n = ingest_pdf(str(pdf), force=force)
        if n == 0:
            skipped += 1
        else:
            results[pdf.name] = n

    total_chunks = sum(results.values())
    print(
        f"\n[SUMMARY] {len(results)} file mới / đã thay đổi "
        f"({skipped} skip) → {total_chunks} chunks tổng cộng"
    )
    return results


# =========================
# REMOVE (xóa 1 file khỏi DB)
# =========================

def remove_pdf(source_name: str) -> int:
    """
    Xóa tất cả chunks của một file khỏi ChromaDB và manifest.

    Args:
        source_name: Tên file (vd: "document.pdf"), không phải full path.

    Returns:
        Số chunk đã xóa.
    """
    results = collection.get(where={"source": source_name})
    ids = results.get("ids", [])

    if not ids:
        print(f"[WARN] Không tìm thấy chunks cho: {source_name}")
        return 0

    collection.delete(ids=ids)

    manifest = load_manifest()
    manifest.pop(source_name, None)
    save_manifest(manifest)

    print(f"[REMOVE] {source_name} → đã xóa {len(ids)} chunks")
    return len(ids)