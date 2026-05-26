from pathlib import Path
from app.ingest import ingest_pdf

pdf_dir = Path("./data/pdfs")

for pdf_file in pdf_dir.glob("*.pdf"):
    print(f"Ingesting {pdf_file}...")
    ingest_pdf(str(pdf_file))