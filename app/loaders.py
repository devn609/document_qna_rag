from pathlib import Path
import csv

def load_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return "\n".join(" | ".join(row) for row in csv.reader(f))

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")

def load_directory(directory: Path):
    allowed = {".txt", ".md", ".csv", ".pdf", ".docx"}
    documents = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            text = load_file(path)
            if text.strip():
                documents.append((str(path.relative_to(directory)), text))
    return documents
