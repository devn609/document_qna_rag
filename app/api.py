from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile

from app.config import DATA_DIR
from app.loaders import load_file
from app.rag import RAGEngine

app = FastAPI(title="Document Q&A RAG API", version="1.0.0")
engine = RAGEngine()

class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(engine.chunks)}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".pdf", ".docx"}:
        raise HTTPException(400, "Supported files: txt, md, csv, pdf, docx")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = Path(tmp.name)

    try:
        text = load_file(temp_path)
        count = engine.add_documents([(file.filename or "upload", text)])
    finally:
        temp_path.unlink(missing_ok=True)

    return {"filename": file.filename, "chunks_added": count}

@app.post("/ask")
def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    return engine.answer(payload.question)
