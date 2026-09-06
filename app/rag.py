from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from app.config import (
    CHAT_MODEL, EMBEDDING_MODEL, OPENAI_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, INDEX_DIR
)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class RAGEngine:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index = None
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self):
        index_path = INDEX_DIR / "faiss.index"
        chunks_path = INDEX_DIR / "chunks.pkl"
        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with chunks_path.open("rb") as f:
                self.chunks = pickle.load(f)

    def _split(self, text: str) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vectors = np.array([x.embedding for x in response.data], dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors

    def add_documents(self, documents: list[tuple[str, str]]) -> int:
        new_chunks = []
        for source, text in documents:
            for i, chunk in enumerate(self._split(text)):
                new_chunks.append(Chunk(chunk, source, i))

        if not new_chunks:
            return 0

        vectors = self._embed([c.text for c in new_chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        self._save()
        return len(new_chunks)

    def _save(self):
        faiss.write_index(self.index, str(INDEX_DIR / "faiss.index"))
        with (INDEX_DIR / "chunks.pkl").open("wb") as f:
            pickle.dump(self.chunks, f)

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        vector = self._embed([query])
        scores, ids = self.index.search(vector, min(top_k, len(self.chunks)))
        return [(self.chunks[i], float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]

    def answer(self, query: str) -> dict:
        results = self.search(query)
        if not results:
            return {
                "answer": "I don't have any indexed documents yet. Upload or add documents first.",
                "sources": []
            }

        context = "\n\n".join(
            f"[Source: {c.source}, chunk {c.chunk_id}]\n{c.text}"
            for c, _ in results
        )

        system = """You are a document question-answering assistant.
Answer using ONLY the supplied context. If the context does not contain
enough information, say you don't know based on the indexed documents.
Do not invent facts. Cite sources inline using [source, chunk N]."""

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": [
                {"source": c.source, "chunk": c.chunk_id, "score": round(score, 4), "text": c.text}
                for c, score in results
            ],
        }
