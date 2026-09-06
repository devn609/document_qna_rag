import numpy as np
import pytest

from app.rag import RAGEngine, Chunk


def create_engine():
    """
    Create a RAGEngine without calling __init__.
    This prevents the test from requiring OPENAI_API_KEY.
    """
    return object.__new__(RAGEngine)


def test_split_empty_document():
    """Blank documents should produce no chunks."""
    engine = create_engine()

    result = engine._split("   ")

    assert result == []


def test_split_short_document():
    """A short document should remain a single chunk."""
    engine = create_engine()

    result = engine._split("This is a short document.")

    assert len(result) == 1
    assert result[0] == "This is a short document."


def test_search_empty_index():
    """Searching an empty index should return no results."""
    engine = create_engine()

    engine.index = None
    engine.chunks = []

    result = engine.search("What is this document about?")

    assert result == []


def test_search_returns_relevant_chunks(monkeypatch):
    """Search should return chunks in the order returned by FAISS."""
    engine = create_engine()

    engine.chunks = [
        Chunk(
            text="The company revenue was $25 million.",
            source="financial_report.pdf",
            chunk_id=0,
        ),
        Chunk(
            text="The company has 400 employees.",
            source="company_report.pdf",
            chunk_id=1,
        ),
    ]

    class FakeIndex:
        def search(self, vector, k):
            scores = np.array(
                [[0.95, 0.60]],
                dtype="float32",
            )

            ids = np.array(
                [[0, 1]],
                dtype="int64",
            )

            return scores, ids

    engine.index = FakeIndex()

    monkeypatch.setattr(
        engine,
        "_embed",
        lambda texts: np.array(
            [[1.0, 0.0]],
            dtype="float32",
        ),
    )

    results = engine.search(
        "What was the company revenue?",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0][0].text == (
        "The company revenue was $25 million."
    )

    assert results[0][1] == pytest.approx(0.95)

    assert results[1][0].text == (
        "The company has 400 employees."
    )


def test_answer_uses_retrieved_context(monkeypatch):
    """The generated answer should be based on retrieved chunks."""
    engine = create_engine()

    chunk = Chunk(
        text="The company revenue was $25 million in 2024.",
        source="annual_report.pdf",
        chunk_id=3,
    )

    monkeypatch.setattr(
        engine,
        "search",
        lambda query: [(chunk, 0.92)],
    )

    class FakeMessage:
        content = (
            "The company revenue was $25 million in 2024. "
            "[annual_report.pdf, chunk 3]"
        )

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            messages = kwargs["messages"]

            user_prompt = messages[-1]["content"]

            # Verify retrieved context was passed to the LLM.
            assert "The company revenue was $25 million" in user_prompt

            # Verify the actual question was passed.
            assert "What was the revenue?" in user_prompt

            # RAG application should use deterministic generation.
            assert kwargs["temperature"] == 0

            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    engine.client = FakeClient()

    result = engine.answer("What was the revenue?")

    assert "25 million" in result["answer"]

    assert len(result["sources"]) == 1

    assert result["sources"][0]["source"] == "annual_report.pdf"

    assert result["sources"][0]["chunk"] == 3

    assert result["sources"][0]["score"] == pytest.approx(0.92)
