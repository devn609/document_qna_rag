# Document Q&A — RAG

A complete Python RAG application for asking questions over PDF, DOCX, TXT, Markdown, and CSV documents.

## Architecture

```text
Documents
   ↓
Document loaders
   ↓
Chunking with overlap
   ↓
OpenAI embeddings
   ↓
FAISS vector index
   ↓
Top-K semantic retrieval
   ↓
LLM prompt with retrieved context
   ↓
Grounded answer + sources
```

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```env
OPENAI_API_KEY=your_api_key_here
```

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Option 1: Web UI

```bash
streamlit run app/web.py
```

Open the URL Streamlit prints, upload documents, click **Index documents**, then ask questions.

## Option 2: CLI

Put documents under `data/documents/` and run:

```bash
python -m app.cli ingest
python -m app.cli ask "What is the main conclusion?"
```

You can also ingest a specific file:

```bash
python -m app.cli ingest ./my_document.pdf
```

## Option 3: REST API

Start:

```bash
uvicorn app.api:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ingest a document:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@./my_document.pdf"
```

Ask:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the key findings?"}'
```

## RAG behavior

The bot:

- splits documents into overlapping chunks;
- creates embeddings with `text-embedding-3-small`;
- stores normalized vectors in a local FAISS index;
- retrieves the most relevant chunks;
- sends only retrieved context to the chat model;
- instructs the model not to invent information;
- returns the retrieved source chunks for inspection.

## Reset the index

Delete:

```text
data/index/faiss.index
data/index/chunks.pkl
```

Then ingest again.

## Tests

```bash
pytest -v
```

## Production notes

For a production deployment, consider:

- persistent object/blob storage for documents;
- a managed vector database or PostgreSQL + pgvector;
- authentication and authorization;
- document-level access controls;
- asynchronous ingestion;
- OCR for scanned PDFs;
- reranking;
- query rewriting;
- evaluation datasets and retrieval/answer quality metrics;
- rate limiting and request logging;
- secret management rather than `.env`.
