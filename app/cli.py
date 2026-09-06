from pathlib import Path
import argparse

from AI.projects.document_qa_rag.app.config import DATA_DIR
from AI.projects.document_qa_rag.app.loaders import load_directory, load_file
from AI.projects.document_qa_rag.app.rag import RAGEngine

def main():
    parser = argparse.ArgumentParser(description="Document Q&A RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("path", nargs="?", default=str(DATA_DIR / "documents"))

    ask = sub.add_parser("ask")
    ask.add_argument("question")

    args = parser.parse_args()
    engine = RAGEngine()

    if args.command == "ingest":
        path = Path(args.path)
        if path.is_dir():
            docs = load_directory(path)
        else:
            docs = [(path.name, load_file(path))]
        count = engine.add_documents(docs)
        print(f"Indexed {count} chunks from {len(docs)} document(s).")

    elif args.command == "ask":
        result = engine.answer(args.question)
        print("\n" + result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['source']} (chunk {source['chunk']}, score={source['score']})")

if __name__ == "__main__":
    main()
