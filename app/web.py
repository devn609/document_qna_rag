import streamlit as st
from pathlib import Path
import tempfile

from app.loaders import load_file
from app.rag import RAGEngine

def main():
    st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
    st.title("Document Q&A")
    st.caption("RAG-powered question answering over your documents")

    @st.cache_resource
    def get_engine():
        return RAGEngine()

    engine = get_engine()

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("Index documents"):
            total = 0
            for file in uploaded:
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file.getvalue())
                    path = Path(tmp.name)
                try:
                    total += engine.add_documents([(file.name, load_file(path))])
                finally:
                    path.unlink(missing_ok=True)
            st.success(f"Indexed {total} chunks.")

    question = st.text_input("Ask a question about your documents")
    if question:
        with st.spinner("Searching and generating answer..."):
            result = engine.answer(question)
        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Retrieved sources"):
            for source in result["sources"]:
                st.markdown(f"**{source['source']} — chunk {source['chunk']} — score {source['score']}**")
                st.write(source["text"])

if __name__ == "__main__":
    main()