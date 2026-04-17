from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: str,
    gemini_api_key: str,
    index_dir: str = "data/faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Optional[str]:
    """Load a PDF, chunk it, embed it, and store a FAISS index."""
    if not gemini_api_key:
        logger.error("Cannot ingest PDF without GEMINI_API_KEY")
        return None

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error("PDF does not exist: %s", pdf_path)
        return None

    try:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
    except Exception:
        logger.exception("Failed to load PDF: %s", pdf_path)
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        logger.error("No chunks generated from PDF: %s", pdf_path)
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key,
    )

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_dir)
    except Exception:
        logger.exception("Failed to build/save FAISS index at %s", index_dir)
        return None

    return index_dir
