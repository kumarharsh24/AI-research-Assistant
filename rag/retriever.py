from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


logger = logging.getLogger(__name__)


def load_retriever(
    gemini_api_key: str,
    index_dir: str = "data/faiss_index",
    top_k: int = 4,
) -> Optional[BaseRetriever]:
    """Load the saved FAISS index and return a retriever."""
    if not gemini_api_key:
        logger.error("Cannot load retriever without GEMINI_API_KEY")
        return None

    index_path = Path(index_dir)
    if not index_path.exists():
        logger.warning("FAISS index path does not exist: %s", index_dir)
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key,
    )

    try:
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        logger.exception("Failed to load FAISS index from %s", index_dir)
        return None

    return vectorstore.as_retriever(search_kwargs={"k": top_k})
