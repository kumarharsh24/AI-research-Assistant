from __future__ import annotations

import logging
from typing import Optional

from langchain.tools import tool
from langchain_core.retrievers import BaseRetriever


logger = logging.getLogger(__name__)


@tool("web_search")
def web_search_tool(query: str) -> str:
    """Search the internet for recent information using Tavily."""
    try:
        from tavily import TavilyClient
    except ImportError:
        return "Tavily client is not installed. Install tavily-python to use web search."

    import os

    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key or tavily_api_key == "YOUR_TAVILY_API_KEY":
        return "TAVILY_API_KEY is missing. Add it to your environment to enable web search."

    try:
        client = TavilyClient(api_key=tavily_api_key)
        result = client.search(query=query, max_results=5)
    except Exception as exc:
        logger.exception("Web search failed")
        return f"Web search failed: {exc}"

    rows = []
    for item in result.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        rows.append(f"Title: {title}\nURL: {url}\nSnippet: {content}")

    if not rows:
        return "No web results found."
    return "\n\n".join(rows)


def make_document_tool(retriever: Optional[BaseRetriever]):
    """Return a retriever tool that limits answers to document context."""

    @tool("document_search")
    def document_search_tool(query: str) -> str:
        """Find relevant details from uploaded documents."""
        if retriever is None:
            return "No document index loaded yet. Upload and process a PDF first."

        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            logger.exception("Document retrieval failed")
            return f"Document retrieval failed: {exc}"

        if not docs:
            return "No relevant document chunks were found."

        context_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "n/a")
            context_blocks.append(
                f"[Chunk {i}] Source: {source}, Page: {page}\n{doc.page_content}"
            )

        return "\n\n".join(context_blocks)

    return document_search_tool
